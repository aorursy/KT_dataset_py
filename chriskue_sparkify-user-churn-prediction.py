# install PySpark for the project
!pip install pyspark
# import relevant libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from time import time
from functools import reduce

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf
from pyspark.sql.types import IntegerType, DateType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import MinMaxScaler
from pyspark.mllib.evaluation import MulticlassMetrics

%matplotlib inline
# create a Spark session
spark = SparkSession.builder.appName('Sparkify').getOrCreate()
# load the file
path = "../input/sparkify/mini_sparkify_event_data.json"
df = spark.read.json(path)
df.persist()
# show the schema of the dataset
df.printSchema()
# number of records and columns in the dataset
print((df.count(), len(df.columns)))
# show the first 5 rows in the dataset
df.show(5)
# get further informations of all columns / features
for col in df.columns:
    df.describe([col]).show()
    df.select([col]).distinct().show()
# count empty values in every column
for col in df.columns:
    print(col,  df.filter((df[col].isNull()) | (df[col] == "")).count())
# removing empty values from feature userId
df_clean = df.dropna(how = 'any', subset = ['userId'])
df_clean = df_clean.filter(df_clean['userId'] != '')
df_clean.persist()
# number of records in the cleaned dataset
df_clean.count()
# transforming the ts and registation time format
get_date = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime("%Y-%m-%d %H:%M:%S"))
df_clean = df_clean.withColumn('event_time', get_date('ts'))
df_clean = df_clean.withColumn('registration_time', get_date('registration'))
df_clean.select('event_time').show(5)
# time range in the data set   
print(df_clean.agg({'event_time': 'min'}).collect()[0][0])
print(df_clean.agg({'event_time': 'max'}).collect()[0][0])
# get the day opf month of the timeseries
get_day = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).day)
df_clean = df_clean.withColumn('day_of_month', get_day('ts'))
df_clean.select('day_of_month').show(5)
# extract Operating System
map = {'Compatible': 'Windows', 'Ipad': 'iPad', 'Iphone': 'iPhone', 'Macintosh': 'Mac',  
       'Windows nt 5.1': 'Windows', 'Windows nt 6.0': 'Windows', 'Windows nt 6.1': 'Windows', 
       'Windows nt 6.2': 'Windows',  'Windows nt 6.3': 'Windows', 'X11': 'Linux'}

get_operating_sys = udf(lambda x: map[re.findall('\(([^\)]*)\)', x)[0].split(';')[0].capitalize()])
df_clean = df_clean.withColumn('operating_system', get_operating_sys(df_clean.userAgent))
df_clean.select('operating_system').distinct().show(5)
# extract state from location
get_location = udf(lambda x:x[-2:])
df_clean = df_clean.withColumn('location_state', get_location(df_clean.location))
df_clean.select('location_state').distinct().show(5)
#show the page content, to see what actions can be used to get informations about churn
df_clean.select('page').distinct().show()
# create feature "downgrade"
downgrade_event = udf(lambda x: 1 if x == 'Submit Downgrade' else 0, IntegerType())
df_clean = df_clean.withColumn('downgrade_event', downgrade_event('page'))
df_clean = df_clean.withColumn('downgrade', max('downgrade_event').over(Window.partitionBy('userId')))
# create feature "churn"
churn_event = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0, IntegerType())
df_clean = df_clean.withColumn('churn_event', churn_event('page'))
df_clean = df_clean.withColumn('churn', max('churn_event').over(Window.partitionBy('userId')))
num_churn = df_clean.dropDuplicates(['userId', 'churn']).groupby(['churn']).count().toPandas()

ax = sns.barplot(x='churn', y='count', data=num_churn);
plt.xlabel('churn')
plt.ylabel('count')
plt.title('How many user churned?');
# churn and downgrade pattern between genders
churn_gender = df_clean.dropDuplicates(['userId', 'gender']).groupby(['churn', 'gender']).count().sort('churn').toPandas()
downgrade_gender = df_clean.dropDuplicates(['userId', 'gender']).groupby(['downgrade', 'gender']).count().sort('downgrade').toPandas()

fig, (ax1, ax2) = plt.subplots(1,2)

sns.barplot(x='churn', y='count', hue='gender', data=churn_gender, ax=ax1) 
ax1.set_xlabel('churn')
ax1.set_ylabel('count')
ax1.legend(title='gender', loc='best')
ax1.set_title('User has churned')

sns.barplot(x='downgrade', y='count', hue='gender', data=downgrade_gender, ax=ax2)
ax2.set_xlabel('downgrade')
ax2.set_ylabel('count')
ax2.legend(title='gender', loc='best')
ax2.set_title('User has downgraded')

fig.show();
# level (account) of the user
account_churn = df_clean.dropDuplicates(['userId', 'gender']).groupby(['gender', 'level']).count().toPandas()

ax = sns.barplot(x='level', y='count',hue='gender', data=account_churn);
plt.xlabel('account')
plt.ylabel('count')
plt.title('In which accout does the user churn?')
plt.legend(title='Gender', loc='best');
# lifetime (membership) of the users
lifetime = df_clean.select('userId','registration','ts','churn')
lifetime = lifetime.withColumn('lifetime',(df_clean.ts - df_clean.registration)/(1000*3600*24)).groupBy('userId','churn').agg({'lifetime':'max'})
lifetime = lifetime.withColumnRenamed('max(lifetime)','lifetime').select('userId', 'churn', 'lifetime').toPandas()

ax = sns.boxplot(x='lifetime', y='churn', data=lifetime, orient='h');
plt.xlabel('days member');
plt.ylabel('churned');
plt.title('How long does a user use Sparkify before churn?');
# locations (state) of the users
location_state = df_clean.dropDuplicates(['userId', 'location_state']).groupby(['location_state', 'churn']).count().toPandas()

fig = plt.figure(figsize=(20, 6))

ax = sns.barplot(x='location_state', y='count', hue='churn', data=location_state);
plt.xlabel('lcation (state)')
plt.ylabel('count')
plt.title('User count by state');
plt.legend(title='churn', loc='best');
# page actions of the users
page_churn = df_clean.dropDuplicates(['userId', 'page']).groupby(['page', 'churn']).count().toPandas()

fig = plt.figure(figsize=(20, 6))

ax = sns.barplot(x='page', y='count',hue='churn', data=page_churn)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.xlabel('page activities')
plt.ylabel('count')
plt.title('Activities of the user')
plt.legend(title='Churn', loc='best');
# churn by operationg system
os_churn = df_clean.dropDuplicates(['userId', 'operating_system']).groupby(['churn', 'operating_system']).count().toPandas()

ax = sns.barplot(x='operating_system', y='count',hue='churn', data=os_churn);
plt.xlabel('operating System')
plt.ylabel('count')
plt.title('Operation System of the users')
plt.legend(title='Churn', loc='best');
# sum of listened music of the users
sum_listened = df_clean.select('userID','length','churn').groupby(['userID','churn']).sum().withColumnRenamed('sum(length)', 'listen_time').toPandas()

fig = plt.figure(figsize=(20, 6))

ax = sns.boxplot(x='listen_time', y='churn', data=sum_listened, orient='h');
plt.xlabel('listened music');
plt.ylabel('churned');
plt.title('How long does the users listen to music');
# difference in the day - by page "Home"
day_home = df_clean.filter(df_clean.page == 'Home').groupby(['churn', 'day_of_month']).count().orderBy(df_clean.day_of_month.cast('int')).toPandas()

fig = plt.figure(figsize=(20, 6))

ax = sns.barplot(x='day_of_month', y='count', hue='churn', data=day_home, order=day_home.day_of_month);
plt.xlabel('day of month');
plt.ylabel('count');
plt.title('Activity (Login) of the users in a month');
# gender of the user
gender = df_clean.select(['userId', 'gender']).dropDuplicates(['userId']).replace(['F', 'M'], ['0', '1'], 'gender')
df_gender = gender.withColumn('gender', gender.gender.cast('int'))
df_gender.show(5)
# level of the user
payment = df_clean.select(['userId', 'level']).dropDuplicates(['userId']).replace(['paid', 'free'], ['0', '1'], 'level')
df_payment = payment.withColumn('level', payment.level.cast('int'))
df_payment.show(5)
# did the user downgrade
downgrade = df_clean.select(['userId','downgrade']).dropDuplicates(['userId']) 
df_downgrade = downgrade.withColumn('downgrade', downgrade.downgrade.cast('int'))
df_downgrade.show(5)
# did the user churn
churn = df_clean.select(['userId','churn']).dropDuplicates(['userId'])
df_churn = churn.withColumn('churn', churn.churn.cast('int'))
df_churn.show(5) 
# number of songs the user listened to in total
num_songs = df_clean.select('userID','song').groupBy('userID').count()
num_songs.show(5) 
# number of Thumbs-Up/Down
num_thumbs_up = df_clean.select('userID','page').where(df_clean.page == 'Thumbs Up').groupBy('userID').count().withColumnRenamed('count', 'num_thumbs_up') 
print(num_thumbs_up.show(5))
num_thumbs_down = df_clean.select('userID','page').where(df_clean.page == 'Thumbs Down').groupBy('userID').count().withColumnRenamed('count', 'num_thumbs_down') 
print(num_thumbs_down.show(5)) 
# number of songs added to playlist
num_playlist = df_clean.select('userID','page').where(df_clean.page == 'Add to Playlist').groupBy('userID').count().withColumnRenamed('count', 'num_playlist')
num_playlist.show(5) 
# number of friends added
num_friends = df_clean.select('userID','page').where(df_clean.page == 'Add Friend').groupBy('userID').count().withColumnRenamed('count', 'num_friend')
num_friends.show(5)
# total length of listening
sum_listened = df_clean.select('userID','length').groupBy('userID').sum().withColumnRenamed('sum(length)', 'sum_listened')
sum_listened.show(5) 
# Number of songs listened per session
av_song_session = df_clean.where('page == "NextSong"').groupby(['userId', 'sessionId']).count().groupby(['userId']).agg({'count':'avg'}).withColumnRenamed('avg(count)', 'av_song_session')
av_song_session.show(5)
# number of artists listened to
num_artists = df_clean.filter(df_clean.page=="NextSong").select(['userId', 'artist']).dropDuplicates().groupby('userId').count().withColumnRenamed('count', 'num_artists') 
num_artists.show(5) 
# time since registration in days
days_member = df_clean.select('userId','ts','registration').withColumn(
    'days_member',((df_clean.ts - df_clean.registration)/1000/3600/24)).groupBy('userId').agg(
    {'days_member':'max'}).withColumnRenamed('max(days_member)','days_member') 
days_member.show(5) 
# session count per user
num_session = df_clean.select('userId', 'sessionId').dropDuplicates().groupby('userId').count().withColumnRenamed('count', 'num_sessions') 
num_session.show(5) 
# duration of the session
session_start = df_clean.groupBy('userId', 'sessionId').min('ts').withColumnRenamed('min(ts)', 'start')
session_end = df_clean.groupBy('userId', 'sessionId').max('ts').withColumnRenamed('max(ts)', 'end')
dur_session = session_start.join(session_end, ['userId', 'sessionId'])
dur_session = dur_session.select('userId', 'sessionId', ((dur_session.end-dur_session.start)/(1000*60*60)).alias('dur_session'))
dur_session.show(5)
# list all the features for the model data set
model_features = [df_payment, df_downgrade, df_gender, num_songs,num_thumbs_up, num_thumbs_down, 
                  num_friends, num_playlist, sum_listened, av_song_session, num_artists, 
                  days_member, num_session, dur_session]
# generate the data set for the model 
df_final = df_churn
df_final.persist()

for i, feature_to_join in enumerate(model_features):
    df_final = df_final.join(feature_to_join,'userID','outer')
# drop userID as no longer needed and fill all remaining missing values with 0
df_final = df_final.drop('userID') 
df_final = df_final.na.fill(0)
# show schema of the final data set
df_final.printSchema()
# number of records and columns in the dataset
print((df_final.count(), len(df_final.columns)))
# rename feature churn for the model
df_final = df_final.withColumnRenamed("churn","label")
# vector assembler
assembler = VectorAssembler(inputCols=df_final.columns[1:], outputCol="features")
data = assembler.transform(df_final)
data
# standard scaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True)
scalerModel = scaler.fit(data)
data = scalerModel.transform(data)
# drop column "features" - as no longer needed
data = data.drop("features")
# split the data into train and test 
train, test = data.randomSplit([0.8, 0.2], seed=42)
def get_model_metrics(model):
    """
    Classification model evaluation
    >> Calculate the Accuracy and F1-Score from the model
    
    While there are many different types of classification algorithms, the evaluation of 
    classification models all share similar principles. In a supervised classification problem, 
    there exists a true output and a model-generated predicted output for each data point. 
    For this reason, the results for each data point can be assigned to one of four categories:

    True Positive (TP) - label is positive and prediction is also positive
    True Negative (TN) - label is negative and prediction is also negative
    False Positive (FP) - label is negative but prediction is positive
    False Negative (FN) - label is positive but prediction is negative

    These four numbers are the building blocks for most classifier evaluation metrics. 
    A fundamental point when considering classifier evaluation is that pure accuracy 
    (i.e. was the prediction correct or incorrect) is not generally a good metric. 
    The reason for this is because a dataset may be highly unbalanced. For example, 
    if a model is designed to predict fraud from a dataset where 95% of the data points 
    are not fraud and 5% of the data points are fraud, then a naive classifier that 
    predicts not fraud, regardless of input, will be 95% accurate. For this reason, 
    metrics like precision and recall are typically used because they take into account 
    the type of error. In most applications there is some desired balance between precision 
    and recall, which can be captured by combining the two into a single metric, 
    called the F-measure.
    
    Source: https://spark.apache.org/docs/2.2.0/mllib-evaluation-metrics.html
    
    Source of code: https://stackoverflow.com/questions/41032256/
    get-same-value-for-precision-recall-and-f-score-in-apache-spark-logistic-regres
    """
    
    true_positive = model.where((model.label==1) & (model.prediction==1)).count()
    true_negtive = model.where((model.label==0) & (model.prediction==0)).count()
    
    false_positive = model.where((model.label==0) & (model.prediction==1)).count()
    false_negative = model.where((model.label==1) & (model.prediction==0)).count()
        
    accuracy = (true_positive + true_negtive) / model.count()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    f1 = 2.0 * (precision * recall) / (precision + recall)
    
    print('Accuracy: ', round(accuracy, 2))
    print('Precision: ', round(precision, 2))
    print('Recall: ', round(recall, 2))
    print('F1-Score: ', round(f1, 2))
lr =  LogisticRegression(featuresCol='scaled_features', labelCol='label', maxIter=10, regParam=0.0, elasticNetParam=0)
lr_model = lr.fit(train)
results_lr = lr_model.transform(test)

get_model_metrics(results_lr)
feature_coef = lr_model.coefficients.values.tolist()
feature_coef_df = pd.DataFrame(list(zip(df_final.columns[1:], feature_coef)), columns=['Feature', 'Coefficient'])\
        .sort_values('Coefficient', ascending=False)

plt.figure(figsize=(20,8))
sns.barplot(x='Feature', y='Coefficient', data=feature_coef_df)
plt.title('Feature importance for Logistic Rgression')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.xticks(rotation=45, ha='right')
plt.show()
dt = DecisionTreeClassifier(labelCol="label", featuresCol="scaled_features")
dt_model = dt.fit(train)
results_dt = dt_model.transform(test)

get_model_metrics(results_dt)
feature_ind = dt_model.featureImportances.indices.tolist()
feature_name = [df_final.columns[1:][ind] for ind in feature_ind]
feature_coef = dt_model.featureImportances.values.tolist()
feature_coef_df = pd.DataFrame(list(zip(df_final.columns[1:], feature_coef)), columns=['Feature', 'Coefficient'])\
            .sort_values('Coefficient', ascending=False)

plt.figure(figsize=(20,8))
sns.barplot(x='Feature', y='Coefficient', data=feature_coef_df)
plt.title('Feature importance for DecisionTreeClassifier')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.show()
rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features")
rf_model = rf.fit(train)
results_rf = rf_model.transform(test)

get_model_metrics(results_rf)
feature_ind = rf_model.featureImportances.indices.tolist()
feature_name = [df_final.columns[1:][ind] for ind in feature_ind]
feature_coef = rf_model.featureImportances.values.tolist()
feature_coef_df = pd.DataFrame(list(zip(df_final.columns[1:], feature_coef)), columns=['Feature', 'Coefficient'])\
            .sort_values('Coefficient', ascending=False)

plt.figure(figsize=(20,8))
sns.barplot(x='Feature', y='Coefficient', data=feature_coef_df)
plt.title('Feature importance for Random Forest')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.xticks(rotation=45, ha='right')
plt.show()
gbt = GBTClassifier(labelCol="label", featuresCol="scaled_features")
gbt_model = gbt.fit(train)
results_gbt = gbt_model.transform(test)

get_model_metrics(results_gbt)
feature_ind = gbt_model.featureImportances.indices.tolist()
feature_name = [df_final.columns[1:][ind] for ind in feature_ind]
feature_coef = gbt_model.featureImportances.values.tolist()
feature_coef_df = pd.DataFrame(list(zip(df_final.columns[1:], feature_coef)), columns=['Feature', 'Coefficient'])\
            .sort_values('Coefficient', ascending=False)

plt.figure(figsize=(20,8))
sns.barplot(x='Feature', y='Coefficient', data=feature_coef_df)
plt.title('Feature importance for GBTClassifier')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.xticks(rotation=45, ha='right')
plt.show()