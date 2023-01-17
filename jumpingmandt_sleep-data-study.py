# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# This CSV is separated by the "delimiter=;"
df = pd.read_csv('/kaggle/input/sleep-data/sleepdata.csv',delimiter=";")
df
df.info()
import seaborn as sns
#check the null part in the whole data set, red part is missing data
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
import time
import datetime

df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])
df['Time in bed'] = df['End'] - df['Start']
df['Time in bed'] = df['Time in bed'].astype('timedelta64[s]')
df['Sleep quality'] = df['Sleep quality'].apply(lambda x: np.nan if x in ['-'] else x[:-1]).astype(int)
df['Start time'] = pd.Series([val.time() for val in df['Start']])
df['End time'] = pd.Series([val.time() for val in df['End']])
df['Start time in second'] = df['Start time'].apply(lambda x: (x.hour*60+x.minute)*60 + x.second)
df['End time in second'] = df['End time'].apply(lambda x: (x.hour*60+x.minute)*60 + x.second)
import matplotlib.pyplot as plt
# visualisation of this correlation
fig = plt.figure(figsize = (12,10))
r = sns.heatmap(df.corr(),cmap='Oranges')
# set title
r.set_title('Correlation')
df.corr()['Sleep quality'].sort_values(ascending = False)
# So we can replace these symbols with positive and negative number 
df['Wake up'] = df['Wake up'].replace({':)':2, ':|':1, ':(':0})
df2 = df[["Sleep quality", "Wake up", "Time in bed", "Start time in second", "End time in second","Activity (steps)"]]
# Drop the NaN elements
df2 = df2.dropna()
# convert the type from object to interger
df2['Wake up'] = df2['Wake up'].astype('int')
# Let's check the correlations of features to the "sleep quality"
df2.corr()['Sleep quality'].sort_values(ascending = False)
df3 = df[["Sleep quality", "Wake up","Heart rate","Time in bed", "Start time in second", "End time in second"]]
# Drop the NaN elements
df3 = df3.dropna()
df3['Wake up'] = df3['Wake up'].astype('int')
# Let's check the correlations of features to the "sleep quality"
df3.corr()['Sleep quality'].sort_values(ascending = False)
# visualisation of this correlation
fig = plt.figure(figsize = (12,10))
r = sns.heatmap(df3.corr(),cmap='Oranges')
# set title
r.set_title('Correlation')
# Pairplot
sns.pairplot(df2, hue='Wake up')
# Joint plot of features "Sleep quality" and "Time in bed" with unit second.
sns.jointplot(x='Sleep quality',y='Time in bed',data=df,color='blue',kind = 'kde')
# The average of "Time in bed"

print ('The average time in bed of these users is :', df['Time in bed'].mean(), 'second')
print ('The average time in bed of these users is :', df['Time in bed'].mean()/3600, 'hour')
# The Histogram of Start time and End time
plt.figure(figsize=(10,6))
df['Start in hour'] = df['Start time in second'].apply(lambda x: x/3600)
df['End in hour'] = df['End time in second'].apply(lambda x: x/3600)
df['Start in hour'].hist(alpha=0.5,color='blue',label='Start Time',bins=50)
df['End in hour'].hist(alpha=0.5,color='red',label='End Time',bins=50)
plt.legend()
plt.xlim((0, 24)) 
plt.xticks(np.arange(0, 25, 1))
plt.xlabel('Hour in a day')
plt.ylabel('Count')
# The Histogram of Steps
plt.figure(figsize=(10,6))
df['Activity (steps)'].hist(alpha=0.5,color='green',label='Steps',bins=50)
plt.legend()

plt.xlabel('Steps')
plt.ylabel('Count')
# Joint plot of features "Sleep quality" and "Activity" with unit second.
sns.jointplot(x='Sleep quality',y='Activity (steps)',data=df,color='red',kind = 'kde')
# Drop the non-meaning value of steps (0)

df_new = df[df['Activity (steps)'] != 0]
df_new
# Let's check the correlations of features to the "sleep quality"
df_new.corr()['Sleep quality'].sort_values(ascending = False)
# Scatter plot
plt.figure(figsize=(10,6))
plt.scatter(df_new['Sleep quality'],df_new['Activity (steps)'], c="g", alpha=0.5, marker=r'$\clubsuit$',
            label="Sleep quality vs. Steps")
plt.xlabel("Sleep quality")
plt.ylabel("Steps during the day")
plt.legend(loc='upper left')
plt.show()
# We use features of "Time in bed","Start time in second", "End time in second" and "Activity (steps)" to predict the feature "Sleep quality"
# We choose to use df
X = df[['Time in bed', 'Start time in second','End time in second','Activity (steps)']].values
y = df['Sleep quality'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
# In order to normalize the features, it is better to use MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.score(X_test,y_test)

print('test accuracy:', lm.score(X_test,y_test))
from sklearn.neighbors import KNeighborsClassifier

error_rate =[]
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,20),error_rate, color ='red',linestyle='dashed',marker='v',
        markerfacecolor = 'blue', markersize=10)
plt.title('Error Rate vs. K value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=14) # why 5 is because of Elbow method
knn.fit(X_train,y_train)
print('test accuracy:', knn.score(X_test,y_test))
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
print('test accuracy:', logmodel.score(X_test,y_test))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
print('test accuracy:', dtree.score(X_test,y_test))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(X_train,y_train)
print('test accuracy:', rfc.score(X_test,y_test))
# First SVM model
from sklearn.svm import SVC
svm=SVC(random_state=101)
svm.fit(X_train, y_train)
print('train accuracy:', svm.score(X_train,y_train))
print('test accuracy:', svm.score(X_test,y_test))
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
accuracy_list_train=[]
k=np.arange(1,5,1)
for each in k:
    x_new = SelectKBest(f_classif, k = each).fit_transform(X_train,y_train)
    svm.fit(x_new, y_train)
    accuracy_list_train.append(svm.score(x_new,y_train))
    
plt.plot(k, accuracy_list_train, color='green', label='train')
plt.xlabel('k values')
plt.ylabel('train accuracy')
plt.legend()
plt.show()
d = {'best features number': k, 'train_score': accuracy_list_train}
df3 = pd.DataFrame(data=d)
print ('max accuracy:', df3['train_score'].max())
print ('max accuracy id:', df3['train_score'].idxmax())
# To sum up,
print ('Using the normalisation preprocessing: \n'
    'Linear Regresion Model precision:',lm.score(X_test,y_test),'\n',
    'KNN Model precision:', knn.score(X_test,y_test),'\n',
      'Logistic Regression Model precision:',logmodel.score(X_test,y_test),'\n',
      'Decision Tree Model precision:', dtree.score(X_test,y_test),'\n',
      'Random Tree Model precision:', rfc.score(X_test,y_test),'\n',
      'Support Machine Vector precision:', svm.score(X_test,y_test))
#REPLACE THIS WITH YOUR OWN GOOGLE PROJECT ID
PROJECT_ID = 'optimal-chimera-279914'
#REPLACE THIS WITH A NEW BUCKET NAME. NOTE: BUCKET NAMES MUST BE GLOBALLY UNIQUE
BUCKET_NAME = 'optimal-chimera-279914'
#Note: the bucket_region must be us-central1.
BUCKET_REGION = 'us-central1'
from google.cloud import storage, automl_v1beta1 as automl

storage_client = storage.Client(project=PROJECT_ID)
tables_gcs_client = automl.GcsClient(client=storage_client, bucket_name=BUCKET_NAME)
automl_client = automl.AutoMlClient()
# Note: AutoML Tables currently is only eligible for region us-central1. 
prediction_client = automl.PredictionServiceClient()
# Note: This line runs unsuccessfully without each one of these parameters
tables_client = automl.TablesClient(project=PROJECT_ID, region=BUCKET_REGION, client=automl_client, gcs_client=tables_gcs_client, prediction_client=prediction_client)
# Create your GCS Bucket with your specified name and region (if it doesn't already exist)
bucket = storage.Bucket(storage_client, name=BUCKET_NAME)
if not bucket.exists():
    bucket.create(location=BUCKET_REGION)
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    
def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(full_file_path)
# get the clean data without NaN
df_clean = df[['Time in bed', 'Start time in second','End time in second','Activity (steps)','Sleep quality']]
#rename the columns by removing the space ' '
df_clean.columns = ['Timeinbed','Starttimeinsecond','Endtimeinsecond','Activity','Sleepquality']
df_clean
# Randomly split the data into train and test set including the features and prediction
train_set = df_clean.sample(frac=0.75, random_state=0)
test_set = df_clean.drop(train_set.index)
# add a new column named 'ID'
train_set['ID'] = np.arange(1,666)
test_set['ID'] = np.arange(666,888)
train_set = train_set.set_index(np.arange(1,666))
train_set
test_set = test_set.set_index(np.arange(666,888))
test_set
# Any results you write to the current directory are saved as output.
# Write the dataframes back out to a csv file, which we can more easily upload to GCS. 
train_set.to_csv(path_or_buf='train.csv', index=False)
test_set.to_csv(path_or_buf='test.csv', index=False)
upload_blob(BUCKET_NAME, 'train.csv', 'train.csv')
upload_blob(BUCKET_NAME, 'test.csv', 'test.csv')
dataset_display_name = 'sleep_quality'
new_dataset = False
try:
    dataset = tables_client.get_dataset(dataset_display_name=dataset_display_name)
except:
    new_dataset = True
    dataset = tables_client.create_dataset(dataset_display_name)
# gcs_input_uris have the familiar path of gs://BUCKETNAME//file

if new_dataset:
    gcs_input_uris = ['gs://' + BUCKET_NAME + '/train.csv']

    import_data_operation = tables_client.import_data(
        dataset=dataset,
        gcs_input_uris=gcs_input_uris
    )
    print('Dataset import operation: {}'.format(import_data_operation))

    # Synchronous check of operation status. Wait until import is done.
    import_data_operation.result()
print(dataset)
