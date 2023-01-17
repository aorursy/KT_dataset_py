!pip3 install pyspark --quiet
!pip3 list | grep pyspark
import os
import pyspark as spark
import pyspark.sql.functions as F
import pyspark.ml as ml
import pyspark.mllib as mllib
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from datetime import datetime
import numpy as np
sc = spark.SparkContext()
sql = spark.sql.SQLContext(sc)
root_dir = '../input/restaurant-recommendation-challenge'
files = dict()
for f in os.listdir(root_dir):
    if f.endswith('.csv') and f != 'SampleSubmission (1).csv':
        files[f] = sql.read.format('csv').options(header='true').load(os.path.join(root_dir, f))
files['train_full.csv'].groupBy('discount_percentage').count().orderBy('count').show()
files['train_full.csv'].groupby('target').count().orderBy('count').show()
prep_s = files['train_full.csv'].select('prepration_time').toPandas()
prep_s['prepration_time'].astype('float').hist(color='gold')
del prep_s
p_method = files['orders.csv'].select('payment_mode').toPandas()
p_method['payment_mode'].astype('float').hist()
del p_method
total = files['orders.csv'].select('grand_total').toPandas()
sns.distplot(total['grand_total'].astype('float'), color='purple')
del total
"""I defined a function to automate dropping columns iterrating trough 
columns of each dataframe but it is painfully slow. Feel free to use 
it if you have plenty of spare time"""

def remove_cols(frame):
    for col in tqdm(frame.columns):
        nans = frame.rdd.map(lambda row: (
            row[col], sum([c == None for c in row]))).collect()
        #print(nans)
        if len(nans) > 0:
            distincts = frame.select(col).distinct().collect()
            #print(distincts)
            if len(distincts) == 2:
                frame = frame.drop(col)
    return frame
                
#for k, v in files.items():
#    files[k] = remove_cols(v)
weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 
            'friday', 'saturday', 'sunday']
to_drop = {'train_full.csv': ['commission', 'display_orders', 
                              'country_id', 'CID X LOC_NUM X VENDOR',
                             'city_id', 'vendor_category_en', 'latitude_x', 
                              'latitude_y', 'longitude_x', 'longitude_y'],
          'test_full.csv': ['commission', 'display_orders', 
                            'country_id', 'CID X LOC_NUM X VENDOR',
                           'city_id', 'vendor_category_en', 'latitude_x', 
                              'latitude_y', 'longitude_x', 'longitude_y'],
          'orders.csv': ['akeed_order_id', 'CID X LOC_NUM X VENDOR'],
          'train_customers.csv': ['language'],
          'train_customers.csv': ['language']}

for k, v in to_drop.items():
    for col in v:
        files[k] = files[k].drop(col)

for each in ['train_full.csv', 'test_full.csv']:
    for col in weekdays:
        for column in files[each].columns:
            if col in column:
                files[each] = files[each].drop(column)
numeric_cols = ['delivery_charge', 'serving_distance', 'vendor_rating', 
                'prepration_time', 'discount_percentage', 'verified_x', 
                'is_open', 'status_y', 'verified_y', 'rank', 
                'open_close_flags', 'location_number_obj']
for col in numeric_cols:
    files['train_full.csv'] = files['train_full.csv'].withColumn(
        col, files['train_full.csv'][col].cast(DoubleType()))
    files['test_full.csv'] = files['test_full.csv'].withColumn(
        col, files['test_full.csv'][col].cast(DoubleType()))

files['train_full.csv'] = files['train_full.csv'].withColumn(
        'target', files['train_full.csv']['target'].cast(DoubleType()))
files['train_full.csv'] = files['train_full.csv'].withColumn(
    'primary_tags', F.regexp_extract(
        files['train_full.csv']['primary_tags'], r"[0-9]+", 0))
files['test_full.csv'] = files['test_full.csv'].withColumn(
    'primary_tags', F.regexp_extract(
        files['test_full.csv']['primary_tags'], r"[0-9]+", 0))

files['train_full.csv'] = files['train_full.csv'].withColumn(
    'primary_tags', files['train_full.csv'][col].cast(DoubleType()))
files['test_full.csv'] = files['test_full.csv'].withColumn(
    'primary_tags', files['test_full.csv'][col].cast(DoubleType()))
files['train_customers.csv'] = files['train_customers.csv'].withColumnRenamed(
    'akeed_customer_id', 'customer_id')
files['test_customers.csv'] = files['test_customers.csv'].withColumnRenamed(
    'akeed_customer_id', 'customer_id')
train_df = files['train_full.csv'].join(files['train_customers.csv'], on=['customer_id'])
test_df = files['test_full.csv'].join(files['test_customers.csv'], on=['customer_id'])
train_df = train_df.drop('gender').drop('language')
test_df = test_df.drop('gender').drop('language')
train_df = train_df.withColumn('dob', train_df.dob.cast(
    DoubleType())).na.fill(2020.0)
test_df = test_df.withColumn('dob', test_df.dob.cast(
    DoubleType())).na.fill(2020.0)

train_df = train_df.fillna({'location_type': 'unknown'})
test_df = test_df.fillna({'location_type': 'unknown'})
train_df = train_df.withColumn('age', (2020-train_df.dob)).drop('dob')
test_df = test_df.withColumn('age', (2020-test_df.dob)).drop('dob')
median_age = np.array(train_df.select('age').collect())
median_age = np.median(median_age[median_age != 0.0])

train_df = train_df.withColumn('age', F.when(
    (train_df.age<100) & (train_df.age>12), train_df.age).otherwise(median_age))
test_df = test_df.withColumn('age', F.when(
    (test_df.age<100) & (test_df.age>12), test_df.age).otherwise(median_age))
train_df.select('age').distinct().show()
train_df = train_df.withColumn('created_at_x', F.to_date(train_df.created_at_x))
train_df = train_df.withColumn('created_at_y', F.to_date(train_df.created_at_y))
train_df = train_df.withColumn('updated_at_x', F.to_date(train_df.updated_at_x))
train_df = train_df.withColumn('updated_at_y', F.to_date(train_df.updated_at_y))

test_df = test_df.withColumn('created_at_x', F.to_date(test_df.created_at_x))
test_df = test_df.withColumn('created_at_x', F.to_date(test_df.created_at_x))
test_df = test_df.withColumn('updated_at_x', F.to_date(test_df.updated_at_x))
test_df = test_df.withColumn('updated_at_y', F.to_date(test_df.updated_at_y))
try:
    train_df = train_df.withColumn('x_loyal', F.datediff(
        train_df.updated_at_x, train_df.created_at_x))
    train_df = train_df.withColumn('y_loayl', F.datediff(
        train_df.updated_at_y, train_df.created_at_y))

    test_df = test_df.withColumn('x_loyal', F.datediff(
        test_df.updated_at_x, test_df.created_at_x))
    test_df = test_df.withColumn('y_loayl', F.datediff(
        test_df.updated_at_y, test_df.created_at_y))
except Exception:
    pass

train_df = train_df.drop('created_at_x').drop(
    'created_at_y').drop('updated_at_x').drop('updated_at_y')
test_df = test_df.drop('created_at_x').drop(
    'created_at_y').drop('updated_at_x').drop('updated_at_y')
train_df.select('x_loyal').distinct().show(10)
to_drop = ['OpeningTime', 'OpeningTime2', 'language', 
           'customer_id', 'vendor_tag', 'vendor_tag_name', 
           'created_at', 'updated_at', 'id', 'authentication_id', 
           'id_obj', 'is_akeed_delivering', 'one_click_vendor']
target = 'target'

for col in to_drop:
    train_df = train_df.drop(col)
    test_df = test_df.drop(col)
train_df.show(1)
categorical = ['location_number', 'location_type', 'status_x',
               'vendor_category_id', 'device_type', 'status', 
               'verified']

# train_df.select('one_click_vendor').distinct().show()
for col in categorical:
    stringIndexer = ml.feature.StringIndexer(inputCol=col, outputCol=col + "_ind")
    indexer = stringIndexer.fit(train_df)
    train_df = indexer.transform(train_df)
    test_df = indexer.transform(test_df)
    encoder = ml.feature.OneHotEncoder(
        inputCols=[stringIndexer.getOutputCol()], outputCols=[col + "_ohe"])
    ohe_encoder = encoder.fit(train_df)
    train_df = ohe_encoder.transform(train_df)
    test_df = ohe_encoder.transform(test_df)
"""
numeric_cols = ['delivery_charge', 'serving_distance', 'vendor_rating', 
                'prepration_time', 'discount_percentage', 'verified_x', 
                'is_open', 'status_y', 'verified_y', 'rank', 
                'open_close_flags', 'location_number_obj']

"""
train_df.show(1)
columns = numeric_cols + [col+'_ohe' for col in categorical]
assembler = ml.feature.VectorAssembler(
    inputCols=columns, 
    outputCol="features")

train = assembler.transform(train_df)
test = assembler.transform(test_df)
train_fit, train_eval = train.randomSplit([0.75, 0.25], seed=13)

l_reg = ml.classification.LogisticRegression(labelCol='target', featuresCol='features', maxIter=20)
l_reg=l_reg.fit(train_fit)

predict_train=l_reg.transform(train_fit)
predict_test=l_reg.transform(train_eval)
predict_test.select('prediction').distinct().show()
train_fit.groupBy('target').count().orderBy('count').show()
train_eval.groupBy('target').count().orderBy('count').show()
bal_train = train.filter(train.target==1.0)
target_count = bal_train.count()
target_df = train.filter(train.target==0.0).distinct()
target_df = target_df.sample(False, fraction=target_count/target_df.count())
target_df.count()
bal_train = bal_train.unionByName(target_df)
bal_train.sample(False, 0.1).show(10)
train_fit, train_eval = bal_train.randomSplit([0.75, 0.25], seed=13)

l_reg = ml.classification.LogisticRegression(labelCol='target', featuresCol='features', maxIter=20)
l_reg=l_reg.fit(train_fit)

predict_train=l_reg.transform(train_fit)
predict_test=l_reg.transform(train_eval)
predict_test.select('prediction').distinct().show()
predict_test.show()