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
# Libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#color
from colorama import Fore, Back, Style

#plotly
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

# To build models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# preview the train dataframe
train_df
# preview the train dataframe
test_df
# Check the list of files or folders in the data source
list(os.listdir("../input/house-prices-advanced-regression-techniques/"))
# check if there is missing data in the dataframe
# check the null part in the whole data set, red part is missing data, blue is non-null
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
train_df.isnull().sum()
# check the Missing data distribution in train_df
fig = px.scatter(train_df.isnull().sum())

fig.update_layout(
    title="Missing Data in train_df",
    xaxis_title="Columns",
    yaxis_title="Missing data count",
    showlegend=False,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="RebeccaPurple"
    )
)

fig.show()
#missing data thanks for the link https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/len(train_df)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# check if there is missing data in the dataframe
# check the null part in the whole data set, red part is missing data, blue is non-null
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
test_df.isnull().sum()
# check the Missing data distribution in test_df
fig = px.scatter(test_df.isnull().sum())

fig.update_layout(
    title="Missing Data in test_df",
    xaxis_title="Columns",
    yaxis_title="Missing data count",
    showlegend=False,
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="RebeccaPurple"
    )
)

fig.show()
# Shape of train and test dataframe
print(Fore.RED + 'Training data shape: ',Style.RESET_ALL,train_df.shape)
print(Fore.BLUE + 'Test data shape: ',Style.RESET_ALL,test_df.shape)
# Show the list of columns
columns = train_df.keys()
columns = list(columns)
print(Fore.RED + "List of columns in the train_df",Fore.GREEN + "", columns)
# distribution of price
plt.figure(figsize =(10,6))
sns.distplot(train_df['SalePrice'], bins = 100)
# BedroomAbvGr: Number of bedrooms above basement level
sns.countplot(train_df['BedroomAbvGr'])
train_df3.corr()['SalePrice'].sort_values(ascending = False).head(20)
# OverallQual: Overall material and finish quality
plt.figure(figsize = (10,5))
sns.scatterplot(x = 'SalePrice', y = 'OverallQual', data = train_df3 )
# TotRmsAbvGrd : Total rooms above grade (does not include bathrooms)
plt.figure(figsize = (10,6))
sns.boxplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = train_df3)
# MoSold : Month Sold
plt.figure(figsize = (10,6))
sns.boxplot(x = 'MoSold', y = 'SalePrice', data = train_df3)
# Thanks the code in https://www.kaggle.com/devvret/automl-tables-tutorial-notebook

# This dataset has some missing values, which we set to the median of the column for the purpose of this tutorial. 
cleaned_train_df = train_df.fillna(train_df.median())
cleaned_test_df = test_df.fillna(test_df.median())
# Any results you write to the current directory are saved as output.
# Write the dataframes back out to a csv file, which we can more easily upload to GCS. 
cleaned_train_df.to_csv(path_or_buf='train.csv', index=False)
cleaned_test_df.to_csv(path_or_buf='test.csv', index=False)
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

upload_blob(BUCKET_NAME, 'train.csv', 'train.csv')
upload_blob(BUCKET_NAME, 'test.csv', 'test.csv')
dataset_display_name = 'housing_prices_2'
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
# print(dataset)
model_display_name = 'housing_prediction_model2'
TARGET_COLUMN = 'SalePrice'
ID_COLUMN = 'Id'

# TODO: File bug: if you run this right after the last step, when data import isn't complete, you get a list index out of range
# There might be a more general issue, if you provide invalid display names, etc.

tables_client.set_target_column(
    dataset=dataset,
    column_spec_display_name=TARGET_COLUMN
)
# Make all columns nullable (except the Target and ID Column)
for col in tables_client.list_column_specs(PROJECT_ID,BUCKET_REGION,dataset.name):
    if TARGET_COLUMN in col.display_name or ID_COLUMN in col.display_name:
        continue
    tables_client.update_column_spec(PROJECT_ID,
                                     BUCKET_REGION,
                                     dataset.name,
                                     column_spec_display_name=col.display_name,
                                     type_code=col.data_type.type_code,
                                     nullable=True)
# Train the model. This will take hours (up to your budget). AutoML will early stop if it finds an optimal solution before your budget.
# On this dataset, AutoML usually stops around 2000 milli-hours (2 hours)

TRAIN_BUDGET = 5000 # (specified in milli-hours, from 1000-72000)
model = None
try:
    model = tables_client.get_model(model_display_name=model_display_name)
except:
    response = tables_client.create_model(
        model_display_name,
        dataset=dataset,
        train_budget_milli_node_hours=TRAIN_BUDGET,
        exclude_column_spec_names=[ID_COLUMN, TARGET_COLUMN]
    )
    print('Create model operation: {}'.format(response.operation))
    # Wait until model training is done.
    model = response.result()
# print(model)
# before the train time is too long, I need to reload the model that I have created in Google Cloud

# check the list of AutoML models in my account 
from google.cloud import automl

# TODO(developer): Uncomment and set the following variables
# project_id = "YOUR_PROJECT_ID"

client = automl.AutoMlClient()
# A resource that represents Google Cloud Platform location.
project_location = client.location_path(PROJECT_ID, "us-central1")
response = client.list_models(project_location, "")

print("List of models:")
for model in response:
    # Display the model information.
    if (
        model.deployment_state
        == automl.enums.Model.DeploymentState.DEPLOYED
    ):
        deployment_state = "deployed"
    else:
        deployment_state = "undeployed"

    print("Model name: {}".format(model.name))
    print("Model id: {}".format(model.name.split("/")[-1]))
    print("Model display name: {}".format(model.display_name))
    print("Model create time:")
    print("\tseconds: {}".format(model.create_time.seconds))
    print("\tnanos: {}".format(model.create_time.nanos))
    print("Model deployment state: {}".format(deployment_state))
# get the information of one model
from google.cloud import automl

# TODO(developer): Uncomment and set the following variables
# project_id = "YOUR_PROJECT_ID"
model_id = "TBL118327242357997568"

client = automl.AutoMlClient()
# Get the full path of the model.
model_full_id = client.model_path(PROJECT_ID, "us-central1", model_id)
model = client.get_model(model_full_id)

# Retrieve deployment state.
if model.deployment_state == automl.enums.Model.DeploymentState.DEPLOYED:
    deployment_state = "deployed"
else:
    deployment_state = "undeployed"

# Display the model information.
print("Model name: {}".format(model.name))
print("Model id: {}".format(model.name.split("/")[-1]))
print("Model display name: {}".format(model.display_name))
print("Model create time:")
print("\tseconds: {}".format(model.create_time.seconds))
print("\tnanos: {}".format(model.create_time.nanos))
print("Model deployment state: {}".format(deployment_state))
gcs_input_uris = 'gs://' + BUCKET_NAME + '/test.csv'
gcs_output_uri_prefix = 'gs://' + BUCKET_NAME + '/predictions'

batch_predict_response = tables_client.batch_predict(
    model=model, 
    gcs_input_uris=gcs_input_uris,
    gcs_output_uri_prefix=gcs_output_uri_prefix,
)
print('Batch prediction operation: {}'.format(batch_predict_response.operation))
# Wait until batch prediction is done.
batch_predict_result = batch_predict_response.result()
batch_predict_response.metadata
# The output directory for the prediction results exists under the response metadata for the batch_predict operation
# Specifically, under metadata --> batch_predict_details --> output_info --> gcs_output_directory
# Then, you can remove the first part of the output path that contains the GCS Bucket information to get your desired directory
gcs_output_folder = batch_predict_response.metadata.batch_predict_details.output_info.gcs_output_directory.replace('gs://' + BUCKET_NAME + '/','')
download_to_kaggle(BUCKET_NAME,'/kaggle/working','tables_1.csv', prefix=gcs_output_folder)
preds_df = pd.read_csv("tables_1.csv")
preds_df
submission3 = preds_df[['Id', 'predicted_SalePrice']]
submission3
submission3.columns = ['Id', 'SalePrice']
submission3
submission_df = preds_df[['Id', 'predicted_SalePrice']]
submission_df.columns = ['Id', 'SalePrice']
submission_df.to_csv('submission.csv', index=False)
submission_df
# because the price in train_df is interger, I want to convert the float to integer
submission_df['SalePrice'] = submission_df['SalePrice'].round()
submission_df.to_csv('submission.csv', index=False)
# review the dataframe
train_df2 = train_df.copy()
train_df2
# test dataframe
test_df2 = test_df.copy()
test_df2
# add the 'SalePrice' column to 0 temperally
test_df2['SalePrice'] = 0
#missing data thanks for the link https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
total = train_df2.isnull().sum().sort_values(ascending=False)
percent = (train_df2.isnull().sum()/len(train_df2)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# drop the columns with missing value over than 15 %
cleaned_train_df2 = train_df2.drop(columns = missing_data.index[:6], axis = 1).copy()
cleaned_train_df2
# drop the rows with missing value less than 10 %
cleaned_train_df2 = cleaned_train_df2.dropna()
cleaned_train_df2
# Then the test_df need to keep the same columns as train_df.
test_df2 = test_df2[cleaned_train_df2.columns]
test_df2
test_df2.fillna(test_df2.median())
# We first predict the value with more features. Then we can predict the value with less features which constain the NaN value. 
# We can find them with Id. The missing value can be predicted by the Google Cloud AutoML
test_df2 = test_df2.dropna()
# In order to be sure the object converted to the same integer after using LabelEncoder(), we need to first concatenate the train and test dataframe
all_df = pd.concat([cleaned_train_df2, test_df2], sort=False)
all_df
# select the columns with the object elements
all_df.select_dtypes(include=['object']).columns
# Convert the object to integer
enc = LabelEncoder()
for x in all_df.select_dtypes(include=['object']).columns:
    all_df[x] = enc.fit_transform(all_df[x])
all_df
# separate the train and test dataframe
train_df3 = all_df[:1338].copy()
train_df3
# separate the train and test dataframe
test_df3 = all_df[1338:]
test_df3
# save the dataframe to csv
cleaned_train_df2.to_csv('cleaned_train_df2.csv')
train_df3.to_csv('train_df3.csv')

test_df3 = test_df3.drop(columns = 'SalePrice').copy()
test_df3
test_df3.to_csv('test_df3.csv')
X_train = train_df3.drop(columns = 'SalePrice').values
X_train
y_train = train_df3['SalePrice'].values
y_train
X_test = test_df3.values
X_test
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(X_train,y_train)
# get the nearest integer using function .round()
prediction = lm.predict(X_test).round()
X_test['SalePrice'] = prediction
X_test
submission2 = X_test[['Id','SalePrice']]
submission2
# get the difference of ID 
set_difference = set(submission1['Id']) - set(submission2['Id'])
list_difference = list(set_difference)
list_difference
# I need to fill the missing value using the results obtained in Google Cloud AutoML

submission1 = pd.read_csv('../input/housing-price-prediction-using-different-ml/submission.csv')
submission1
# the different part using a list f[df['A'].isin([3, 6])]
submission_difference = submission1[submission1['Id'].isin(list_difference)]
submission_difference
# concatenate the 'submission2' and 'submission_difference'
submission = pd.concat([submission2, submission_difference], sort=False)
submission
submission.to_csv('submission.csv', index=False)
# Any results you write to the current directory are saved as output.
# Write the dataframes back out to a csv file, which we can more easily upload to GCS. 
train_df3.to_csv(path_or_buf='train.csv', index=False)
test_df3.to_csv(path_or_buf='test.csv', index=False)
# I need to fill the missing value using the results obtained in Google Cloud AutoML

submission1 = pd.read_csv('../input/housing-price-prediction-using-different-ml/submission.csv')
submission1
# get the difference of ID 
set_difference = set(submission1['Id']) - set(submission3['Id'])
list_difference = list(set_difference)
list_difference
# the different part using a list f[df['A'].isin([3, 6])]
submission_difference = submission1[submission1['Id'].isin(list_difference)]
submission_difference
# concatenate the 'submission2' and 'submission_difference'
submission = pd.concat([submission3, submission_difference], sort=False)
submission
submission.to_csv('submission2.csv', index=False)
submission1.to_csv('submission.csv', index=False)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train
X_test = scaler.transform(test_df3)
X_test
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X_train.shape
model = Sequential()

model.add(Dense(74, activation = 'relu'))
model.add(Dense(74, activation = 'relu'))
model.add(Dense(74, activation = 'relu'))
model.add(Dense(74, activation = 'relu'))

model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')
from sklearn.model_selection import train_test_split
# train test split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size = 0.2, random_state = 86)
model.fit(x = X_train2, y = y_train2, 
          validation_data = (X_test2, y_test2),
          batch_size = 128, epochs = 400)
losses_df = pd.DataFrame(model.history.history)
losses_df.plot()
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model.predict(X_test2)
mean_squared_error(y_test2,predictions)
np.sqrt(mean_squared_error(y_test2,predictions))
mean_absolute_error(y_test2,predictions)
train_df['SalePrice'].describe()
explained_variance_score(y_test2, predictions)
plt.figure(figsize=(12,6))
plt.scatter(y_test2, predictions)
plt.plot(y_test2,y_test2, 'r')
predictions2 = model.predict(X_test)
predictions2
test_df3['SalePrice'] = predictions2
submission3 = test_df3[['Id','SalePrice']]
submission3
submission
# get the difference of ID 
set_difference = set(submission1['Id']) - set(submission3['Id'])
list_difference = list(set_difference)
list_difference
# the different part using a list f[df['A'].isin([3, 6])]
submission_difference = submission1[submission1['Id'].isin(list_difference)]
submission_difference
# concatenate the 'submission2' and 'submission_difference'
submission3 = pd.concat([submission3, submission_difference], sort=False)
submission3
submission3.to_csv('submission3.csv', index=False)
