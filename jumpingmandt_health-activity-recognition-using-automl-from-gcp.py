# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load datasets
train_df = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')
test_df = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/test.csv')
train_df
import seaborn as sns
#check if the dataset has the missing data
sns.heatmap(train_df.isnull(),yticklabels='auto',cbar=False,cmap='coolwarm')
#check if the dataset has the missing data
sns.heatmap(test_df.isnull(),yticklabels='auto',cbar=False,cmap='coolwarm')
# add a new column named 'ID'
train_df['Id'] = np.arange(1,7353)
test_df['Id'] = np.arange(7353,10300)
test_df
train_df.columns = np.arange(1,565).astype(str)
train_df = train_df.rename(columns={'564': 'Id','563':'Activity','562':'subject'})
train_df.columns
test_df.columns = np.arange(1,565).astype(str)
test_df = test_df.rename(columns={'564': 'Id','563':'Activity','562':'subject'})
test_df.columns
# save data into a temporary file
train_df.to_csv(path_or_buf = 'train.csv', index = False)
test_df.to_csv(path_or_buf = 'test.csv', index = False)
#REPLACE THIS WITH YOUR OWN GOOGLE PROJECT ID
PROJECT_ID = 'optimal-chimera-279914'
#REPLACE THIS WITH A NEW BUCKET NAME. NOTE: BUCKET NAMES MUST BE GLOBALLY UNIQUE
BUCKET_NAME = 'optimal-chimera-279914'
#Note: the bucket_region must be us-central1.
BUCKET_REGION = 'us-central1'
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
model_id = "TBL6336883534082342912"

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
# It is very important to run this before running the model
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
# upload the train and test csv files

upload_blob(BUCKET_NAME, 'train.csv', 'train.csv')
upload_blob(BUCKET_NAME, 'test.csv', 'test.csv')
# each time you should choose a new dataset_display_name (very important) 
dataset_display_name = 'act_recog'
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
#print(dataset)
model_display_name = 'activity_model'
TARGET_COLUMN = 'Activity'
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

TRAIN_BUDGET = 1200 # (specified in milli-hours, from 1000-72000)
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
# I can go directly to this step if I have already created the model
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
predicted_activity = []
for i in np.arange(len(preds_df)):
    ind = np.argmax(preds_df.loc[i][-6:])
    if ind == 0:
        predicted_activity.append('LAYING')
    elif ind == 1:
        predicted_activity.append('STANDING')
    elif ind == 2:
        predicted_activity.append('SITTING')
    elif ind == 3:
        predicted_activity.append('WALKING')
    elif ind == 4:
        predicted_activity.append('WALKING_UPSTAIRS')
    else:
        predicted_activity.append('WALKING_DOWNSTAIRS')
preds_df['Predicted_Activity'] = predicted_activity
preds_df
submission_df = preds_df[['Id', 'Predicted_Activity']]
submission_df.columns = ['Id', 'Activity']
submission_df.to_csv('submission.csv', index=False)
submission_df.sort_values(by = 'Id',axis=0, ascending=True, inplace=True)
submission_df
test_df
from sklearn.metrics import classification_report
print (classification_report(test_df['Activity'], submission_df['Activity']))
# X_train contains the columes except the 'subject' and 'Activity'
X_train = train_df.drop(['subject', 'Activity'], axis=1)
X_train
# because we want to predict the activities so y_train only contains the 'Activity'
y_train = train_df['Activity']
y_train
# In order to make the ML model works, it is necessary to replace the String to Integer of 'Activity'
integer_activity = []
for i in np.arange(len(y_train)):
    ind = y_train.loc[i]
    if ind == 'LAYING':
        integer_activity.append(0)
    elif ind == 'STANDING':
        integer_activity.append(1)
    elif ind == 'SITTING':
        integer_activity.append(2)
    elif ind == 'WALKING':
        integer_activity.append(3)
    elif ind == 'WALKING_UPSTAIRS':
        integer_activity.append(4)
    else:
        integer_activity.append(5)
y_train = integer_activity
y_train
# the type of y_train
type(y_train[0])
len(y_train)
# X_train contains the columes except the 'subject' and 'Activity'
X_test = test_df.drop(['subject', 'Activity'], axis=1)
X_test
# because we want to predict the activities so y_train only contains the 'Activity'
y_test = test_df['Activity']
y_test
# convert the String to corresponding Integer in y_test
integer_activity2 = []
for i in np.arange(len(y_test)):
    ind = y_test.loc[i]
    if ind == 'LAYING':
        integer_activity2.append(0)
    elif ind == 'STANDING':
        integer_activity2.append(1)
    elif ind == 'SITTING':
        integer_activity2.append(2)
    elif ind == 'WALKING':
        integer_activity2.append(3)
    elif ind == 'WALKING_UPSTAIRS':
        integer_activity2.append(4)
    else:
        integer_activity2.append(5)

y_test =integer_activity2
y_test
len(y_test)
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.score(X_test,y_test)
# get the nearest integer using function .round()
prediction = lm.predict(X_test).round()
print ('Linear regression test accuracy', lm.score(X_test, y_test))
print (classification_report(y_test, prediction))
plt.figure(figsize=[10,8])
plt.hist(prediction, bins = 100)
# It is interesting to separately get the accuracy for each activity.So we need to separate the X_test to separated groups of activities. 
label = test_df['Activity']
label_counts = label.value_counts()
label_counts
# iterate over each activity

for activity in label_counts.index:
    #create dataset
    act_data = test_df[label==activity].copy()
    act_X_test = act_data.drop(['subject', 'Activity'], axis=1)
    act_y_test = act_data['Activity']
    y_temp = []
    for x in act_y_test:
        if x == 'LAYING':
            y_temp.append(0)
        elif x == 'STANDING':
            y_temp.append(1)
        elif x == 'SITTING':
            y_temp.append(2)
        elif x == 'WALKING':
            y_temp.append(3)
        elif x == 'WALKING_UPSTAIRS':
            y_temp.append(4)
        else:
            y_temp.append(5)

    act_y_test = y_temp

    y_predict = lm.predict(act_X_test).round()
    
    print ('Activity : {}'.format(activity))
    print ('Classification report : ',classification_report(act_y_test, y_predict))
 
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

knn = KNeighborsClassifier(n_neighbors=9) # why 9 is because of Elbow method
knn.fit(X_train,y_train)
print('test accuracy:', knn.score(X_test,y_test))
# get the nearest integer using function .round()
prediction = knn.predict(X_test).round()
print ('KNN Model test accuracy', knn.score(X_test, y_test))
print (classification_report(y_test, prediction))
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
# get the nearest integer using function .round()
prediction = logmodel.predict(X_test).round()
print ('Logistic regression test accuracy', logmodel.score(X_test, y_test))
print (classification_report(y_test, prediction))
from sklearn.tree import DecisionTreeClassifier 
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
# get the nearest integer using function .round()
prediction = dtree.predict(X_test).round()
print ('Decision Tree Classifier test accuracy', dtree.score(X_test, y_test))
print (classification_report(y_test, prediction))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 20)
rfc.fit(X_train, y_train)
# get the nearest integer using function .round()
prediction = rfc.predict(X_test).round()
print ('Random Forest Classifier test accuracy', rfc.score(X_test, y_test))
print (classification_report(y_test, prediction))
from sklearn.svm import SVC
svm=SVC(random_state=101)
svm.fit(X_train, y_train)
# get the nearest integer using function .round()
prediction = svm.predict(X_test).round()
print ('Support Machine Vector test accuracy', svm.score(X_test, y_test))
print (classification_report(y_test, prediction))
import re
X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(n_estimators = 500, random_state = 3)
lgbm = lgbm.fit(X_train, y_train)

# get the nearest integer using function .round()
prediction = lgbm.predict(X_test).round()
print ('LGBM Classifier test accuracy', lgbm.score(X_test, y_test))
print (classification_report(y_test, prediction))
data = [['Cloud AutoML', 0.92], ['Linear Regression Model', 0.85], ['KNN Classifier Model', 0.91], ['Logistic Regression Model', 0.96],
       ['Decission Tree Classifier Model', 0.86],['Random Forest Classifier Model', 0.92], ['Support Vector Machine Classifier Model', 0.95],
       ['LGBM Classifier Model', 0.94]] 
accuracy_df = pd.DataFrame(data, columns = ['Models','Accuracy'])
accuracy_df
