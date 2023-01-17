import numpy as np

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

pd.set_option('display.max_colwidth', -1) #to max the column width
nlp_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

nlp_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



nlp_train.head(2)
PROJECT_ID = 'kaggle-bqml-course'

bucket_name = 'leosuky_kaggle_competitions'

region = 'us-central1'



from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

from google.cloud import automl_v1beta1 as automl

automl_client = automl.AutoMlClient()

from automlwrapper import AutoMLWrapper

#Function to upload files to google cloud storage

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):

    "Uploads a file to the bucket. https://cloud.google.com/storage/docs/"

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    
#bucket = storage.Bucket(storage_client, name=bucket_name)

#if not bucket.exists():

 #   bucket.create(location=region)
#nlp_train[['text','target']].to_csv('train.csv', index=False, header=False)

nlp_test.head(3)
destination_path = 'uploads/kaggle_auto_ml/train.csv'

#upload_to_gcs(bucket_name, 'train.csv', destination_path)
#Setting up a few variables

dataset_name = 'disaster_tweets_nlp'

model_name = 'disaster_tweets_nlp'

client = automl_client

region = region

project_id = PROJECT_ID

bucket_name = bucket_name
amw = AutoMLWrapper(client=client, 

                    project_id=PROJECT_ID, 

                    bucket_name=bucket_name, 

                    region='us-central1', 

                    dataset_display_name=dataset_name, 

                    model_display_name=model_name)
#define a function to create dataset

def create_dataset(project_id, dataset_name):

    #specify the client and project location

    client = automl.AutoMlClient()

    project_location = client.location_path(project_id, region)

    

    #specfiy the classification type

    #types are: 1. Text Classification, 2. Sentiment Analysis, 3. Entity Extraction.

    metadata = automl.types.TextSentimentDatasetMetadata(sentiment_max=1)

    dataset = automl.types.Dataset(display_name=dataset_name,

                                  text_sentiment_dataset_metadata=metadata)

    

    #create the dataset with its respsective metadata

    result = client.create_dataset(project_location, dataset)

    

    #display the created dataset info

    print(result)
#create_dataset(project_id=PROJECT_ID, dataset_name='disaster_tweets_nlp')
#Now we import the csv file to the created dataset

def import_dataset(project_id, dataset_id, path):

    #specify the client and full dataset id

    client = automl.AutoMlClient()

    dataset_full_id = client.dataset_path(project_id, 'us-central1', dataset_id)

    

    #Get the Google Cloud Storage URI's

    input_uris = path.split(",")

    gcs_source = automl.types.GcsSource(input_uris=input_uris)

    input_config = automl.types.InputConfig(gcs_source=gcs_source)

    

    #Now import the data from the URI

    result = client.import_data(dataset_full_id, input_config)

    

    print("Processing import...")

    print("Data imported. {}".format(result))
file_path = 'gs://leosuky_kaggle_competitions/uploads/kaggle_auto_ml/train.csv'

dataset_id = 'TST2133943162303938560'

#import_dataset(project_id=PROJECT_ID, dataset_id=dataset_id, path=file_path)
#define a function to create our model

def create_model(project_id, dataset_id, dataset_name):

    #specify client and project location

    client = automl.AutoMlClient()

    project_location = client.location_path(project_id, 'us-central1')

    

    #Model metadata

    metadata = automl.types.TextSentimentModelMetadata()

    model = automl.types.Model(display_name=dataset_name,

                              dataset_id=dataset_id,

                              text_sentiment_model_metadata=metadata)

    

    #creating the model

    result = client.create_model(project_location, model)

    

    print(u"Training operation name: {}".format(result.operation.name))

    print("Training started...")
#create_model(project_id=PROJECT_ID, dataset_id=dataset_id, dataset_name=dataset_name)



print("model has already been created and trained!")
#define a function to list our model evaluations



def model_evaluations(project_id, model_id):

    #specify the client and full model id.

    client = automl.AutoMlClient()

    full_model_id = client.model_path(project_id, 'us-central1', model_id)

    

    print('Model Evaluations:')

    for evaluation in client.list_model_evaluations(full_model_id, ""):

        print('Model Evaluation Name: {}'.format(evaluation.name))

        print('Model Annotation Spec Id: {}'.format(evaluation.annotation_spec_id))

        print("Create Time:")

        print("\tseconds: {}".format(evaluation.create_time.seconds))

        print("\tnanos: {}".format(evaluation.create_time.nanos / 1e9))

        print('Evaluation Example Count: {}'.format(evaluation.evaluated_example_count))

        print('Sentiment Analysis Evaluation Metrics: {}'.format(evaluation.text_sentiment_evaluation_metrics))

        print('Translation Model Evaluation Metrics: {}'.format(evaluation.translation_evaluation_metrics))

        
model_id = "TST4544356324388896768"

model_evaluations(project_id=project_id, model_id=model_id)
if not amw.get_dataset_by_display_name(dataset_display_name=dataset_name):

    print('dataset not found')

    amw.create_dataset()

    amw.import_gcs_data(training_gcs_path)

    

amw.dataset
if not amw.get_model_by_display_name():

    amw.train_model()

    

amw.deploy_model()

amw.model
amw.model_full_path
nlp_test.head(3)
corpus = list(nlp_test.text)

print(corpus[2375])

print(len(corpus))
#We'll define a function that takes input and returns our sentiment score as output

def make_predictions(project_id, model_id, content):

    #specify the prediction client and full_model_id

    prediction_client = automl.PredictionServiceClient()

    model_full_id = prediction_client.model_path(project_id, "us-central1", model_id)

    

    #specify text snippet and payload

    text_snippet = automl.types.TextSnippet(content=content, mime_type="text/plain")

    payload = automl.types.ExamplePayload(text_snippet=text_snippet)

    

    #prediction response

    response = prediction_client.predict(model_full_id, payload)

    for annotation_payload in response.payload:

        return annotation_payload.text_sentiment.sentiment

        
#Test the function

make_predictions(project_id=PROJECT_ID, model_id=model_id, content=corpus[1])
predictions = []

print('Starting predictions...')



print('Predicting 1st batch.....')

for document in corpus[:600]:

    predictions.append(

    make_predictions(project_id=PROJECT_ID, model_id=model_id, content=document)

    )

    

print('Predicting 2nd batch.....')

for document in corpus[600:1200]:

    predictions.append(

    make_predictions(project_id=PROJECT_ID, model_id=model_id, content=document)

    )

    

print('Predicting 3rd batch.....')

for document in corpus[1200:1800]:

    predictions.append(

    make_predictions(project_id=PROJECT_ID, model_id=model_id, content=document)

    )    

    

print('Predicting 4th batch.....')

for document in corpus[1800:2400]:

    predictions.append(

    make_predictions(project_id=PROJECT_ID, model_id=model_id, content=document)

    )    

    

print('Predicting last batch.....')

for document in corpus[2400:]:

    predictions.append(

    make_predictions(project_id=PROJECT_ID, model_id=model_id, content=document)

    )    

print('Done.')
len(predictions)
sentiment_predictions = pd.DataFrame(predictions, columns=['target'])

sentiment_predictions.head(4)
submission_df = pd.concat([nlp_test['id'], sentiment_predictions['target']], axis=1)

print(submission_df.shape)

submission_df.head()
submission_df.to_csv("submission.csv", index=False, header=True)