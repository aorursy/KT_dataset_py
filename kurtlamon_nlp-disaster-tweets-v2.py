import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
import re





def clean_keyword(keyword):

    return str(keyword).replace('%20', ' ')





def clean_location(location):

    return location





def clean_text(text):

    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

    clean_text = text

    for character in text:

        if ord(character) > 126 or ord(character) < 32:

            clean_text = clean_text.replace(character, '')



    return clean_text
test.fillna(value='')

train.fillna(value='')



test_arr = []

for _, row  in test.iterrows():

    test_arr.append([row['id'], clean_keyword(row['keyword']), clean_location(row['location']), clean_text(row['text'])])



nlp_test_df = pd.DataFrame(test_arr, columns=['id', 'keyword', 'location', 'text'])



train_arr = []

for _, row in train.iterrows():

    train_arr.append([row['id'], clean_keyword(row['keyword']), clean_location(row['location']), clean_text(row['text']), row['target']])

    

nlp_train_df = pd.DataFrame(train_arr, columns=['id', 'keyword', 'location', 'text', 'target'])
import numpy as np

import time

from datetime import datetime



from sklearn.model_selection import train_test_split



from google.cloud import storage

from google.cloud import automl_v1beta1 as automl



from google.api_core.gapic_v1.client_info import ClientInfo
PROJECT_ID = 'nlp-disaster-tweets-267516'

BUCKET_NAME = 'nlp-disaster-tweets'

BUCKET_REGION = 'us-central1'



dataset_display_name = 'kaggle_tweets'

model_display_name = 'kaggle_starter_model1'



storage_client = storage.Client(project=PROJECT_ID)



# adding ClientInfo here to get the gapic_v1 call in place

client = automl.AutoMlClient(client_info=ClientInfo())



print(f'Starting AutoML notebook at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
def gcs_upload(bucket_name, source_file_name, destination_blob_name):

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}'.format(

        source_file_name,

        'gs://' + bucket_name + '/' + destination_blob_name))

    

def gcs_download(bucket_name, destination_directory, file_name,prefix=None):

    os.makedirs(destination_directory, exist_ok = True)

    full_file_path = os.path.join(destination_directory, file_name)

    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)

    for blob in blobs:

        blob.download_to_filename(full_file_path)
bucket = storage.Bucket(storage_client, name=BUCKET_NAME)

if not bucket.exists():

    bucket.create(location=BUCKET_REGION)
nlp_train_df[['text', 'target']].to_csv('train.csv', index=False, header=False)
training_gcs_path = 'data/train.csv'

gcs_upload(BUCKET_NAME, 'train.csv', training_gcs_path)
dataset_name = 'disaster_tweets'

dataset_filter = 'display_name=' + dataset_name

location_path = client.location_path(PROJECT_ID, BUCKET_REGION)
print(f'Getting dataset ready at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')



list_datasets_response = client.list_datasets(location_path, dataset_filter)



dataset_results = []

for dataset_response in list_datasets_response:

    dataset_results.append(dataset_response)



dataset = dataset_results[0]

    

if not dataset_results:

    print('No matching datasets found, creating one...')

    dataset_metadata = {'classification_type': 'MULTICLASS'}

    dataset_info = {'display_name': dataset_name,

                    'text_classification_dataset_metadata': dataset_metadata}

    print('Creating new dataset...')

    dataset = client.create_dataset(location_path, dataset_info)

    data_config = {'gcs_source': {'input_uris': [f'gs://{BUCKET_NAME}/{training_gcs_path}']}}



    print('Importing CSV data. This may take a while...')

    operation = client.import_data(name=dataset.name, input_config=data_config)

    print(operation)



    result = operation.result()

    print(result)



print(dataset)

dataset_id = dataset.name.split('/')[-1]

    
print(f'Getting model trained at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')



list_models_response = client.list_models(location_path, dataset_filter)



models_results = []

for models_response in list_models_response:

    models_results.append(models_response)



if not models_results:

    print('No matching models found, creating one...')

    

    model_info = {

        "display_name": dataset_name,

        "dataset_id": dataset_id,

        "text_classification_model_metadata": {}

    }

    

    print('Creating and training model...')

    create_model_response = client.create_model(location_path, model_info)

    print(create_model_response)

    result = create_model_response.result()

    print(result)

    

    list_models_response = client.list_models(location_path, dataset_filter)



    models_results = []

    for model_response in list_models_response:

        models_results.append(models_reponse)

        

if models_results:

    print('Found model.')

    model = models_results[0]

else:

    print('Still no models found.')

    

print(model)
if model.deployment_state == model.UNDEPLOYED:

    print(f'Deploying model: {dataset_name} at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

    response = client.deploy_model(name=model.name)

    while model.deployment_state == model.UNDEPLOYED:

        time.sleep(120)

        list_models_response = client.list_models(location_path, dataset_filter)

        models_results = []

        for models_response in list_models_response:

            models_results.append(models_response)

        if not models_response:

            print('Model Not Found')

            quit()

        model = models_results[0]

    print(f'Finished deploying model: {dataset_name} at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

else:

    print(f'Model {dataset_name} is already deployed.')
input_col_name = 'text'

threshold = 0.5
print(f'Beginning Predictions at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')



prediction_client = automl.PredictionServiceClient()



predictions_list = []

correct = 0

total_test_size = len(nlp_test_df)



for i in range(total_test_size):

    row = nlp_test_df.iloc[i]

    snippet = row[input_col_name]

    

    payload = {"text_snippet": {"content": snippet,

                            "mime_type": "text/plain"}}

    params = {}

    response = prediction_client.predict(model.name, payload, params)



    for result in response.payload:

        if result.classification.score >= threshold:

            prediction = {'score': result.classification.score,

                          'class': result.display_name,

                          'text': snippet}

            predictions_list.append(prediction)

    time.sleep(0.3)



predictions_df = pd.DataFrame(predictions_list)



print(f'Finished getting predictions at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
print(predictions_df.head())
if model.deployment_state == model.DEPLOYED:

    print(f'Undepploying Model: {dataset_name} at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

    response = client.undeploy_model(name=model.name)

    while model.deployment_state == model.DEPLOYED:

        time.sleep(120)

        list_models_response = client.list_models(location_path, dataset_filter)

        models_results = []

        for models_response in list_models_response:

            models_results.append(models_response)

        if not models_response:

            print('Model Not Found')

            quit()

        model = models_results[0]

    print(f'Finished Undeploying Model: {dataset_name} at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

else:

    print(f'Model {dataset_name} is already undeployed.')
submission_df = pd.concat([nlp_test_df['id'], predictions_df['class']], axis=1)

submission_df = submission_df.rename(columns={'class': 'target'})

submission_df.to_csv("submission.csv", index=False, header=True)
! ls -l submission.csv