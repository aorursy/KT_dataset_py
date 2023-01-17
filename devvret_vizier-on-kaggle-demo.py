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
! pip install -U google-api-python-client

! pip install -U google-cloud

! pip install -U google-cloud-storage

! pip install -U requests



# Automatically restart kernel after installs

import IPython

import time

# app = IPython.Application.instance()

# app.kernel.do_shutdown(True)  
time.sleep(5)
from kaggle_gcp import KaggleKernelCredentials

import io
%env GOOGLE_APPLICATION_CREDENTIALS /kaggle/input/automltablessa/kaggle-playground-170215-6ed5acf02acd.json
!gcloud auth activate-service-account --key-file=/kaggle/input/automltablessa/kaggle-playground-170215-6ed5acf02acd.json



# ! gcloud auth application-default login

# ! gcloud auth login
PROJECT_ID = "kaggle-playground-170215"

! gcloud config set project $PROJECT_ID
import json

import time

import datetime

from googleapiclient import errors
USER = 'devvret' #@param {type: 'string'}



STUDY_ID = '{}_study_{}'.format(USER, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) #@param {type: 'string'}

REGION = 'us-central1'



def study_parent():

  return 'projects/{}/locations/{}'.format(PROJECT_ID, REGION)





def study_name(study_id):

  return 'projects/{}/locations/{}/studies/{}'.format(PROJECT_ID, REGION, study_id)





def trial_parent(study_id):

  return study_name(study_id)





def trial_name(study_id, trial_id):

  return 'projects/{}/locations/{}/studies/{}/trials/{}'.format(PROJECT_ID, REGION,

                                                                study_id, trial_id)



def operation_name(operation_id):

  return 'projects/{}/locations/{}/operations/{}'.format(PROJECT_ID, REGION, operation_id)





print('USER: {}'.format(USER))

print('PROJECT_ID: {}'.format(PROJECT_ID))

print('REGION: {}'.format(REGION))

print('STUDY_ID: {}'.format(STUDY_ID))
from google.cloud import storage

from googleapiclient import discovery





_OPTIMIZER_API_DOCUMENT_BUCKET = 'caip-optimizer-public'

_OPTIMIZER_API_DOCUMENT_FILE = 'api/ml_public_google_rest_v1.json'





def read_api_document():

  client = storage.Client(PROJECT_ID, credentials=KaggleKernelCredentials())

  bucket = client.get_bucket(_OPTIMIZER_API_DOCUMENT_BUCKET)

  blob = bucket.get_blob(_OPTIMIZER_API_DOCUMENT_FILE)

  return blob.download_as_string()





ml = discovery.build_from_document(service=read_api_document())

print('Successfully built the client.')
param_learning_rate = {

    'parameter': 'learning_rate',

    'type' : 'DOUBLE',

    'double_value_spec' : {

        'min_value' : 0.00001,

        'max_value' : 1.0

    },

    'scale_type' : 'UNIT_LOG_SCALE',

    'parent_categorical_values' : {

        'values': ['LINEAR', 'WIDE_AND_DEEP']

    },

}



param_dnn_learning_rate = {

    'parameter': 'dnn_learning_rate',

    'type' : 'DOUBLE',

    'double_value_spec' : {

        'min_value' : 0.00001,

        'max_value' : 1.0

    },

    'scale_type' : 'UNIT_LOG_SCALE',

    'parent_categorical_values' : {

        'values': ['WIDE_AND_DEEP']

    },

}



param_model_type = {

    'parameter': 'model_type',

    'type' : 'CATEGORICAL',

    'categorical_value_spec' : {'values': ['LINEAR', 'WIDE_AND_DEEP']},

    'child_parameter_specs' : [param_learning_rate, param_dnn_learning_rate,]

}



metric_accuracy = {

    'metric' : 'accuracy',

    'goal' : 'MAXIMIZE'

}



study_config = {

    'algorithm' : 'ALGORITHM_UNSPECIFIED',  # Let the service choose the `default` algorithm.

    'parameters' : [param_model_type,],

    'metrics' : [metric_accuracy,],

}



study = {'study_config': study_config}

print(json.dumps(study, indent=2, sort_keys=True))

# Creates a study

req = ml.projects().locations().studies().create(

    parent=study_parent(), studyId=STUDY_ID, body=study)

try :

  print(req.execute())

except errors.HttpError as e:

  if e.resp.status == 409:

    print('Study already existed.')

  else:

    raise e
# `job_dir` will be `gs://${OUTPUT_BUCKET}/${OUTPUT_DIR}/${job_id}`

OUTPUT_BUCKET = 'vizier-test-kaggle-playground' #@param {type: 'string'}

OUTPUT_DIR = 'test-dir' #@param {type: 'string'}

TRAINING_DATA_PATH = 'gs://caip-optimizer-public/sample-data/raw_census_train.csv' #@param {type: 'string'}



print('OUTPUT_BUCKET: {}'.format(OUTPUT_BUCKET))

print('OUTPUT_DIR: {}'.format(OUTPUT_DIR))

print('TRAINING_DATA_PATH: {}'.format(TRAINING_DATA_PATH))



# Create the bucket in Cloud Storage

#! gsutil mb -p $PROJECT_ID gs://$OUTPUT_BUCKET/
import logging

import math

import subprocess

import os

import yaml



from google.cloud import storage



_TRAINING_JOB_NAME_PATTERN = '{}_condition_parameters_{}_{}'

_IMAGE_URIS = {'LINEAR' : 'gcr.io/cloud-ml-algos/linear_learner_cpu:latest',

               'WIDE_AND_DEEP' : 'gcr.io/cloud-ml-algos/wide_deep_learner_cpu:latest'}

_STEP_COUNT = 'step_count'

_ACCURACY = 'accuracy'





def EvaluateTrials(trials):

  """Evaluates trials by submitting training jobs to AI Platform Training service.



     Args:

       trials: List of Trials to evaluate



     Returns: A dict of <trial_id, measurement> for the given trials.

  """

  trials_by_job_id = {}

  mesurement_by_trial_id = {}



  # Submits a AI Platform Training job for each trial.

  for trial in trials:

    trial_id = int(trial['name'].split('/')[-1])

    model_type = _GetSuggestedParameterValue(trial, 'model_type', 'stringValue')

    learning_rate = _GetSuggestedParameterValue(trial, 'learning_rate',

                                                'floatValue')

    dnn_learning_rate = _GetSuggestedParameterValue(trial, 'dnn_learning_rate',

                                                    'floatValue')

    job_id = _GenerateTrainingJobId(model_type=model_type, 

                                    trial_id=trial_id)

    trials_by_job_id[job_id] = {

        'trial_id' : trial_id,

        'model_type' : model_type,

        'learning_rate' : learning_rate,

        'dnn_learning_rate' : dnn_learning_rate,

    }

    _SubmitTrainingJob(job_id, trial_id, model_type, learning_rate, dnn_learning_rate)



  # Waits for completion of AI Platform Training jobs.

  print(trials_by_job_id.keys())

  while not _JobsCompleted(trials_by_job_id.keys()):

    time.sleep(60)



  # Retrieves model training result(e.g. global_steps, accuracy) for AI Platform Training jobs.

  metrics_by_job_id = _GetJobMetrics(trials_by_job_id.keys())

  for job_id, metric in metrics_by_job_id.items():

    measurement = _CreateMeasurement(trials_by_job_id[job_id]['trial_id'],

                                     trials_by_job_id[job_id]['model_type'],

                                     trials_by_job_id[job_id]['learning_rate'],

                                     trials_by_job_id[job_id]['dnn_learning_rate'],

                                     metric)

    mesurement_by_trial_id[trials_by_job_id[job_id]['trial_id']] = measurement

  return mesurement_by_trial_id





def _CreateMeasurement(trial_id, model_type, learning_rate, dnn_learning_rate, metric):

  if not metric[_ACCURACY]:

    # Returns `none` for trials without metrics. The trial will be marked as `INFEASIBLE`.

    return None

  print(

      'Trial {0}: [model_type = {1}, learning_rate = {2}, dnn_learning_rate = {3}] => accuracy = {4}'.format(

          trial_id, model_type, learning_rate,

          dnn_learning_rate if dnn_learning_rate else 'N/A', metric[_ACCURACY]))

  measurement = {

      _STEP_COUNT: metric[_STEP_COUNT],

      'metrics': [{'metric': _ACCURACY, 'value': metric[_ACCURACY]},]}

  return measurement





def _SubmitTrainingJob(job_id, trial_id, model_type, learning_rate, dnn_learning_rate=None):

  """Submits a built-in algo training job to AI Platform Training Service."""

  try:

    if model_type == 'LINEAR':

      subprocess.check_output(_LinearCommand(job_id, learning_rate), stderr=subprocess.STDOUT)

    elif model_type == 'WIDE_AND_DEEP':

      subprocess.check_output(_WideAndDeepCommand(job_id, learning_rate, dnn_learning_rate), stderr=subprocess.STDOUT)

    print('Trial {0}: Submitted job [https://console.cloud.google.com/ai-platform/jobs/{1}?project={2}].'.format(trial_id, job_id, PROJECT_ID))

  except subprocess.CalledProcessError as e:

    logging.error(e.output)





def _GetTrainingJobState(job_id):

  """Gets a training job state."""

  cmd = ['gcloud', 'ai-platform', 'jobs', 'describe', job_id,

         '--project', PROJECT_ID,

         '--format', 'json']

  try:

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=3)

  except subprocess.CalledProcessError as e:

    logging.error(e.output)

    print(e.output)

  return json.loads(output)['state']





def _JobsCompleted(jobs):

  """Checks if all the jobs are completed."""

  all_done = True

  for job in jobs:

    if _GetTrainingJobState(job) not in ['SUCCEEDED', 'FAILED', 'CANCELLED']:

      print('Waiting for job[https://console.cloud.google.com/ai-platform/jobs/{0}?project={1}] to finish...'.format(job, PROJECT_ID))

      all_done = False

  return all_done





def _RetrieveAccuracy(job_id):

  """Retrices the accuracy of the trained model for a built-in algorithm job."""

  storage_client = storage.Client(project=PROJECT_ID)

  bucket = storage_client.get_bucket(OUTPUT_BUCKET)

  blob_name = os.path.join(OUTPUT_DIR, job_id, 'model/deployment_config.yaml')

  blob = storage.Blob(blob_name, bucket)

  try: 

    blob.reload()

    content = blob.download_as_string()

    accuracy = float(yaml.safe_load(content)['labels']['accuracy']) / 100

    step_count = int(yaml.safe_load(content)['labels']['global_step'])

    return {_STEP_COUNT: step_count, _ACCURACY: accuracy}

  except:

    # Returns None if failed to load the built-in algo output file.

    # It could be due to job failure and the trial will be `INFEASIBLE`

    return None





def _GetJobMetrics(jobs):

  accuracies_by_job_id = {}

  for job in jobs:

    accuracies_by_job_id[job] = _RetrieveAccuracy(job)

  return accuracies_by_job_id





def _GetSuggestedParameterValue(trial, parameter, value_type):

  param_found = [p for p in trial['parameters'] if p['parameter'] == parameter]

  if param_found:

    return param_found[0][value_type]

  else:

    return None





def _GenerateTrainingJobId(model_type, trial_id):

  return _TRAINING_JOB_NAME_PATTERN.format(STUDY_ID, model_type, trial_id)





def _GetJobDir(job_id):

  return os.path.join('gs://', OUTPUT_BUCKET, OUTPUT_DIR, job_id)





def _LinearCommand(job_id, learning_rate):

  return ['gcloud', 'ai-platform', 'jobs', 'submit', 'training', job_id,

          '--scale-tier', 'BASIC',

          '--region', 'us-central1',

          '--master-image-uri', _IMAGE_URIS['LINEAR'],

          '--project', PROJECT_ID,

          '--job-dir', _GetJobDir(job_id),

          '--',

          '--preprocess',

          '--model_type=classification',

          '--batch_size=250',

          '--max_steps=1000',

          '--learning_rate={}'.format(learning_rate),

          '--training_data_path={}'.format(TRAINING_DATA_PATH)]





def _WideAndDeepCommand(job_id, learning_rate, dnn_learning_rate):

  return ['gcloud', 'ai-platform', 'jobs', 'submit', 'training', job_id,

          '--scale-tier', 'BASIC',

          '--region', 'us-central1',

          '--master-image-uri', _IMAGE_URIS['WIDE_AND_DEEP'],

          '--project', PROJECT_ID,

          '--job-dir', _GetJobDir(job_id),

          '--',

          '--preprocess',

          '--test_split=0',

          '--use_wide',

          '--embed_categories',

          '--model_type=classification',

          '--batch_size=250',

          '--learning_rate={}'.format(learning_rate),

          '--dnn_learning_rate={}'.format(dnn_learning_rate),

          '--max_steps=1000',

          '--training_data_path={}'.format(TRAINING_DATA_PATH)]
client_id = 'client1' #@param {type: 'string'}

suggestion_count_per_request =   2 #@param {type: 'integer'}

max_trial_id_to_stop =   4 #@param {type: 'integer'}



print('client_id: {}'.format(client_id))

print('suggestion_count_per_request: {}'.format(suggestion_count_per_request))

print('max_trial_id_to_stop: {}'.format(max_trial_id_to_stop))

current_trial_id = 0

while current_trial_id < max_trial_id_to_stop:

  # Request trials

  resp = ml.projects().locations().studies().trials().suggest(

    parent=trial_parent(STUDY_ID), 

    body={'client_id': client_id, 'suggestion_count': suggestion_count_per_request}).execute()

  op_id = resp['name'].split('/')[-1]



  # Polls the suggestion long-running operations.

  get_op = ml.projects().locations().operations().get(name=operation_name(op_id))

  while True:

      operation = get_op.execute()

      if 'done' in operation and operation['done']:

        break

      time.sleep(1)



  # Featches the suggested trials.

  trials = []

  for suggested_trial in get_op.execute()['response']['trials']:

    trial_id = int(suggested_trial['name'].split('/')[-1])

    trial = ml.projects().locations().studies().trials().get(name=trial_name(STUDY_ID, trial_id)).execute()

    if trial['state'] not in ['COMPLETED', 'INFEASIBLE']:

      print("Trial {}: {}".format(trial_id, trial))

      trials.append(trial)



  # Evaluates trials - Submit model training jobs using AI Platform Training built-in algorithms.

  measurement_by_trial_id = EvaluateTrials(trials)



  # Completes trials.

  for trial in trials:

    trial_id = int(trial['name'].split('/')[-1])

    current_trial_id = trial_id

    measurement = measurement_by_trial_id[trial_id]

    print(("=========== Complete Trial: [{0}] =============").format(trial_id))

    if measurement:

      # Completes trial by reporting final measurement.

      ml.projects().locations().studies().trials().complete(

        name=trial_name(STUDY_ID, trial_id), 

        body={'final_measurement' : measurement}).execute()

    else:

      # Marks trial as `infeasbile` if when missing final measurement.

      ml.projects().locations().studies().trials().complete(

        name=trial_name(STUDY_ID, trial_id), 

        body={'trial_infeasible' : True}).execute()
resp = ml.projects().locations().studies().trials().list(parent=trial_parent(STUDY_ID)).execute()

print(json.dumps(resp, indent=2, sort_keys=True))
!pip install --upgrade google-cloud-language
from google.cloud import language_v1

from google.cloud.language import enums

from google.cloud.language import types
language_client = language_v1.LanguageServiceClient(credentials=KaggleKernelCredentials())
text = u'This product is excellent! It works exactly like I would expect, look forward to recommending it to a friend'

document = types.Document(

    content=text,

    type=enums.Document.Type.PLAIN_TEXT)



# Detects the sentiment of the text

sentiment = language_client.analyze_sentiment(document=document).document_sentiment



print('Text: {}'.format(text))

print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))