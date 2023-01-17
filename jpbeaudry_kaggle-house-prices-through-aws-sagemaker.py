!pip install kaggle
!kaggle competitions list
!mkdir ~/.kaggle
!mv kaggle.json ../.kaggle/
!chmod 600 ../.kaggle/kaggle.json
import numpy as np                                # For matrix operations and numerical processing
import pandas as pd                               # For munging tabular data
import matplotlib.pyplot as plt                   # For charts and visualizations
from IPython.display import Image                 # For displaying images in the notebook
from IPython.display import display               # For displaying outputs in the notebook
from time import gmtime, strftime                 # For labeling SageMaker models, endpoints, etc.
import sys                                        # For writing outputs to notebook
import math                                       # For ceiling function
import json                                       # For parsing hosting outputs
import os                                         # For manipulating filepath names
import sagemaker                                  # Amazon SageMaker's Python SDK provides many helper functions
from sagemaker.predictor import csv_serializer    # Converts strings for HTTP POST requests on inference
bucket = sagemaker.Session().default_bucket()
prefix = 'sagemaker/house_price_xgboost'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()
region = boto3.Session().region_name 
smclient = boto3.Session().client('sagemaker')
!kaggle competitions list
!kaggle competitions download -p ./data house-prices-advanced-regression-techniques
df_train = pd.read_csv('./data/train.csv')
pd.set_option('display.max_columns', 500)     # Make sure we can see all of the columns
df_train.head()
!head ./data/sample_submission.csv
df_train.describe()['SalePrice']
df_competition = pd.read_csv('./data/test.csv')
df_competition.head()
df_submit = pd.DataFrame(df_competition['Id'], dtype=int)
df_submit['SalePrice'] = df_train.describe()['SalePrice']['mean']
df_submit.head()
df_submit.tail()
df_submit.to_csv('./data/sub_mean.csv',index=False)
!head ./data/sub_mean.csv
!tail ./data/sub_mean.csv

!kaggle competitions submit -f ./data/sub_mean.csv -m "Means-based submission" house-prices-advanced-regression-techniques
#
# Need UI buffer space so horizontal scrollbar does not get in the way
df_train = pd.get_dummies(df_train)   # Convert categorical variables to sets of indicators
df_train.describe()
#model_data = data
train_data, validation_data, test_data = np.split(df_train.sample(frac=1, random_state=1729), [int(0.7 * len(df_train)), int(0.9*len(df_train))])  
train_data.shape
#pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
#pd.concat([validation_data['y_yes'], validation_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)
pd.concat([train_data['SalePrice'], train_data.drop(['SalePrice'], axis=1)], axis=1).to_csv('./data/sm_train.csv', index=False, header=False)
pd.concat([validation_data['SalePrice'], validation_data.drop(['SalePrice'], axis=1)], axis=1).to_csv('./data/sm_validation.csv', index=False, header=False)
!ls -l ./data
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('./data/sm_train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('./data/sm_validation.csv')
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'xgboost')
s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')
sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)

# These are the default parameters that came with the Targeting Direct Marketing example
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='reg:linear',
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 
xgb_predictor = xgb.deploy(initial_instance_count=1,
                           instance_type='ml.m4.xlarge')
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
df_competition = pd.get_dummies(df_competition)
df_competition.shape
df_train.shape
df_comp_padded=df_competition
for x in range(0, 18):
    df_comp_padded[x] = pd.Series(1, index=df_comp_padded.index)
df_comp_padded.shape
df_comp_padded.head()
def predict(data):
    ids = data['Id']
    saleprice = np.array(xgb_predictor.predict(data.as_matrix()).decode('utf-8').split(',')).astype(np.float)
    predictions = list(zip(ids,saleprice))
    return predictions

#predictions = predict(test_data.drop(['SalePrice'], axis=1).as_matrix())
#predictions = predict(kaggle_data.drop(['SalePrice'], axis=1).as_matrix())
#predictions = predict(kd.as_matrix())
%time predictions = predict(df_comp_padded)
predictions[0:5]
np.array(predictions).shape
#np.savetxt("../data/housing.csv", predictions, delimiter=",", header='Id,SalePrice', fmt='%u')
df = pd.DataFrame(predictions, columns=['Id', 'SalePrice'])
print(df.head())
df.to_csv('./data/sub_xgboost_default.csv', header=True, index=False)
!kaggle competitions submit -f ./data/sub_xgboost_default.csv -m "Default XGBoost submission" house-prices-advanced-regression-techniques
#
# Damned scroll bar
from time import gmtime, strftime, sleep
tuning_job_name = 'xgboost-tuningjob-' + strftime("%d-%H-%M-%S", gmtime())

print (tuning_job_name)

tuning_job_config = {
    "ParameterRanges": {
      "CategoricalParameterRanges": [],
      "ContinuousParameterRanges": [
        {
          "MaxValue": "1",
          "MinValue": "0",
          "Name": "eta",
        },
        {
          "MaxValue": "10",
          "MinValue": "1",
          "Name": "min_child_weight",
        },
        {
          "MaxValue": "2",
          "MinValue": "0",
          "Name": "alpha",            
        }
      ],
      "IntegerParameterRanges": [
        {
          "MaxValue": "10",
          "MinValue": "1",
          "Name": "max_depth",
        }
      ]
    },
    "ResourceLimits": {
      "MaxNumberOfTrainingJobs": 20,
      "MaxParallelTrainingJobs": 3
    },
    "Strategy": "Bayesian",
    "HyperParameterTuningJobObjective": {
      "MetricName": "validation:rmse",
      "Type": "Minimize"
    }
  }
from sagemaker.amazon.amazon_estimator import get_image_uri
training_image = get_image_uri(region, 'xgboost', repo_version='latest')
     
s3_input_train = 's3://{}/{}/train'.format(bucket, prefix)
s3_input_validation ='s3://{}/{}/validation/'.format(bucket, prefix)
    
training_job_definition = {
    "AlgorithmSpecification": {
      "TrainingImage": training_image,
      "TrainingInputMode": "File"
    },
    "InputDataConfig": [
      {
        "ChannelName": "train",
        "CompressionType": "None",
        "ContentType": "csv",
        "DataSource": {
          "S3DataSource": {
            "S3DataDistributionType": "FullyReplicated",
            "S3DataType": "S3Prefix",
            "S3Uri": s3_input_train
          }
        }
      },
      {
        "ChannelName": "validation",
        "CompressionType": "None",
        "ContentType": "csv",
        "DataSource": {
          "S3DataSource": {
            "S3DataDistributionType": "FullyReplicated",
            "S3DataType": "S3Prefix",
            "S3Uri": s3_input_validation
          }
        }
      }
    ],
    "OutputDataConfig": {
      "S3OutputPath": "s3://{}/{}/output".format(bucket,prefix)
    },
    "ResourceConfig": {
      "InstanceCount": 1,
      "InstanceType": "ml.m4.xlarge",
      "VolumeSizeInGB": 10
    },
    "RoleArn": role,
    "StaticHyperParameters": {
      "eval_metric": "rmse",
      "num_round": "100",
      "objective": "reg:linear",
      "rate_drop": "0.3",
      "tweedie_variance_power": "1.4"
    },
    "StoppingCondition": {
      "MaxRuntimeInSeconds": 43200
    }
}
smclient.create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tuning_job_name,
                                            HyperParameterTuningJobConfig = tuning_job_config,
                                            TrainingJobDefinition = training_job_definition)
smclient.describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name)['HyperParameterTuningJobStatus']
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)