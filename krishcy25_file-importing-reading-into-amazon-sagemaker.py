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
import pandas as pd
import numpy as np

#Reading files into Amazon SageMaker Notebook Instance
#Assumptions- All the data files need to be uploaded into S3 bucket and with the role
#Role created above need to be used while creating the SageMaker Notebook instance

!pip install s3fs #This line of code need to be executed only during the first run and can be avoided for subsequent runs

#Boto3 is the name of the Python SDK for AWS. It allows you to directly create, update, and delete AWS resources from your Python scripts.
import boto3
client=boto3.client('s3')

#Assuming we have the following 4 files with in our S3 bucket: train_identity.csv, train_transaction.csv, test_identity.csv, test_transaction.csv
#fraudml is the name of the bucket with in S3
path1='s3://fraudml/train_identity.csv'
path2='s3://fraudml/train_transaction.csv'
path3='s3://fraudml/test_identity.csv'
path4='s3://fraudml/test_transaction.csv'

train_identity=pd.read_csv(path1)
train_transaction=pd.read_csv(path2)
test_identity=pd.read_csv(path3)
test_transaction=pd.read_csv(path4)

#Now you have 4 dataframes which can be used for further data preparation, model building