#basic python libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

#File libraries
import os
import zipfile as z
print(os.listdir("../input"))

#Viz Libraries
import matplotlib.pyplot as plt
import plotly as pl
%matplotlib inline

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

period_train = pd.read_csv('../input/periods_train.csv')
period_test = pd.read_csv('../input/periods_test.csv')

train_active = pd.read_csve('../input/train_active.csv')
test_active = pd.read_csve('../input/test_active.csv')

train_dir = '../input/train_images'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    
train_extract = z.ZipFile('../input/train_jpg.zip', 'r')
train_extract.extractall(train_dir)
train_extract.close()


test_extract = z.ZipFile('../input/test_jpg.zip', 'r')
test_extract.extractall('../input/test_images')
test_extract.close()

print(os.listdir("../input"))
train_data.head(5)
test_data.head(5)
period_train.head(5)
period_test.head(5)
print(train_data.shape,'is the train data dimension')
print(test_data.shape,'is the test data dimension')
print(period_train.shape,'is the period data for training samples')
print(period_test.shape,'is the period data for testing samples')
train_data.info()
train_columns = train_data.columns
train_num_cols = train_data._get_numeric_data().columns
print('Numerical Columns of train_data:', train_num_cols)
#Index([u'0', u'1', u'2'], dtype='object')

print('Categorical Columns of train_data : ', sorted(list(set(train_columns) - set(train_num_cols))))
test_columns = test_data.columns
test_num_cols = test_data._get_numeric_data().columns
print('Numerical Columns of test_data:', test_num_cols)

print('Categorical Columns of train_data : ', sorted(list(set(test_columns) - set(test_num_cols))))
train_data.describe()
train_data.describe(include=["O"])
test_data.info()
test_data.describe()
test_data.describe(include=["O"])
def get_percentage(series):
    num = series.isnull().sum()
    den = len(series) 
    return (num/den)*100
train_data_empty = train_data[train_data.columns[train_data.isnull().any()].tolist()]
print('percentage of null values in train data: \n\n', get_percentage(train_data_empty))
test_data_empty = test_data[test_data.columns[test_data.isnull().any()].tolist()]
print('percentage of null values in test data: \n\n', get_percentage(test_data_empty))
