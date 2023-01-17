# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
i = 0
#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        i = i+1
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
input_data_path=os.path.join('..','input','data')
stored_data_path = os.path.join('..','input','resnet-weights')
csv_filename='Data_Entry_2017.csv'
all_xray_df = pd.read_csv(os.path.join(input_data_path,csv_filename))
mask = all_xray_df['Finding Labels']!='No Finding' # set all 'No Finiding' labels to 0 and rest to 1
ctr = 0
for i in range(mask.shape[0]):
    if mask[i]==0:
        ctr+=1
    if ctr%5==0:
        mask[i]=1  # select every 5th 'No Finding' label
# No Finding class reduced to 20%
all_xray_df = all_xray_df[mask].copy(deep=True)
all_image_paths = {os.path.basename(f): f  for f in glob(os.path.join(input_data_path,'images*','*','*.png'))   }  
# create a dict mapping image names to their path
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)
all_labels=set()
def sep_diseases(x):
    list_diseases=x.split('|')
    for item in list_diseases:
        all_labels.add(item)
    return list_diseases
# Since the image may contain multiple disease labels
# Create a list of all disesases and append a new column named output to the x_ray dataframe
all_xray_df['disease_vec']=all_xray_df['Finding Labels'].apply(sep_diseases)
all_labels=list(all_labels)
all_labels.remove('No Finding')
all_labels.sort()
disease_freq={}
for sample in all_xray_df['disease_vec']:
    for disease in sample:
        if disease in disease_freq:
            disease_freq[disease]+=1
        else:
            disease_freq[disease]=1
print(disease_freq)
for label in all_labels:
    all_xray_df[label]=all_xray_df['disease_vec'].apply(lambda x: float(label in x))
all_xray_df.loc[:,'disease_vec':]
from sklearn.model_selection import train_test_split
# 15% of the data will be used for testing of model performance
# random state is set so as to get the same split everytime
train_df, test_df = train_test_split(all_xray_df,test_size = 0.15, random_state = 2020)
print('Number of training examples:', train_df.shape[0])
print('Number of validation examples:', test_df.shape[0])