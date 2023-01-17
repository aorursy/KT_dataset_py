# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
     #   print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATASET_DIR = "../input/covid-detection-project/"
metadata_path = str(DATASET_DIR) + 'metadata.csv'
metadata = pd.read_csv(metadata_path)
metadata.shape
metadata.Disease.unique()
covid_data=metadata[metadata['Disease']=='COVID-19']
covid_path= str(DATASET_DIR)+ 'Covid'
covid_list=[]
for entry in os.listdir(covid_path):
    covid_list.append(entry)
not_list=[]
for name in covid_data['Filename']:
    if name in covid_list:
        pass
    else: 
        print(name)


normal_data=metadata[metadata['Disease']=='normal']
normal_path= str(DATASET_DIR)+ 'Normal'
normal_list=[]
for entry in os.listdir(normal_path):
    normal_list.append(entry)
not_list=[]
for name in normal_data['Filename']:
    if name in normal_list:
        pass
    else: 
        print(name)

pmonia_data=metadata[metadata['Disease']=='pneumonia']
pmonia_path= str(DATASET_DIR)+ 'Pneumonia'
pmonia_list=[]
for entry in os.listdir(pmonia_path):
    pmonia_list.append(entry)
not_list=[]
for name,uploader in zip(pmonia_data['Filename'],pmonia_data['Uploader']):
    if name in pmonia_list:
        pass
    else: 
        not_list.append(name)

len(not_list)
len(pmonia_list)
print(not_list)

for name2 in zip(pmonia_data['Filename']):
    if name2 in not_list:
        name2=name2[:-1]
        
        