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
import os
import path

# library used for plots
import seaborn as sns 
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

Path="../input/siim-isic-melanoma-classification/"
# Import the train and test csv files
Train_File= os.path.join(Path + 'train.csv')
Test_File = os.path.join(Path + 'test.csv')
# Create the DataFrame
TrainFile_DF = pd.read_csv(Train_File)
TestFile_DF = pd.read_csv(Test_File)
# Check the Train Data
TrainFile_DF.head()
# Check the Test Data
TestFile_DF.head()
# Check if null values present in train data frame
TrainFile_DF.isnull().values.any()
# Check the count of null values present in train data frame
TrainFile_DF.isnull().sum().sum()
# Number of nulls in Train Data set
print('Number of nulls in image_name column=', TrainFile_DF['image_name'].isnull().sum())
print('Number of nulls in patient_id column=', TrainFile_DF['patient_id'].isnull().sum())
print('Number of nulls in sex column=', TrainFile_DF['sex'].isnull().sum())
print('Number of nulls in age_approx column=', TrainFile_DF['age_approx'].isnull().sum())
print('Number of nulls in anatom_site_general_challenge column=', TrainFile_DF['anatom_site_general_challenge'].isnull().sum())
print('Number of nulls in benign_malignant column=', TrainFile_DF['benign_malignant'].isnull().sum())
# Number of nulls in Test Data set
print('Number of nulls in image_name column=', TestFile_DF['image_name'].isnull().sum())
print('Number of nulls in patient_id column=', TestFile_DF['patient_id'].isnull().sum())
print('Number of nulls in sex column=', TestFile_DF['sex'].isnull().sum())
print('Number of nulls in age_approx column=', TestFile_DF['age_approx'].isnull().sum())
print('Number of nulls in anatom_site_general_challenge column=', TestFile_DF['anatom_site_general_challenge'].isnull().sum())
fig,ax1=plt.subplots(figsize=(30, 15))
ax1 = plt.subplot(2, 2, 1)


df1 = TrainFile_DF.pivot_table(index='target', columns='sex', values='age_approx', aggfunc='mean')
sns.heatmap(df1)

ax2 = plt.subplot(2, 2, 2)
df2 = TrainFile_DF.pivot_table(index='target', columns='anatom_site_general_challenge', values='age_approx', aggfunc='mean')
sns.heatmap(df2)


ax3 = plt.subplot(2, 2, 3)
df2 = TrainFile_DF.pivot_table(index='target', columns='diagnosis', values='age_approx', aggfunc='mean')
sns.heatmap(df2)
#droping the rows where values for sex column is missing as this is a categorical column
TrainFile_DF.dropna(subset = ["sex"], inplace=True)

#replacing the rows where values for age column is missing with mean
TrainFile_DF['age_approx'].fillna((TrainFile_DF['age_approx'].mean()), inplace=True)


#replacing the rows where values for diagnosis and anatom_site_general_challenge column is missing with unknown
TrainFile_DF.fillna('unknown', inplace=True)

# creating a dict file  
sex = {'male': 0,'female': 1} 
anatom_site_general_challenge={'head/neck':1, 'lower extremity':2, 'oral/genital':3, 'palms/soles':4, 'torso':5, 'upper extremity':6,'unknown':7}
diagnosis={'atypical melanocytic proliferation':1, 'cafe-au-lait macule':2, 'lentigo NOS':3, 'lichenoid keratosis':4, 'melanoma':5, 'nevus':6,'seborrheic keratosis':7, 'solar lentigo':8,'unknown':9}




# Looping through dataframe 
TrainFile_DF.sex = [sex[value] for value in TrainFile_DF.sex] 
TrainFile_DF.diagnosis = [diagnosis[value] for value in TrainFile_DF.diagnosis] 
TrainFile_DF.anatom_site_general_challenge = [anatom_site_general_challenge[value] for value in TrainFile_DF.anatom_site_general_challenge]
sns.set()  #Set aesthetic parameters in one step.
sns.pairplot(TrainFile_DF,height=5 , hue='target',diag_kind='hist')