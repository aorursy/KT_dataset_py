# Import the necessary libraries 

import os

import numpy as np 

import pandas as pd 

from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# For This Hiring hackathon i'm using kaggle environment

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#loading train,test and sample data



train=pd.read_csv("/kaggle/input/Dataset/Train.csv")

test = pd.read_csv("/kaggle/input/Dataset/Test.csv")

sample=pd.read_csv("/kaggle/input/Dataset/sample_submission.csv")
# checking the shapes

print("test shape  = ",test.shape)

print("train shape = ",train.shape)

print("sample shape =",sample.shape)
#combining train and test dataset to generate new data set "df" data preparation



test['MULTIPLE_OFFENSE']=np.nan

train['istrain']=1

test['istrain']=0

INCIDENT_ID=test['INCIDENT_ID']

test=test[train.columns]

df=pd.concat([train,test],axis=0)

print("Combinded shape = ",df.shape)
# here i'm writing a function/method to extract the week of month

from math import ceil



def week_of_month(dt):

    """ Returns the week of the month for the specified date.

    """



    first_day = dt.replace(day=1)



    dom = dt.day

    adjusted_dom = dom + first_day.weekday()



    return int(ceil(adjusted_dom/7.0))
# from Date column i'm extracting more features like day,month,week of month,weekday and year



df['DATE']=pd.to_datetime(df['DATE'],format='%d-%b-%y') 

df['day'] = df['DATE'].dt.day

df['month'] = df['DATE'].dt.month

df['wom']=df['DATE'].apply(lambda x: week_of_month(x))

df['weekday'] = df['DATE'].dt.weekday

df['year'] = df['DATE'].dt.year



# dropping the date column as i extraced some usefulfeatures from this

df.drop(['DATE'],axis=1,inplace=True)  
# seperating train and test dataset after data preparation



train=df[df['istrain']==1]

del train['istrain']

test=df[df['istrain']==0]

test.drop(['MULTIPLE_OFFENSE','istrain'],axis=1,inplace=True)

#checking for the class imbalance



print('No hack (Target:0)', round(train['MULTIPLE_OFFENSE'].value_counts()[0]/len(train) * 100,2), '% of the dataset')

print('hack (Target:1)', round(train['MULTIPLE_OFFENSE'].value_counts()[1]/len(train) * 100,2), '% of the dataset')



# from output we can confirm that class imbalance exists in the given data ( 4.5% is no-hack i.e '0' and 95.5% is hack i.e '1')



## whenever we have class imbalance we need to take  care the below things:-

### we need do stratify split our  train and test data for accurate validation

### Need to maintain class balance in train data by either oversampling or under sampling

### here in this hackathon i'm not doing any oversampling or under sampling because many advance algorithm has inbuilt feature to handle this class imbalance problem

# dropping duplicate rows in train dataset



train.drop(['INCIDENT_ID'],axis=1,inplace=True)



print("Train shape before dropping duplicates -->",train.shape)



train.drop_duplicates(keep='first',inplace=True)

print("Train shape after dropping duplicates -->",train.shape)
# checking for null values in data frame (train+test)

print(df.drop(['MULTIPLE_OFFENSE'],axis=1).isnull().sum())



# From output we have a total of 309 null values for X_12 variable (train and test dataset combined)
# Now lets check for the pattern in null values 

df.loc[(df['X_12'].isnull(),'year')].value_counts()



# from output, it's clear that null values exists only for year 2017 and 2018



## Below few things can be followed for handling null values for the given data set



# 1) The null rows in the train data can deleted ( works best when we have a few null rows compared to entire dataset) 

# 2) Impute the null values using column mean/mode/median values ( mode is more suitable for our given dataset)

# 3) Imputing null values by machine learning algorithms like KNN

# 4) Many of the advance machine learning algorithms have inbuilt functionalities to handle null values



# importing missing values using ffill/bfill



#train['X_12'] = train['X_12'].ffill()

#test['X_12'] = test['X_12'].ffill()

#train['X_12'] = train['X_12'].bfill()

#test['X_12'] = test['X_12'].bfill()
train.info() # displays the column name,not-null count and their datatypes
train.describe() # gives the distribution of the train dataset
# visualing the frequency counts of columns X_1,X_2....X_9

fig, axarr = plt.subplots(3, 3, figsize=(22,13))



train['X_1'].value_counts().sort_index().plot.bar(ax=axarr[0][0])

axarr[0][0].set_title('X_1', fontsize=18)

train['X_2'].value_counts().sort_index().plot.bar(ax=axarr[0][1])

axarr[0][1].set_title('X_2', fontsize=18)

train['X_3'].value_counts().sort_index().plot.bar(ax=axarr[0][2])

axarr[0][2].set_title('X_3', fontsize=18)

train['X_4'].value_counts().sort_index().plot.bar(ax=axarr[1][0])

axarr[1][0].set_title('X_4', fontsize=18)

train['X_5'].value_counts().sort_index().plot.bar(ax=axarr[1][1])

axarr[1][1].set_title('X_5', fontsize=18)

train['X_6'].value_counts().sort_index().plot.bar(ax=axarr[1][2])

axarr[1][2].set_title('X_6', fontsize=18)

train['X_7'].value_counts().sort_index().plot.bar(ax=axarr[2][0])

axarr[2][0].set_title('X_7', fontsize=18)

train['X_8'].value_counts().sort_index().plot.bar(ax=axarr[2][1])

axarr[2][1].set_title('X_8', fontsize=18)

train['X_9'].value_counts().sort_index().plot.bar(ax=axarr[2][2])

axarr[2][2].set_title('X_9', fontsize=18)

# visualing the frequency counts of columns X_10,X_11....X_15

fig, axarr = plt.subplots(2, 3, figsize=(22,8))



train['X_10'].value_counts().sort_index().plot.bar(ax=axarr[0][0])

axarr[0][0].set_title('X_10', fontsize=18)

train['X_11'].value_counts().head(15).sort_index().plot.bar(ax=axarr[0][1])

axarr[0][1].set_title('X_11', fontsize=18)

train['X_12'].value_counts().sort_index().plot.bar(ax=axarr[0][2])

axarr[0][2].set_title('X_12', fontsize=18)

train['X_13'].value_counts().head(15).sort_index().plot.bar(ax=axarr[1][0])

axarr[1][0].set_title('X_13', fontsize=18)

train['X_14'].value_counts().head(15).sort_index().plot.bar(ax=axarr[1][1])

axarr[1][1].set_title('X_14', fontsize=18)

train['X_15'].value_counts().sort_index().plot.bar(ax=axarr[1][2])

axarr[1][2].set_title('X_15', fontsize=18)

# Displaying the unique values in the each column

cols= [ 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8',

       'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15',

       'MULTIPLE_OFFENSE', 'day', 'month', 'wom', 'weekday', 'year']

for col in cols:

    print("unique values in column ",col," is -->",train[col].nunique())
#displays hack and no_hack counts grouped by year

train.groupby(['year','MULTIPLE_OFFENSE'])['MULTIPLE_OFFENSE'].count()
# plotting the MULTIPLE_OFFENSE count with respect to years



fig, axarr = plt.subplots(figsize=(22,8))

sns.countplot(x='year',data=train,hue='MULTIPLE_OFFENSE')
#x = train.drop(['MULTIPLE_OFFENSE'], axis=1)

#y = train['MULTIPLE_OFFENSE']



#sm = RandomOverSampler(random_state=5,sampling_strategy='not majority')

#x_sm, y_sm = sm.fit_resample(x,y)

#x_sm = pd.DataFrame(x_sm)

#x_sm.columns = x.columns

#train=pd.concat([x_sm,y_sm],axis=1,)
import h2o

from h2o.automl import H2OAutoML



# learn more about h2o in the given url "https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html"
h2o.init(nthreads=-1,max_mem_size='16g',min_mem_size='12g')
#load data as h2o frames



train = h2o.H2OFrame(train)

test = h2o.H2OFrame(test)
train_x=train.drop(['MULTIPLE_OFFENSE'],axis=1)

x = train_x.columns

y ='MULTIPLE_OFFENSE'



train['MULTIPLE_OFFENSE']=train['MULTIPLE_OFFENSE'].asfactor()
# Run automl to train the model

aml_ti = H2OAutoML(max_runtime_secs=4000,max_models=None,seed= 7,nfolds=10,include_algos = ["GBM", "DeepLearning","XGBoost","DRF","StackedEnsemble"])

aml_ti.train(x = x, y = y,training_frame =train)
#multiples models and their auc scores are displayed in sorted order

lb_ti = aml_ti.leaderboard

lb_ti
# Get the first model which have high accuracy 

m = h2o.get_model(lb_ti[1,"model_id"])

m
# displays the variable importance of the dataset

m.varimp(use_pandas=True)
# visualizing the variable importance of the dataset

m.varimp_plot()
# Doing predictions on the test data

pred=aml_ti.leader.predict(test)

test_pred = pred.as_data_frame()
test_pred.head()
result=pd.concat([INCIDENT_ID,test_pred['predict']],axis=1,ignore_index=True)
#generating output file

result=result.rename(columns={0: "INCIDENT_ID", 1: "MULTIPLE_OFFENSE"})

result.to_csv("saikumar_ganneboyina.csv",index=False)

result.head()
# save the model

model_path = h2o.save_model(model=m,force=True)

print(model_path)
# load the model

saved_model = h2o.load_model(model_path)

print(saved_model)