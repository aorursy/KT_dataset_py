# import module

import numpy as np

import pandas as pd # for data processing 

import seaborn as sns # for plotting 

import matplotlib.pyplot as plt

%matplotlib inline 
# read in data 

df=pd.read_csv('../input/HR_comma_sep.csv')
# check data 

df.shape
df.columns
df.dtypes
df.isnull().any()
df.head()
# convert object ('sales' & 'salary') into number

# the 'left' column should be results for prediction

df.left.value_counts()   # check if data is unbalanced 
y=np.array(df['left'])  # transform data into array for classification 
# check how each columns affect or related with df['left']

# first convert 'sales' and 'salary' into number 

sale=df['sales']; salary=df['salary']

salemapkey={'sales':0,'technical':1,'support':2,'IT':3,'product_mng':4,

        'marketing':5,'RandD':6,'accounting':7,'hr':8,'management':9}

salarymapkey={'low':0,'medium':1,'high':2}
sale_2num=sale.map(salemapkey)

saralry_2num=salary.map(salarymapkey)
# drop old columns and add new column 

df2=df.drop(['sales','salary'],axis=1)
df2['job']=sale_2num
df2['salary']=saralry_2num
# use heat map to check relationship between each column 

sns.heatmap(df2.corr(),cmap='bone',annot=True,fmt='.2f')
# left is anticorrelated with satisfaction, then salary and work_accident

# train data with tree classification, for this data, we will use Randomforest 

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
# convert into array

x=np.array(df2[['satisfaction_level','last_evaluation',

               'number_project','average_montly_hours',

               'time_spend_company','Work_accident','promotion_last_5years',

               'job','salary']])
y=np.array(df2['left'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# normalize data 

sc=StandardScaler()

sc.fit(x_train)

x_train_std=sc.transform(x_train)

sc.fit(x_test)

x_test_std=sc.transform(x_test)
model=RandomForestClassifier(criterion='entropy',max_depth=None)

model.fit(x_train_std,y_train)
prediction=model.predict(x_test_std)

accuracy_score(y_test,prediction)