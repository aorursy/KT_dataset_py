# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

train = pd.read_excel('../input/food-cost/Data_Train.xlsx')

test = pd.read_excel('../input/food-cost/Data_Test.xlsx')

submission = pd.read_excel('../input/food-cost-submission/Sample_submission.xlsx')

train["source"] = "train"

test["source"] = "test"
df = pd.concat([train,test])
df.info()
#Investigating the entire dataset first

df.duplicated().sum()
df= df.drop_duplicates()
df.isna().sum()
# Data exploration for CITY

# CITY has 147 null values

#combining City and locality

df['Location']=df['CITY']+' '+df['LOCALITY']

df.drop(columns=['CITY','LOCALITY'])
df.dropna(subset=['Location'],inplace=True)


from fuzzywuzzy import process

 

names_array=[]

def match_names(wrong_names,correct_names):

    for row in wrong_names:

        x=process.extractOne(row, correct_names)

        if x[1]<60:

            names_array.append('Others')

        else:

            names_array.append(x[0])

    return names_array

  

#Wrong country names dataset



correct_names=['Bangalore','Thane',

'Hyderabad','Andheri',

'Delhi', 'Kerala',

'Chennai', 'Bandra',

'Mumbai', 'Telangana',

'Kochi', 

'Noida', 

'Gurgaon', 'Ernakulam',

'Faridabad', 'Ghaziabad',

'Secunderabad' ]

name_match=match_names(df.Location,correct_names)    



print(len(names_array))

df['Location']=names_array
cuisines_list=[]

for row in df['CUISINES']:

    cuisines_list.append(list(row.split(',')))



df['CUISINES']=cuisines_list
df['CUISINES'].isna().sum()
df_cuisines=df['CUISINES'].apply(lambda x: pd.Series(1, x))
title_list=[]

for row in df['TITLE']:

    title_list.append(list(row.split(',')))

df['TITLE']=title_list
df_title=df['TITLE'].apply(lambda x: pd.Series(1, x))
df_title.head()
# cleaning time - pending
df[df['RATING'].isna()]
df["RATING"] = df.groupby("CITY").RATING.transform(lambda x : x.fillna(x.mode()[0]))
df['RATING']=df['RATING'].str.extract('(\d+)').astype(float)
df['VOTES'].isna().sum()
df.VOTES.fillna('0',inplace=True)

df['VOTES']=df['VOTES'].str.extract('(\d+)').astype(float)
df.drop(columns='CITY',inplace=True)

df.drop(columns='LOCALITY',inplace=True)

df.drop(columns='CUISINES',inplace=True)
#df.drop(columns='CUISINES',inplace=True)

#df.drop(columns='CUISINES++',inplace=True)

#df.drop(columns='Location++',inplace=True)

#df.drop(columns='TITLE++',inplace=True)
df_City=pd.get_dummies(df['Location'])

df.drop(columns='Location',inplace=True)

df_City.head()
df = pd.concat([df,df_City,df_cuisines,df_title], axis=1)
df.drop(columns='TITLE',inplace=True)
df_column_category = df.select_dtypes(exclude=np.number).columns

df_column_category
#df.drop(columns='City found',inplace=True)

df.drop(columns='TIME',inplace=True)
df.fillna(0,inplace=True)


train_final = df[df.source=="train"]

test_final = df[df.source=="test"]
train_final.shape
train_final.drop(columns=["source"],inplace=True)
test_final.drop(columns=["source",'COST'],inplace=True)
train_X = train_final.drop(columns=["COST",'RESTAURANT_ID'])
train_Y = train_final["COST"]
test_X = test_final.drop(columns=["RESTAURANT_ID"])
train_X.fillna(0,inplace=True)

train_X.isna().sum()
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(train_X, train_Y)

dtrain_predictions = model.predict(train_X)
from sklearn.model_selection import cross_val_score

a = cross_val_score(model, train_X, train_Y, cv=5, scoring='neg_mean_squared_error')
#Print model report:

from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error, r2_score

print("\nModel Report")

print("RMSE : %.4g" % np.sqrt(mean_squared_error(train_Y.values, dtrain_predictions)))

    

#Predict on testing data:

test_X.fillna(0,inplace=True)

test_final["res_linear"] =  model.predict(test_X)
print('r2 train',r2_score(train_Y,dtrain_predictions))

#print('r2 test',r2_score(test_y,test_predict))
Linear_submission = test_final[["RESTAURANT_ID","res_linear"]]
Linear_submission.head(20)