# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/titanic/train.csv')





data.head()
for i in range(len(data)):

    data.loc[i,'name_title']=data.loc[i,'Name'].split(',')[1].split('.')[0]
data.head()
for i in range(len(data)):

    if(pd.isnull(data.loc[i,"Age"])):

        temp_df=pd.DataFrame(data[data["name_title"]==data.loc[i,"name_title"]])

        data.loc[i,"Age"]=temp_df["Age"].mean()

        data.loc[i,"Age"]=int(data.loc[i,"Age"])

        print(data.loc[i,"Age"])
data.columns
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

lb_make = LabelEncoder()

data["gender_code"] = lb_make.fit_transform(data["Sex"])

pred_vars=['PassengerId','Pclass',"gender_code","Age","SibSp","Parch"]

outcome=['Survived']

# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = DecisionTreeClassifier()

# Train the model on training data

rf.fit(data[pred_vars], data['Survived']);
#Prediction

import pandas as pd

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

test_data["gender_code"] = lb_make.fit_transform(test_data["Sex"])

for i in range(len(test_data)):

    test_data.loc[i,'name_title']=test_data.loc[i,'Name'].split(',')[1].split('.')[0].rstrip().lstrip()

    print(test_data.loc[i,'name_title'])


df=data.groupby('name_title' ,as_index=False)['Age'].mean()

for i in range(len(df["name_title"])):

    df.loc[i,"name_title"]=df.loc[i,"name_title"].lstrip()

#df2=df[df['name_title']=='Capt']

#c=df[df["name_title"]=="Mr"]

#c.Age

df.loc[0,"Age"]

u_name=data['name_title'].unique()



age_dict={}

c=0

for i in u_name:

    ag=df[df["name_title"]==i]

    

    age_dict[i]=ag.Age

    print(ag.Age)

    c=c+1

age_dict

for i in range(len(test_data)):

    if(pd.isnull(test_data.loc[i,"Age"]) or pd.isna(test_data.loc[i,"Age"])):

        #temp_df=pd.DataFrame(test_data[df["name_title"]==test_data.loc[i,"name_title"]])

        #test_data.loc[i,"Age"]=temp_df["Age"].mean()

        #print(temp_df["Age"].mean())

        #test_data.loc[i,"Age"]=int(test_data.loc[i,"Age"])

        for j in range(len(df["name_title"])):

            if(test_data.loc[i,"name_title"]==df.loc[j,"name_title"]):

                test_data.loc[i,"Age"]=df.loc[j,"Age"]

                



print(test_data["Age"])
predictions=rf.predict(test_data[pred_vars])

len(predictions)
check_data=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

y_true=check_data['Survived']

y_true
predictions=pd.DataFrame(predictions)

predictions.iloc[:,0]
y_true["Survived"]
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_true, predictions))