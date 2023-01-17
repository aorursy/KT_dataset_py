# This is the default Kaggle environment

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib as plt

import copy

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


cars_final_prediction = pd.read_csv("../input/umgc-data-650-spring-2020-challenge/cars-final-prediction.csv")

test = pd.read_csv("../input/umgc-data-650-spring-2020-challenge/cars-test.csv")

train = pd.read_csv("../input/umgc-data-650-spring-2020-challenge/cars-train.csv")



# to make things simpler, lets concatenate these two dataframes into one for processing

df=pd.concat([test,train])

df2 = pd.concat([test,train])
print('final data ',cars_final_prediction.shape)

print('test data ', test.shape)

print('train data', train.shape )

final=cars_final_prediction
# This shows I need to convert my categorical data into numeric data

# all of the data is type 'object'

df.info()
# rename the column 'class' as 'rating' to avoid problems

df.rename(columns={'class':'label'}, inplace=True)

df2.rename(columns={'class':'label'}, inplace=True)

final.rename(columns={'car.id':'carid'}, inplace=True)  # the period in the column name is also a pain
# check for null values

print('null values in data =', df.isnull().values.sum())

# The data is formatted as an object

# need to change that into a category

# then I can change to a numeric datatype

# Label encoding changes the order, I don't like that

# df["buying"] = df["buying"].astype('category')

# df["buying_cat"] = df["buying"].cat.codes

# so let's build a library to encode each variable 



cat_to_num = {

    "buying": {"low":0,"med":1,"high":2,"vhigh":3},

    "maint":{"low":0,"med":1,"high":2,"vhigh":3},

    "doors":{"2":0,"3":1,"4":2,"5more":3},

    "persons":{"2":0,"4":1,"more":2},

    "lug_boot":{"small":0,"med":1,"big":2},

    "safety":{"low":0,"med":1,"high":2},

    "label":{"unacc":0,"acc":1,"good":2,"vgood":3}

    

}



# this changes for all of the training data

df.replace(cat_to_num, inplace=True)



# this changes so I can run the classifier on the competition set

final.replace(cat_to_num, inplace=True)

df.head()
#take a look at the distribution of results

label_count = df2['label'].value_counts()

sns.set(style='darkgrid')

ax = sns.barplot(label_count.index, label_count.values, alpha=0.9)

ax.set_title('class range')

ax.set_ylabel('number of occurences',fontsize=12)

ax.set_xlabel('class category', fontsize=12)

#ax.show()
#calculate the percentage of unacceptable in the training data



unacc = df2['label'].value_counts()["unacc"]

#"unacc" in test['label'].values

print('total percentage of unacceptable rating is ', unacc/len(df2))
#create some comparison datasets for testing different machine learning approaches

from sklearn import preprocessing

df_norm = preprocessing.normalize(df) #the numbers are normalized between 0 and 1

df_stand = preprocessing.scale(df)  #the numbers are standardized along a normal distribution

print(df_norm)
# somehow scikit converts the data into an array, time to convert it back

df_norm = pd.DataFrame(data=df_norm)

df_norm.set_axis(['ID', 'buying', 'maint','doors','persons','lug_boot','safety', 'label'], axis=1, inplace=True)

df_norm.head()
# somehow scikit converts the data into an array, time to convert it back

df_stand = pd.DataFrame(data=df_stand)

df_stand.set_axis(['ID', 'buying', 'maint','doors','persons','lug_boot','safety', 'label'], axis=1, inplace=True)

df_stand.head()
# identify the factors for making the prediction

features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']



X_df =df[features]

y_df = df.label

X_final=final[features]



X_df.shape
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

XDF_train, XDF_test, ydf_train, ydf_test = train_test_split(X_df, y_df, test_size=0.3, random_state=1) # 70% training and 30% test

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
dtc = DecisionTreeClassifier()

dtc = dtc.fit(XDF_train, ydf_train)



y_pred = dtc.predict(XDF_test)

final_pred = dtc.predict(X_final)
print("Accuracy:",metrics.accuracy_score(ydf_test, y_pred))
print(y_pred)
output = pd.DataFrame({"carid":final.carid, 'class':final_pred})

output.to_csv('my_submission.csv', index=False)

print('upload successful')