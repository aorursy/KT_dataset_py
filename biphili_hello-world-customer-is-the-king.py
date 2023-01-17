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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
df.describe().T
df.SeniorCitizen.unique()
df.tenure.unique()
len(df.MonthlyCharges.unique())
# Summary of Dataset 

print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df["Churn"].value_counts(sort = False)
# Creating a copy of the data 

df_copy = df.copy()
df_copy.drop(['customerID','MonthlyCharges','TotalCharges','tenure'],axis=1,inplace = True)

df_copy.head()
summary = pd.concat([pd.crosstab(df_copy[x],df_copy.Churn) for x in df_copy.columns[:-1]], keys= df_copy.columns[:-1])

summary
summary['Churn_Percentage'] = summary['Yes']*100/(summary['No'] + summary['Yes'])

summary
from pylab import rcParams # For customizing the plots



# Data to plot

labels = df['Churn'].value_counts(sort= True).index

sizes = df['Churn'].value_counts(sort = True)



colors = ["lightgreen","red"]

explode = (0.05,0)  # explode first slize



rcParams['figure.figsize'] = 7,7



#plot

plt.pie(sizes, explode = explode,labels = labels,colors=colors,autopct='%1.1f%%',shadow = True,startangle=90)



plt.title('Customer churn breakdown')

plt.show()
g = sns.factorplot(x='Churn',y="MonthlyCharges",data=df,kind="violin",palette="spring")
g = sns.factorplot(x='Churn',y="tenure",data=df,kind="violin",palette="spring")
# Removing blank space in our date 

len(df[df["TotalCharges"] == ""])

df = df[df["TotalCharges"] != " "]
# Dropping missing values



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



# Customer id col

Id_col = ['customerID']



#Target columns

target_col = ["Churn"]



#categorical columns 

cat_cols = df.nunique()[df.nunique() < 6].keys().tolist()

cat_cols = [x for x in cat_cols if x not in target_col]



#numerical columns

num_cols = [x for x in df.columns if x not in cat_cols + target_col + Id_col]



#Binary columns with 2 values

bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()



#Columns more than 2 values 

multi_cols = [i for i in cat_cols if i not in bin_cols]       



#Label encoding Binary columns 

le = LabelEncoder()

for i in bin_cols :

    df[i] = le.fit_transform(df[i])



#Duplicating columns for multi value columns

df = pd.get_dummies(data = df ,columns = multi_cols)

df.head()
len(df.columns)
num_cols
# Scaling Numerical columns 

std = StandardScaler()



# Scale data

scaled = std.fit_transform(df[num_cols])

scaled = pd.DataFrame(scaled,columns= num_cols)



#dropping original values merging scaled values for numerical values 

df_telecom_og = df.copy()

df = df.drop(columns = num_cols,axis =1)

df = df.merge(scaled,left_index = True,right_index=True,how='left')



#df.info()

df.head()
df.drop(["customerID"],axis=1,inplace=True)

df.head()
df[df.isnull().any(axis=1)]
df = df.dropna()
# Double check that nulls been removed 

df[df.isnull().any(axis=1)]
from sklearn.model_selection import train_test_split



# We remove the label values from our training data

X = df.drop(['Churn'],axis=1).values



# We have to get the matrix of target variables

y = df["Churn"].values
# Spliting the dataset



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
df_train = pd.DataFrame(X_train)

df_train.head()
type(X_train)
print(len(df.columns))

df.columns
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = LogisticRegression()

model.fit(X_train,y_train)



predictions = model.predict(X_test)

score = model.score(X_test,y_test)



print("Accuracy =" + str(score))

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

# We will be trying the find the parameters which are most important for our prediction

coef = model.coef_[0]

coef = [abs(number) for number in coef]

print(coef)
# Finding and deleting the label columns

cols = list(df.columns)

cols.index("Churn")
del cols[6]

cols
# Sorting on Feature Importance 

sorted_index = sorted(range(len(coef)),key = lambda k:coef[k],reverse = True)

for idx in sorted_index:

    print(cols[idx])
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model_rf = RandomForestClassifier()

model_rf.fit(X_train,y_train)



predictions = model_rf.predict(X_test)

score = model.score(X_test,y_test)



print("Accuracy =" + str(score))

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

import pickle 



#Save

with open('model.pkl','wb') as f:

    pickle.dump(model_rf,f)

    

#Load 

with open('model.pkl','rb') as f:

    loaded_model_rf = pickle.load(f)