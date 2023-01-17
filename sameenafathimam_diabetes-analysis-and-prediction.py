# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.describe()
df.shape
df.dtypes
df.isnull().sum().any()
#Though there is no null value certain columns have 0's which is considered to be NaN values

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN) 
df.isnull().sum()
#Replacing NaN values with their median value

for column in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:

    df[column].replace(np.nan,df[column].median(),inplace = True)

df.head()
df.Outcome.value_counts()
# Grouping predictor variables by target variable

df.groupby("Outcome")[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]].agg(['max','min','mean'])
sns.countplot(x="Outcome", data=df)

#Let's see how the features are correlated

plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(),annot=True)

plt.title("Correlation heatmap")
df['Age_Group'] = pd.cut(df['Age'],

                         [10,20,30,40,50,60],

                         labels=['11-20','21-30','31-40','41-50','51+'])

fig,ax = plt.subplots(figsize=(8,6))

sns.countplot(data=df,x = 'Age_Group',hue= 'Outcome',ax=ax)

plt.title('Age vs Outcome')

df.drop(["Age_Group"],axis = 1,inplace = True)
fig, ax = plt.subplots(4,2, figsize=(16,16))

sns.kdeplot(data=df['Age'],color = 'r',shade = True,ax=ax[0][0])

sns.kdeplot(data=df['Pregnancies'],color = 'b',shade = True,ax=ax[0][1])

sns.kdeplot(data=df['Glucose'], color = 'r',shade = True,ax=ax[1][0])

sns.kdeplot(data=df['BloodPressure'],color = 'b',shade = True,ax=ax[1][1])

sns.kdeplot(data=df['SkinThickness'],shade = True,color = 'r',ax=ax[2][0])

sns.kdeplot(data=df['Insulin'],shade = True,color = 'b',ax=ax[2][1])

sns.kdeplot(data=df['DiabetesPedigreeFunction'],shade = True,color = 'r',ax=ax[3][0])

sns.kdeplot(data=df['BMI'],shade = True,color = 'b',ax=ax[3][1])

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeRegressor
X = df.drop(["Outcome"], axis = 1)

y = df["Outcome"]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.30, random_state = 1)
classifier = DecisionTreeRegressor(random_state = 1)



# fit your model

classifier.fit(train_X, train_y)



#prediction

predicted_values = classifier.predict(test_X)



cm = confusion_matrix(test_y,predicted_values)

print(cm)

accuracy_score(test_y, predicted_values)

classifier =  LogisticRegression(random_state = 1,max_iter = 150)



# fit your model

classifier.fit(train_X, train_y)



#prediction

predicted_values = classifier.predict(test_X)



cm = confusion_matrix(test_y,predicted_values)

print(cm)

accuracy_score(test_y, predicted_values)
