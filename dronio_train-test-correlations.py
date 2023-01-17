import pandas as pd

from sklearn.cross_validation import train_test_split

from sklearn import svm 

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import numpy as np

from sklearn.naive_bayes import GaussianNB
dataframe = pd.read_csv("../input/Churn_Modelling.csv")

dataframe.head()
# The value "14" displayes the number of columns

dataframe.shape
#You may have noticed that we have got some string fields in dataset

#In order to make to predictions we should convert this fields to integer

#But first lets see the unique values of theese fields

print("Geo unique values",dataframe['Geography'].unique())

print("Gender unique values",dataframe['Gender'].unique())
#Field convertation

dataframe['Geography'].replace("France",1,inplace= True)

dataframe['Geography'].replace("Spain",2,inplace = True)

dataframe['Geography'].replace("Germany",3,inplace=True)

dataframe['Gender'].replace("Female",0,inplace = True)

dataframe['Gender'].replace("Male",1,inplace=True)
#Now check if everything was transformed correctly

print("Geo unique values",dataframe['Geography'].unique())

print("Gender unique values",dataframe['Gender'].unique())
#In order to understand what "X" best contributes to "Y" result we should use correlation matrix

import seaborn as sns

corr = dataframe.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.show()
X1 = dataframe['Balance'] 

X2 = dataframe['Age']

X3 = dataframe['Geography']

Y = dataframe['Exited']

X = dataframe[['Balance','Age','Geography']]

#See how range of X looks like 

X.head()
#Lets count the top 3 positive correlation parametrs

#The values are:(0.11853277,0.15377058,0.28532304)

print ('Balance and Exited',np.corrcoef(X1,Y),'\n')

print ('Age and Exited',np.corrcoef(X2,Y),'\n')

print ('Geography and Exited',np.corrcoef(X3,Y),'\n')
#Lets split the data

# I prefer to use train_test_split for cross-validation

# This peace will prove us if we have overfitting 

X_train, X_test, y_train, y_test = train_test_split(

  X, Y, test_size=0.4, random_state=0)
#Now train and find out the prediction rate 

clf = GaussianNB()

clf = clf.fit(X_train ,y_train)

clf.score(X_test, y_test)
#Male:1,Female:0

ax = sns.countplot(x="Gender", data=dataframe,

                   

                   facecolor=(0, 0, 0, 0),

                   linewidth=5,

                  edgecolor=sns.color_palette("dark", 3))

ax.set_title('Gender')

plt.show()
#France:1,Spain:2,Germany:3

ax = sns.countplot(x="Geography", data=dataframe,

                   

                   facecolor=(0, 0, 0, 0),

                   linewidth=5,

                  edgecolor=sns.color_palette("dark", 3))

ax.set_title('Geography')

plt.show()
#Another type of correlation matrix 

correlation = dataframe.corr()

plt.figure(figsize=(15,15))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')

plt.show()
#Using ensemble baggin tree classifier

from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(KNeighborsClassifier(),

                            max_samples=0.5, max_features=0.5)



bagging = bagging.fit(X_train ,y_train)

bagging.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=70,max_features = 3,max_depth=5,n_jobs=-1)

rf.fit(X_train ,y_train)

rf.score(X_test, y_test)