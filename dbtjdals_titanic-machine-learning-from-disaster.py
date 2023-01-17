#essentials

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#visualizations

import matplotlib.pyplot as plt

import seaborn as sns



#machine learning

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
#read data

titanic_data = pd.read_csv('../input/titanic/train_and_test2.csv')
#check head of data

titanic_data.head()
#basic info

titanic_data.info()
#basic stats

titanic_data.describe()
#drop unneccessary columns

titanic_data= titanic_data.drop(['zero','zero.1','zero.2','zero.3','zero.4','zero.5','zero.6','zero.7','zero.8',

            'zero.9','zero.10','zero.11','zero.12','zero.13',

            'zero.14','zero.15','zero.16','zero.17','zero.18','Embarked','Parch'],axis=1)
#rename survived column

titanic_data.rename(columns = {'2urvived':'Survived'}, inplace = True)
#preview new data

titanic_data.head()
#plot number of survivors

sns.set_style('whitegrid')

sns.countplot(x='Survived',data=titanic_data,palette='RdBu_r')
#countplot of survival based on sex

sns.countplot(x='Survived',data=titanic_data,palette='RdBu_r',hue='Sex')
#countplot of survival based on class

sns.countplot(x='Survived',data=titanic_data,palette='rainbow',hue='Pclass')
#boxplot of age vs. class

plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=titanic_data,palette='rainbow')
#distribution of age

sns.distplot(titanic_data['Age'],kde=False)
#FacetGrid showing age distribution based on class and survival

g = sns.FacetGrid(data=titanic_data,col='Survived',row='Pclass')

g.map(sns.distplot,'Age')
sns.stripplot(x='Pclass',y='Age',data=titanic_data, jitter=True,hue='Survived',palette=['r','g'],dodge=True).set_title('Age Distribution on Survival')
#split test and train data

X_train, X_test, y_train, y_test = train_test_split(titanic_data.drop('Survived',axis=1), 

                                                    titanic_data['Survived'], test_size=0.30, 

                                                    random_state=101)
#assign logistic regression model to object

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
#fit training data to model

predictions = logmodel.predict(X_test)
#print classification report

print(classification_report(y_test,predictions))
#print confusion matrix

confusion_matrix(y_test,predictions)
predictions
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'Passengerid':X_test['Passengerid'],'Survived':predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)