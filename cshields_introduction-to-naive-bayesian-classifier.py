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
#The train data has features (independent variables) and targets (dependent variable).
#Feature examples are Name, Age, or Fare. The target is if the passenger survived.
train=pd.read_csv('../input/train.csv')
#We print the first 5 rows of the train dataset to show what this looks like:
ntrain = train.shape[0] #this gets the number rows in the traning dataset
print("Training data (",ntrain,"rows)")

#Display the data in Pandas: .head(n_rows) shows the first n rows of the DataFrame
display(train.head(10))

#The test dataset is used to test how well the classifier performs
#Test data only has features, the targets are empty and must be predicted
test=pd.read_csv('../input/test.csv')

#Lets looks at the test data...
ntest = test.shape[0]
print("Test data (",ntest,"rows), notice that the survived column (target) is missing!")
display(test.head(10))

df_all=pd.concat([train,test],axis=0)
#Use DataFrame.info() and print the results.
print(df_all.info())
#First we get the median age by calling DataFrame.median() on the 'Age' column
age_med=df_all['Age'].median()

#Print the Median Age
print('Median Age = {}'.format(age_med))

#Fill in the missing ages with the median values and overwrite previous column
df_all['Age']=df_all['Age'].fillna(age_med)

display(df_all.head(10))
#Replace the strings in the 'Sex' column with numbers
df_all['Sex']=df_all['Sex'].replace('male',0)
df_all['Sex']=df_all['Sex'].replace('female',1)

#Look at the changed column
display(df_all.head())
#Import the package
import seaborn as sns
#Visualize the training data to differentiate between passengers who survived
#Split the data
train=df_all[:ntrain]

#Plot the data
sns.pairplot(train,vars=['Age','Sex','Pclass',],hue='Survived',)
#Load the model from the sklearn package
#See http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB

#Now call the model so we can train it
clf=GaussianNB()

#Train the model
#First, separate the features and target
y_train=train['Survived'].values
x_train=train[['Age','Sex','Pclass']].values

#Now train the model on the features and target
clf.fit(X=x_train,y=y_train)

#Check accuracy on training set
#Technically, we should use cross-validation to check accuracy.
#Cross-validation helps to prevent overfitting the model by chasing training accuracy
print('Bayesian Classifier Score = {}'.format(clf.score(X=x_train,y=y_train)))

#Now we predict the test set values
#First we get the test values
test_df=df_all[ntrain:]
x_test=test_df[['Age','Sex','Pclass']].values

#Now predict our results
results=clf.predict(x_test)
#Convert the results to int datatypes (real numbers)
results=[int(i) for i in results]

#Get passenger id's from test set with the .iloc command
results_id=df_all['PassengerId'].iloc[ntrain:].values

#Create a dataframe for submission
submission=pd.DataFrame({'PassengerId':results_id,'Survived':results})

#Check what the submission looks like
display(submission.head(10))

#Save the dataFrame as a .csv (save to Kaggle)
submission.to_csv('submisison.csv',index=False)

#Now complete steps 2, 3, and 4 to submit for scoring!
