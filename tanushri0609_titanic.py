#Import the files
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

#Read the train dataset from titaic directory
train = pd.read_csv('../input/titanic/train.csv', sep=',', header=0)
train.head()
#Read the test dataset from the titanic directory
test = pd.read_csv('../input/titanic/test.csv', sep=',', header=0)
test.head()
#Looking for null data in train dataset. Here in column Age, Cabin and Embarked has null data.
train.info()
#Looking for null data in test dataset. Here in column Age, Fare and Cabin has null data.
test.info()
# Remove columns like PassengerID, Ticket, Name and Embarked fron the data frame.
train_mod=train.drop(['PassengerId','Ticket','Name','Embarked'] , axis=1)
test_mod=test.drop(['PassengerId','Ticket','Name','Embarked'] , axis=1)
# In column Sex replacing the 'Female' with 0 and 'Male' with 1
train_mod.replace({'female':0, 'male': 1}, inplace=True)
test_mod.replace({'female':0, 'male': 1}, inplace=True)
train_mod.head()
# In column 'Cabin' replacing the null value with 0 and cabin number with 1
train_mod['Cabin'] = np.where(train_mod['Cabin'].isnull(), 0, 1)
test_mod['Cabin'] = np.where(test_mod['Cabin'].isnull(), 0, 1)
train_mod.head()
#In column 'Age' fill the null value with the average value of the column.
train_mod['Age'].fillna((train_mod['Age'].mean()), inplace= True)
test_mod['Age'].fillna((test_mod['Age'].mean()), inplace= True)
train_mod.head()
# In column 'Fare' fill the null value with the average value of the column
test_mod['Fare'].fillna((test_mod['Fare'].mean()), inplace= True)

#check the description of test data set
test_mod.describe()

# check the description of train data set
train_mod.describe()
# Find the correlation of the train data set 
sb.heatmap(train_mod.corr(),annot=True)
plt.show()
# Divide the train data set into x and y for training the Logistic Regression model
y=train_mod['Survived']
X=train_mod.drop('Survived', axis=1)
X.head()
# Run Logistic Regression Model and make prediction on test data set
classifier=LogisticRegression()
classifier.fit(X, y)
Final_pred=classifier.predict(test_mod)
# Print the final prediction
print(Final_pred)
# Add the final prediction in test data set in Survived column
test['Survived']=Final_pred
test.head()
# Prepare the output data set
final_frame=test.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'] , axis=1)
final_frame.head()
# Store the output dataset in submission_file.csv file
final_frame.to_csv('/kaggle/working/submission_file.csv')