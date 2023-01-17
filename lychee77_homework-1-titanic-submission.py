#Load data
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Drop features we are not going to use
train = train.drop(['Name','SibSp', 'Ticket', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp', 'Ticket', 'Cabin', 'Embarked'],axis=1)

#Look at the first 3 rows of our training data
train.head(5)
#Convert ['male','female'] to [1,0] so that our decision tree can be built
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    
train.head(4)
test.head(4)
#See the missing vaule of each variable
train.isnull().sum()
test.isnull().sum()
import numpy as np
TrainAgeMean = np.mean(train['Age'])
TestAgeMean = np.mean(test['Age'])
TestFareMean = np.mean(test['Fare'])
print(TrainAgeMean,TestAgeMean,TestFareMean)
#Fill in missing age values with mean 
train['Age'] = train['Age'].fillna(TrainAgeMean)
test['Age'] = test['Age'].fillna(TestAgeMean)
test['Fare'] = test['Fare'].fillna(TestFareMean)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary','Parch','Fare']
target = 'Survived'

#Look at the first 3 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train[features].head(3)
#Display first 3 target variables
train[target].head(3).values
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Create classifier object with default hyperparameters
#clf = DecisionTreeClassifier()  
clf = RandomForestClassifier()

#Fit our classifier using the training features and the training target values
clf.fit(train[features],train[target]) 
#Make predictions using the features from the test data set
predictions = clf.predict(test[features])

#Display our predictions - they are either 0 or 1 for each training instance 
#depending on whether our algorithm believes the person survived or not.
predictions
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()
#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions 2.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
import seaborn as sns
cpldata = pd.merge(test, submission,on='PassengerId')
cpldata.head(5)
sns.lmplot('Pclass','Survived',data = cpldata)
#features = ['Pclass','Age','Sex_binary','Parch','Fare']
sns.lmplot('Fare','Survived',data = cpldata)
sns.lmplot('Age','Survived',data = cpldata)
sns.lmplot('Sex_binary','Survived',data = cpldata)