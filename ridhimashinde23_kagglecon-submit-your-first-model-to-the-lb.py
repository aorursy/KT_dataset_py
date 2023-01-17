# This Python 3 environment comes with many helpful analytics libraries installed.

# Let's import few libraries an



import os

print("hello kaggle")

print("Listing in the working directory:" , os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#Drop features we are not going to use

train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)



#Look at the first 3 rows of our training data

train.head(3)
for df in [train,test]:

    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})

    

#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)

train['Age'] = train['Age'].fillna(0)

test['Age'] = test['Age'].fillna(0)



#Select feature column names and target variable we are going to use for training

features = ['Pclass','Age','Sex_binary']

target = 'Survived'



#Look at the first 3 rows (we have over 800 total rows) of our training data.; 

#This is input which our classifier will use as an input.

train[features].head(3)
from sklearn.tree import DecisionTreeClassifier



#Create classifier object with default hyperparameters

clf = DecisionTreeClassifier()  



#Fit our classifier using the training features and the training target values

clf.fit(train[features],train[target]) 
#Make predictions using the features from the test data set

predictions = clf.predict(test[features])

predictions
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})



#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'titanic_prediction_model_base.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)


