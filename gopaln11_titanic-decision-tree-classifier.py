#import all the libraries required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn import tree
# conda scikit-learn version is 0.17.1 needs to be updated to 0.8*
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV

%matplotlib inline
#import train and test datasets from kaggle site
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#display few records
train.head()
train.info()
#from below Age, Cabin and embarked fields have some missing values 
test.head()
#notice Survived column is missing and needs to be predicted using our model
test.info()
#from below Age, Fare and Cabin fields have some missing values
#let us try to find the features that are good indicators for model
train.groupby(['Pclass']).Fare.mean()
#START exploring Pclass feature
train.groupby(['Pclass','Survived']).Survived.count()
#Evident from the below data that passengers in class 1 had more chance of survival, class 2 passeneger had 
#average chance of survival, whereas class 3 passengers unfortunately had poor chance
#Analyze data in Sex column
train.groupby(['Sex','Survived']).Survived.count()
#further more lets check for any relation with Pclass
train.groupby(['Pclass','Sex','Survived']).Survived.count()
#Evident from below Pclass and Sex are good indicators to predict survival
# We now know Pclass and Sex are good indicators in labelling target data, also we are not quantifying the 
#influence of these indicators on target label, thats exactly what the Model is suppose to do
#ignoring Cabin because high % of missing values 
#explore train data
train.Age.describe()
# identify number of records with null values in Age field
train.Age.isnull().sum()
#Explore Age data
train.Age.unique()
len(train.Age.unique())
#Fill null values with median of Age (mean can also be used)
#Make sure to replicate these changes in test dataframe as well
train['Age'] = train.Age.fillna(train.Age.median())
test['Age'] = test.Age.fillna(test.Age.median())
len(train.Age.unique())
#get details of Embarked field
train.Embarked.describe()
#as only 3 values get details of each
train['Embarked'].value_counts(dropna=False)
#Explore Embarked feature
train.groupby(['Embarked','Survived']).Survived.count()
#replace null values with mode or most common value i.e 'S'
train['Embarked'] = train.Embarked.fillna('S')
train['Embarked'].value_counts(dropna=False)
#Create a new column called Has_Cabin and populate False for missing values in Cabin column
#Make sure to replicate these changes in test dataframe as well
train['Has_Cabin'] = ~train.Cabin.isnull()
test['Has_Cabin'] = ~test.Cabin.isnull()
train.Has_Cabin.head()
train.groupby(['Has_Cabin','Survived']).Survived.size()
#Create a new column called Fam_Count adding sibling and parent columns count
train['Fam_Count'] = train.SibSp + train.Parch
test['Fam_Count'] = test.SibSp + test.Parch
train.groupby(['Fam_Count','Survived']).Survived.size()
train.info()
#from below our data has no missing values except Cabin which will be exempt for training our MODEL
test['Fare'] = test.fillna(test.Fare.median())
test.info()
#from below our data has no missing values except Cabin which will be exempt for training our MODEL
#from data exploration we understand Pclass, Sex, Age, Fare, Embarked, Has_Cabin and Fam_Count are good indicator
#for our MODEL also most of these models process only numerical data we need to map strings/char in Sex, Embarked field
#to integers
pd.get_dummies(train,columns = ['Embarked','Sex'], drop_first=True)
#from below for Sex column it supress female values by creating a new column Sex_male with integers
train = pd.get_dummies(train,columns = ['Embarked','Sex'], drop_first=True)
test = pd.get_dummies(test,columns = ['Embarked','Sex'], drop_first=True)
train.head()
test.head()
#select only columns identified for MODEL selection 
#notice Survived is removed as it will be supplied seperately to model
#'values' method is to convert DF to np array
X_train = train[['Pclass','Age', 'Fare','Sex_male','Embarked_Q','Embarked_S','Has_Cabin','Fam_Count']].values
#X_train = train.drop(labels = (['PassengerId','Survived','Name','Ticket','Cabin', 'Embarked']),axis = 1)
y_train = train[['Survived']].values
X_test = test[['Pclass','Age','Fare','Sex_male','Embarked_Q','Embarked_S','Has_Cabin','Fam_Count']].values
#split your TRAIN data for efficient max_depth value calculation
X1_train,X1_test,y1_train,y1_test = train_test_split(X_train,y_train,test_size=0.30, random_state=42)
#Use Decision tree classification MODEL to identify efficient max_depth parameter
dep = np.arange(1,9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

for i,d in enumerate(dep):
    clsf = tree.DecisionTreeClassifier(max_depth=d)
    clsf.fit(X1_train, y1_train)
    train_accuracy[i] = clsf.score(X1_train, y1_train)
    #y_test = clsf.predict(X_test)
    test_accuracy[i] = clsf.score(X1_test,y1_test)

plt.title('clsf accuracy train vs test by max_depth')
plt.plot(dep, train_accuracy, label = 'Train_accuracy')
plt.plot(dep, test_accuracy, label = 'Test_accuracy')
plt.legend(loc=2)
plt.xlabel('depth')
plt.ylabel('Accuracy')
plt.show()
clsf1 = tree.DecisionTreeClassifier(max_depth=3)
clsf1.fit(X_train,y_train)
y_pred = clsf1.predict(X_test)
print(y_pred)
test['Survived'] = y_pred
test.head()
test[['PassengerId','Survived']].to_csv("../input/Titanic_V2.csv",index=False)
