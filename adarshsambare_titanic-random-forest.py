# Let's Get Started by importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing

# Visulation libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Reading the data as train and test
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
# Null values are empty data points
# checking the data for Null values
print(train.isnull().sum())
print('\n')
print(test.isnull().sum())
# we do have null values in some columns
# for combining both the data we need survived column in test data
test['Survived'] = 0
# need a new colume to differencate between the two
train['istest'] = 0
test['istest'] = 1
# Checking if survived was added or not
#train.describe()
test.describe()
# combining the data
dataset = pd.concat([train,test], join = 'inner')
dataset.describe()
dataset.isna().sum()
# Writing a function for changing null values as per median
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age
# Applying above impute on age columns
dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis = 1)
# Dropping the Cabin as too many null values
dataset.drop('Cabin', axis = 1, inplace = True)
# Filling up the null values with logical values
dataset['Fare'].fillna(14.454, inplace = True)
dataset['Embarked'].fillna("S", inplace = True)
dataset.describe()
dataset.isna().sum()
sex_dum = pd.get_dummies(dataset['Sex'],drop_first = True)
embark_dum = pd.get_dummies(dataset['Embarked'],drop_first = True)
pclass_dum = pd.get_dummies(dataset['Pclass'],drop_first = True)
# adding the dummies in the data frame
dataset = pd.concat([dataset,sex_dum,embark_dum,pclass_dum],axis = 1)
dataset.head()
# Checking the new train data
dataset.drop(['PassengerId','Pclass','Sex','Embarked','Name','Ticket'],axis = 1,inplace =True)
dataset.describe()
dataset.head()
train = dataset[dataset['istest'] == 0]
test = dataset[dataset['istest'] == 1]
train.drop('istest', axis = 1,inplace = True)
test.drop('istest', axis = 1,inplace = True)
train.isna().sum()
# splitting the train data for cross validations
x = train.drop('Survived',axis = 1)
y = train['Survived']
# Splitting using sklearn
from sklearn.model_selection import train_test_split
# Actual splitting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x,y)
# Predicting on the splited train data
prediction = dt.predict(x_test)
# As it is classification problem we need confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, prediction))
# Accuracy = 98%  
test.shape
Survived = dt.predict(test.drop('Survived',axis = 1)) 
type(Survived)
PassengerId = (list(range(892,1310)))
type(PassengerId)
kaggle_submission = pd.DataFrame(PassengerId, columns=['PassengerId'])
kaggle_submission['Survived'] = np.array(Survived)
kaggle_submission.describe()
# saving the dataframe 
kaggle_submission.to_csv('kaggle_submission_dt', index=False)
from sklearn.ensemble import RandomForestClassifier
# n_estimators are nothing but no of trees
# do not make to complex forest of decision tree
rfc = RandomForestClassifier(n_estimators=500)
# Fitting the Random Forest to trainng data
rfc.fit(x_train,y_train)
# Predicting on the training splitted test data by Random Forest 
pred_rfc = rfc.predict(x_test)
print(classification_report(y_test, pred_rfc))
# Accuracy = 82%