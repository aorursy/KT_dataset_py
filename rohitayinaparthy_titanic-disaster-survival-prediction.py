### Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##from plotline import *
# Read the data set

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

train_data.tail()
### Checking the dimension of data
train_data.shape
train_data.dtypes
## Checking the passenger class counts
train_data.Pclass.value_counts()

## chekking the count of males and females
train_data.Sex.value_counts()
## chekking the count of males and females
train_data.Survived.value_counts()
train_data.columns
train_data.Age.describe()


## Extracting surnames
train_data["Title"] = train_data.Name.str.split(",").str.get(1).str.split(".").str.get(0).str.strip()
## Checking all the possible values in surname field
train_data["Title"].value_counts()
## Titles with low cell values are clubbed and assigned to "Rare title"
rare_title =  ['Dr',
 'Rev',
 'Col',
 'Major',
 'Capt',
 'Jonkheer',
 'Sir',
 'Lady',
 'Don',
 'the Countess']
train_data["Title"][train_data["Title"].str.contains("Mlle|Ms")] = "Miss" 
train_data["Title"][train_data["Title"].str.contains("Mme")] = "Mrs" 
train_data["Title"][train_data["Title"].str.contains('Dr|Rev|Col|Major|Capt|Jonkheer|Sir|Lady|Don|the Countess')] = "Rare Title" 
train_data.Title.value_counts()
### Identifying titles to test data well

## Extracting surnames
test_data["Title"] = test_data.Name.str.split(",").str.get(1).str.split(".").str.get(0).str.strip()
## Checking all the possible values in surname field
test_data["Title"].value_counts()
## Titles with low cell values are clubbed and assigned to "Rare title"
test_data["Title"].value_counts().index
rare_title_test =  ['Col', 'Rev', 'Dona', 'Dr']
test_data["Title"][test_data["Title"].str.contains("Ms")] = "Miss" 
test_data["Title"][test_data["Title"].str.contains('Col|Rev|Dona|Dr')] = "Rare Title" 
test_data.Title.value_counts()
train_data['Family_Size']=train_data['SibSp']+train_data['Parch']+1
test_data['Family_Size']=test_data['SibSp']+test_data['Parch']+1
train_data.Family_Size.describe()
## Descritize family variable
train_data["FsizeD"] = "NaN"
train_data["FsizeD"][train_data["Family_Size"] == 1]  = "singleton"

mask = (train_data["Family_Size"] < 5) & (train_data["Family_Size"] > 1)
train_data.loc[mask,"FsizeD"] = "small"
train_data.loc[(train_data.Family_Size > 4),"FsizeD"] = "large"

train_data.FsizeD.value_counts()

#### Similar operations for test data as well.
test_data.shape
test_data['Family_Size']=test_data['SibSp']+test_data['Parch']+1
test_data.Family_Size.describe()
## Descritize family variable
test_data["FsizeD"] = "NaN"
test_data["FsizeD"][test_data["Family_Size"] == 1]  = "singleton"

mask = (test_data["Family_Size"] < 5) & (test_data["Family_Size"] > 1)
test_data.loc[mask,"FsizeD"] = "small"
test_data.loc[(test_data.Family_Size > 4),"FsizeD"] = "large"

test_data.FsizeD.value_counts()

### Removing cols from train data
train_data_1 = train_data.drop(axis=1,columns=["PassengerId","Name","Ticket","Cabin","Family_Size"])
train_data_1.head()
### Removing cols from train data
test_data_1 = test_data.drop(axis=1,columns=["PassengerId","Name","Ticket","Cabin","Family_Size"])
test_data_1.head()


## Storing target variable in y
y = train_data_1["Survived"]
## Storing predictors in X
X = train_data_1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked','Title','FsizeD']]
X.head(1)

### Creating dummy variable in training data set
X = pd.get_dummies(data = X,drop_first=True)
X.head(1)
### Creating dummy variable in test data set
test_data_1 = pd.get_dummies(data = test_data_1,drop_first=True)
test_data_1.head(1)

### Split Training data into train and validation data
from sklearn.model_selection import train_test_split
X_train,X_validation, y_train,y_validation = train_test_split(X,y,test_size = 0.2,random_state = 3454)


### Check all the null values we have in all formats of data
X_train.isnull().any()
## So Only Age value has null values in X_Train so lets check X_validation as well
X_validation.isnull().any()
### So both X_train and X_validation has Age null values

test_data_1.isnull().any()
## Check the null fare values. check its corresponding categorical values and impute accordingly

test_data_1[test_data_1.Fare.isnull()]
## The above person belongs to class 3 and lets take median of class 3 and embarkment = S people and impute median value
fare_null_filter = ((test_data_1.Pclass == 3) & (test_data_1.Embarked_S == 1))
fare_null_df = test_data_1.loc[fare_null_filter,]
fare_null_df["Fare"].median()
test_data_1.loc[test_data_1.Fare.isnull(),"Fare"] = fare_null_df["Fare"].median()
### Check again the null values of test data
test_data_1.isnull().any()  ### Now we have null values in only in Age

### Imputing mean age to all the null values
####  from fancyimpute import MICE   - Donot find this package but can consider install later

from sklearn.preprocessing import Imputer
imputer = Imputer(axis=0,missing_values="NaN",strategy="mean")
X_train[:] = imputer.fit_transform(X_train)
X_validation[:] = imputer.transform(X_validation)
### Transform the mean value of training set data to test data set
test_data_1[:] = imputer.transform(test_data_1)

X_train.isnull().any()
X_validation.isnull().any()
test_data_1.isnull().any()

# Now that we know everyone's age, we can create age dependent variables child and mother
# Train data
X_train["Child_col"] = "NaN"
X_train.loc[X_train.Age < 18,"Child_col"] = "Child"
X_train.loc[X_train.Age >= 18,"Child_col"] = "Adult"


# Validation data
X_validation["Child_col"] = "NaN"
X_validation.loc[X_validation.Age < 18,"Child_col"] = "Child"
X_validation.loc[X_validation.Age >= 18,"Child_col"] = "Adult"


### Test data
test_data_1["Child_col"] = "NaN"
test_data_1.loc[test_data_1.Age < 18,"Child_col"] = "Child"
test_data_1.loc[test_data_1.Age >= 18,"Child_col"] = "Adult"




## Train data
X_train["Mother_col"] = "Not Mother"
X_train.loc[((X_train.Parch > 0) & (X_train.Title_Miss == 0) & (X_train.Sex_male == 0) & (X_train.Child_col == "Adult")),"Mother_col"] = "Mother"

## Validation data
X_validation["Mother_col"] = "Not Mother"
X_validation.loc[((X_validation.Parch > 0) & (X_validation.Title_Miss == 0) & (X_validation.Sex_male == 0) & (X_validation.Child_col == "Adult")),"Mother_col"] = "Mother"

## Test data
test_data_1["Mother_col"] = "Not Mother"
test_data_1.loc[((test_data_1.Parch > 0) & (test_data_1.Title_Miss == 0) & (test_data_1.Sex_male == 0) & (test_data_1.Child_col == "Adult")),"Mother_col"] = "Mother"



### Create dummy variables for the newly created cols in all the 3 data sets- Train, Validation and Test
X_train = pd.get_dummies(X_train,drop_first=True)
X_validation = pd.get_dummies(X_validation,drop_first=True)
test_data_1 = pd.get_dummies(test_data_1,drop_first=True)


### Check the columns of all the datasets are in same order---- All the cols are in the sames order as we can check below
X_train.columns

X_validation.columns
test_data_1.columns


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
### Adding contanst(intercept) to train data 
#X_train = add_constant(X_train)
### Lets try to create new columns and calculate vif

pd.Series([variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])],index = X_train.columns)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error
rf_Classifier = RandomForestClassifier(random_state = 1)
rf_Classifier.fit(X_train,y_train)
y_pred = rf_Classifier.predict(X_validation)
mean_absolute_error(y_validation,y_pred)
validation_check_df = pd.DataFrame({"val":y_validation,"pred":y_pred})
from sklearn.metrics import confusion_matrix

confusion_matrix(y_validation,y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_validation,y_pred)
print("The Accuracy of model using Random Forest is : " + str(accuracy_score(y_validation,y_pred)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
my_pipeline = make_pipeline(RandomForestClassifier())
scores = cross_val_score(my_pipeline,X_train,y_train,scoring='accuracy')
scores.mean()
print("The Accuracy of model using Random Forest and Cross validation with pipelines is : " + str(scores.mean()))

from xgboost import XGBClassifier
model_3 = XGBClassifier(n_estimators=2000,learning_rate=0.05)
model_3.fit(X_train,y_train,verbose= False,early_stopping_rounds=10,eval_set=[(X_validation,y_validation)])
y_pred_3 = model_3.predict(X_validation)
mean_absolute_error(y_validation,y_pred_3)
confusion_matrix(y_validation,y_pred_3)
accuracy_score(y_validation,y_pred_3)
print("The Accuracy of model using XGBoost is : " + str(accuracy_score(y_validation,y_pred_3)))

my_pipeline_2 = make_pipeline(XGBClassifier(n_estimators=2000,learning_rate=0.05))

scores = cross_val_score(my_pipeline_2,X_train,y_train,scoring='accuracy')

print("The Accuracy of model using XGBoost and Cross validation with pipelines is : " + str(scores.mean()))


## MAKE PREDICTIONS
y_test_pred = model_3.predict(test_data_1)
## Create output file with only passenger id's and predicted survival values
output = pd.DataFrame({"PassengerId":test_data.PassengerId, "Survived":y_test_pred})
output.to_csv("Titanic_Survival_submission.csv",index= False)

