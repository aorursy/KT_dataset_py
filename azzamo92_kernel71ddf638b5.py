
#Credit for Nadin Tamer post as a used it as a general guide for my 1st model 
# --> https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner


# Import necessary libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Loading data Set 
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')

# Data preprocessing 
y = train_set.Survived  
predics = ['Sex','Age','Parch','SibSp','Pclass']
X = train_set[predics]
ids = test_set['PassengerId']
test_set = test_set[predics]

# Cleaning the data 
X['Age'] = X['Age'].fillna(int(X['Age'].median()))
test_set['Age'] = test_set['Age'].fillna(int(test_set['Age'].median()))
sex_mapping = {"male":1, "female":0}
X['Sex'] = X['Sex'].map(sex_mapping) 
test_set['Sex'] = test_set['Sex'].map(sex_mapping)

# Split training set to compare models and confirm accuracy 
from sklearn.model_selection import train_test_split 
x_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.22, random_state=0) 

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
accuracy = round(accuracy_score(y_pred, y_val) * 100, 2)

#Submission File 
results = gbk.predict(test_set)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived?': results })
output.to_csv(submission.csv,index=False)

