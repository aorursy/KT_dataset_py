# Data analysis modules (super necesarry)
import pandas as pd
import numpy as np

# Visualization libraries
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

# Machine learning libraries
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, Imputer
import os

# find current directory
os.path.realpath('.')

# change directory
os.chdir('/Users/caitlinweigel/Documents/iX_Data_Science/15')

# Importning the datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
train[['Pclass', 'Survived']].head()
pd.unique(train['Pclass'])
# summary of training dataset
train.sum()
# describe the dataset
train.describe()
# Finding out what type of thing each column is
train.info()

# Or
train.dtypes
pd.isna(train).sum()
# Gives how many total missing nan values there are in each column.
# Using LabelEncoder to change the Sex column to integers; male:1, female:0
lb = LabelEncoder()
train['Sex'] = lb.fit_transform(train['Sex']) 
# Setting target to be the survived column
target = train['Survived']
# Dropping columns that I don't think will significantly affect survival. 
# (Also dropping Cabin column b/c I don't want to deal with over 600 Na's.)
# Also removing the target column.
train_dropped = train.drop(columns = ['Ticket', 'Fare', 'Name', 'Cabin', 'PassengerId', 'Survived', 'Embarked'])
# Imputer is used to fill all the occurances of NaN with mean of that column.
im = Imputer()
train_dropped = im.fit_transform(train_dropped)
model = DecisionTreeClassifier()
fit = model.fit(train_dropped, target)
#Test data is cleaned in the same way as the training data
lb3 = LabelEncoder()
test['Sex'] = lb3.fit_transform(test['Sex']) #male:1, female:0

test.head()
test_dropped = test.drop(['Ticket', 'Fare', 'Name', 'Cabin', 'PassengerId', 'Embarked'], axis = 1)
test_dropped = im.fit_transform(test_dropped)
predictions = fit.predict(test_dropped)

predictions
test_data = pd.read_csv("test.csv").values
result = np.c_[test_data[:,0].astype(int), predictions.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('res1.csv', index=False)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(fit, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())