import xgboost as xgb

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
dataset = pd.read_csv('../input/adult-census-income/adult.csv') #read file with data set

# Let's see first 10 lines of our table just to know whith what we will work

dataset.head(10)
# We can see that there are 32561 instances(lines) and 15 attributes(columns) in the data set.

dataset.shape
# This is all columns from data set

dataset.columns
# We can see that our table has missing values "?"(for example line 0, line 2)

# We have to correct it, because if not our program will think that person 0 and person 2 

# have the same workclass which call "?" and have the same occupation also "?",

# I guess it is not true so I just change all "?" to NaN value

dataset[dataset == "?"] = np.nan

dataset.head(3)
# Ok now Thre is NaN 
# In this part of code we change NaN with the most frequent value

for col in dataset.columns:

    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

dataset.head(10)
dataset.dtypes
# Above we see all types of columns I think "age" don't need such big number like int64,

# so I chage to smaller number int8

dataset.astype({'age':'int8', 'education.num':'int8','hours.per.week':'int8'}).dtypes

# Classify columns to numerical group (comparable values) and categorical group (non comparable group)

CATEGORICAL = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

NUMERICAL = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
# We do this becouse it is not comfortable to work with string values and we will hash(encrypt) it to numvers,

# the same values correspond to the same number, the difference values correspond to the different numbers

# we use function fit_transform to do this
for i in CATEGORICAL:

    dataset[i] = LabelEncoder().fit_transform(dataset[i])
# Lts's see what happen, there are only numbers now

dataset.sample(10)
# We do the same with 'income' but manually

dataset['income'] = dataset['income'].apply(lambda x: 0 if(x == '<=50K') else 1)

# look 'income' was changed

dataset.sample(5)
# Divide the table into input and output,  

y = dataset.pop('income') #output

X = dataset.copy() #input

# it is seems like function in math y(x), so we have x and must find y 
# Split data into separate training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, train_size=0.2)
X_train.shape, X_test.shape
# METHOD 1 RANDOM FOREST

# I decided to do this task with random forest, so I create the object "random_forest"

model_random_forest = RandomForestClassifier()

model_random_forest.fit(X_train, y_train) # Build a forest of trees from the training set (X_train, y_train).
y_pred_random_forest = model_random_forest.predict(X_test) # Predict 
y_pred_prob_random_forest = model_random_forest.predict_proba(X_test) # Predict proba
# Use Accuracy Score

metrics.accuracy_score(y_test, y_pred_random_forest)
metrics.roc_auc_score(y_test, y_pred_prob_random_forest[:, 1])
# We also can use Confusion Matrix 

conf_matrix = metrics.confusion_matrix(y_test, y_pred_random_forest)

conf_matrix
# So we have confusion matrix which look like this:

#            Predicted 0   Predicted 1

#  Actual 0   18448           1334

#  Actual 1    2439           3828



tn = conf_matrix[0,0] # true negative

fp = conf_matrix[0,1] # false positive

fn = conf_matrix[1,0] # false negative

tp = conf_matrix[1,1] # true positive
# how often is the classifier correct

(tn+tp)/(tn+fp+fn+tp)
# Aha! Have the same result. But the plus of confusion matrix is that we can calculate variety of metrix:

# For exapmle:

# When the actual value is positive, how often is the prediction correct?

print(tp/(tp+fn))

# When the actual value is negative, how often is the prediction correct?

print(tn/(tn+fp))
# We have the same value with accuracy_score and Confussion_matrix

# But let's see how many 1 and 0 in test set (y_test)

y_test.value_counts()
# So we have about 80% of 0, this is high imbalance 

# This means that even if we just print only 0 we will gave about 80% of correct answers

# Maybe we should use other method ???
# Let's try roc_auc_score

# AUC is useful even when there is high imbalance (unlike classification accuracy)

metrics.roc_auc_score(y_test, y_pred_random_forest)
# Now we see other result
# METHOD 2 XGB

# Do the same as above but using another classifier algorithm

model_xgb = xgb.XGBClassifier()

model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)

y_pred_prob_xgb = model_xgb.predict_proba(X_test)

metrics.accuracy_score(y_test, y_pred_xgb)
metrics.roc_auc_score(y_test, y_pred_prob_xgb[:, 1])
# Conclusions:

# When we use random forest classifier we have accuracy about 90%

# When we use XGBCLassifier we have accuracy about 92%

# So I think better to use XGBCLassifier in this task