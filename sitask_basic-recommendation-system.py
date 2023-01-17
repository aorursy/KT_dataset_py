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
import numpy as np
import pandas as pd

from matplotlib import pyplot as plot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

problemsfile = '../input/problem_data.csv'
userfile = '../input/user_data.csv'
trainfile = '../input/train_submissions.csv'
testfile = '../input/test_submissions_NeDLEvX.csv'

problems_df = pd.read_csv(problemsfile)
users_df = pd.read_csv(userfile)
train_df = pd.read_csv( trainfile )
test_df = pd.read_csv( testfile )

print('Training Data Count: ', len(train_df))
print('Null values in training set:\n', train_df.isnull().sum())

print('Test Data Count: ', len(test_df))
print('Null values in test set:\n', test_df.isnull().sum())

print('Number of problems: ', len(problems_df))
print('Null values in problems:\n', problems_df.isnull().sum())

print('Number of users: ', len(users_df))
print('Null values in user data:\n', users_df.isnull().sum())

print(problems_df.head())
print(users_df.head())
print(train_df.head())
print(test_df.head())

#merge training data with the problems df and then with the users df
df = pd.merge(train_df, problems_df, on='problem_id', how='left')
X = pd.merge(df, users_df, on='user_id', how='left')

print('Training data: ', len(X))
users = X['user_id'].unique()
print('Users: ', len(users)) ## returns 3529 unique users
problems = X['problem_id'].unique()
print('Problems: ', len(problems)) ## returns 5776 unique problems

print('Test data: ', len(test_df))
users = test_df['user_id'].unique()
print('Users: ', len(users)) ## returns 3501 unique users
problems = test_df['problem_id'].unique()
print('Problems: ', len(problems)) ## returns 4716 unique problems


X.head()
#convert the difficulty level to a numeric value
factor = pd.factorize(X['level_type'])
X['diff_level'] = factor[0]

#convert the rank to a numeric value [beginner, intermediate, advanced, expert]
factor = pd.factorize(X['rank'])
X['rank'] = factor[0]
factor = pd.factorize(users_df['rank'])
users_df['rank'] = factor[0]

#convert user_id and problem_id to numeric by removing the user_ and prob_
X['user_id'] = X['user_id'].str.replace('user_', '')
X['problem_id'] = X['problem_id'].str.replace('prob_', '')
#convert them to ints as they are strings
X['user_id'] = X['user_id'].astype('int64', copy=False)
X['problem_id'] = X['problem_id'].astype('int64', copy=False)
X.describe()


#Build the y vector from the attempts_range
y = pd.get_dummies(X['attempts_range'])

#saving for later use
y_xgb = X['attempts_range']

#drop all non-numeric columns and then some more to improve prediction
#attempts_range is the y vector, so dropping that also
X = X.drop(['points','tags','level_type','attempts_range','country','last_online_time_seconds','registration_time_seconds','follower_count'], axis=1)

X.head()
fig = problems_df['level_type'].value_counts().sort_index().plot(kind="bar", stacked=True, title="Problem Difficulty Distribution")
fig.set_ylabel("Number of questions")
plot.show()
ax = sns.pairplot(users_df[["submission_count", "problem_solved", "rating", "rank"]])
var_Corr = X[["submission_count", "problem_solved", "rating", "rank"]].corr()
fig2 = plot.figure()
fig2 = sns.heatmap(var_Corr, xticklabels=var_Corr.columns, yticklabels=var_Corr.columns, annot=True)
plot.show()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=1)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 84)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

ypred_df = pd.DataFrame(y_pred)
ypred_df.index = y_test.index
ypred_df.columns=['1','2','3','4','5','6']

predicted_attempt = ypred_df.idxmax(axis=1)
given_attempt = y_test.idxmax(axis=1)

# Making the Confusion Matrix
print(pd.crosstab(given_attempt, predicted_attempt, rownames=['Actual Attempt Range'], colnames=['Predicted Attempt Range']))

accuracy = accuracy_score(y_test, ypred_df)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

rmse = np.sqrt(mean_squared_error(y_test, ypred_df))
print("RMSE: %f" % (rmse))

# attempting xgboost algorithm

X_train, X_test, y_train, y_test = train_test_split( X, y_xgb, test_size=0.25, random_state=1)

eval_set = [(X_train,y_train),(X_test,y_test)]
xgb_class = xgb.XGBClassifier(max_depth='10',n_estimators=100, gamma=0, objective='multi:softmax')
xgb_class.fit(X_train,y_train,eval_set=eval_set,verbose=1,eval_metric=['mlogloss'])
print(xgb_class)

preds = xgb_class.predict(X_test)
# Making the Confusion Matrix
print(pd.crosstab(given_attempt, preds, rownames=['Actual Attempt Range'], colnames=['Predicted Attempt Range']))

y1 = pd.get_dummies(y_test)
y2 = pd.get_dummies(preds)

accuracy = accuracy_score(y1, y2)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

rmse = np.sqrt(mean_squared_error(y1, y2))
print("RMSE: %f" % (rmse))