import numpy as np
import pandas as pd

# get the data
data = pd.read_csv('../input/Interview.csv')
target = 'Observed Attendance'
predictors = [
    'Have you obtained the necessary permission to start at the required time',
    'Hope there will be no unscheduled meetings',
    'Can I Call you three hours before the interview and follow up on your attendance for the interview',
    'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much',
    'Have you taken a printout of your updated resume. Have you read the JD and understood the same',
    'Are you clear with the venue details and the landmark.',
    'Has the call letter been shared', 
    'Expected Attendance'
]
X = data[predictors]
y = data[target]
print("Unique values are:")
print(pd.unique(X.values.ravel()))
X = X.replace({
    'Yes' : 1,
    float('nan') : 0,
    'Na' : 0,
    'Uncertain' : 0,
    'No' : -1,
    'Havent Checked' : 0,
    'Not yet' : -1,
    'Need To Check' : 0,
    'No- I need to check' : 0,
    'No- will take it soon' : -1,
    'Not sure' : 0,
    'Yet to confirm' : -1,
    'No I have only thi number' : -1,
    'Yet to Check' : 0,
    'Not Sure' : 0,
    'yes' : 1,
    'NO' : -1,
    'No Dont' : -1,
    'cant Say' : 0,
    'no' : -1,
    'na' : 0,
    'Not Yet' : -1, 
    '11:00 AM' : 0,
    '10.30 Am' : 0
})
print("Unique values in X are now:")
print(pd.unique(X.values.ravel()))
print("unique values are: ", pd.unique(y.values.ravel()))
y = y.replace({
    'No' : 0,
    'Yes' : 1,
    'yes' : 1,
    'no' : 0,
    'yes ' : 1,
    'No ' : 0,
    'NO' : 0,
    'no ' : 0,
    float('nan') : 0
})
print("unique values are now: ", pd.unique(y.values.ravel()))
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

model = XGBRegressor(n_estimators=10000000,learning_rate=0.05)
# partition the data
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
# fit the model
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose = False)
# try using the model to predict attendance
predictions = model.predict(test_X)
print("mae before rounding the predictions: ", mean_absolute_error(test_y, predictions))
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
print("mae after rounding the predictions: ", mean_absolute_error(test_y, predictions))
total = test_y.size
num_correct = 0
for actual,prediction in zip(test_y,predictions):
     if actual == prediction:
            num_correct += 1
print("score: ", num_correct, "/", total, ", or ", 100*num_correct/total, " %", sep="")