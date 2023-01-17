# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
train_data.head()
#selecting columns which are going to make sense to my analysis
train_data = train_data.loc[:,['Survived','Pclass','Sex','Age']]
train_data.head()
train_data.describe()
for index, row in train_data.iterrows():
    if row['Survived'] == 1 and row['Sex'] == 'female' and np.isnan(row['Age']):
        train_data.iloc[index, train_data.columns.get_loc('Age')] = 21
        
train_data.describe()
indexes_with_men_nan_age = []
for index, row in train_data.iterrows():
     if row['Survived'] == 1 and row['Sex'] == 'male' and np.isnan(row['Age']):
            indexes_with_men_nan_age.append(index)
            
#now only 50% of these can be assigned an age
for index in range(int(len(indexes_with_men_nan_age)/2)):
    train_data.iloc[indexes_with_men_nan_age[index], train_data.columns.get_loc('Age')] = 21
    
train_data.describe()
train_data = train_data.dropna()
train_data.describe()
#label encode the categorical data
data_for_analysis = pd.get_dummies(train_data, columns=["Sex","Pclass"])

features = data_for_analysis.iloc[:,1:].values
target = data_for_analysis.iloc[:,0].values
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(features, target)
#prediction
test_data = pd.read_csv("../input/test.csv")
test_data = test_data.dropna()
test_data = test_data.reset_index(drop=True)

test_features = pd.get_dummies(test_data.loc[:,['Pclass','Sex','Age']], columns=["Sex","Pclass"])
predicted_survival = lr.predict(test_features)

survival_column = pd.DataFrame({'Survived': predicted_survival})
#append the survival information to the test data set
final_data = pd.concat([test_data, survival_column], axis=1)
final_data.head()
secondtime_train_data = pd.read_csv("../input/train.csv")
#this time I wont be using age
secondtime_train_data = secondtime_train_data.loc[:,['Survived','Pclass','Sex']]
secondtime_train_data.describe()
secondtime_train_data = secondtime_train_data.dropna()
#label encode the categorical data
data2_for_analysis = pd.get_dummies(secondtime_train_data, columns=["Sex","Pclass"])

data2_features = data2_for_analysis.iloc[:,1:].values
data2_target = data2_for_analysis.iloc[:,0].values
#apply logistic regression
lr = LogisticRegression(solver='lbfgs')
lr.fit(data2_features, data2_target)

#prediction
data2_test_data = pd.read_csv("../input/test.csv")
data2_test_features = pd.get_dummies(data2_test_data.loc[:,['Pclass','Sex']], columns=["Sex","Pclass"])
data2_predicted_survival = lr.predict(data2_test_features)

survival_column = pd.DataFrame({'Survived': data2_predicted_survival})
#append the survival information to the test data set
data2_final_predicted_data = pd.concat([data2_test_data, survival_column], axis=1)
data2_final_predicted_data.head()
similar_prediction = []
different_prediction = []

for data2index, data2row in data2_final_predicted_data.iterrows():
    for data1index, data1row in final_data.iterrows():
        #check if passengers present in both indexes then check their survival
        if data2row['PassengerId'] == data1row['PassengerId'] and data2row['Survived'] == data1row['Survived']:
            similar_prediction.append(data1row['PassengerId'])
        elif data2row['PassengerId'] == data1row['PassengerId'] and data2row['Survived'] != data1row['Survived']:
            different_prediction.append(data1row['PassengerId'])

print("similar prediction:",len(similar_prediction))
print("different prediction:",len(different_prediction))
input_test_data = pd.read_csv("../input/test.csv")

#add an empty survival column
input_test_data["Survived"] = np.nan

# lets read the predictions from first predictor and add it to our submission_test_data
for index, row in final_data.iterrows():
    if pd.isna(input_test_data.loc[input_test_data['PassengerId']==row['PassengerId'], 'Survived']).bool():
        input_test_data.loc[input_test_data['PassengerId']==row['PassengerId'], 'Survived'] = row['Survived']
        
#now read the predictions form second predictor  and add it to our submission_test_data
for index, row in data2_final_predicted_data.iterrows():
    if pd.isna(input_test_data.loc[input_test_data['PassengerId']==row['PassengerId'], 'Survived']).bool():
        input_test_data.loc[input_test_data['PassengerId']==row['PassengerId'], 'Survived'] = row['Survived']
from sklearn.metrics import classification_report

y_predicted = input_test_data.loc[:,['PassengerId','Survived']].values
y_test = pd.read_csv("../input/gender_submission.csv").values

print(classification_report(y_predicted[:,1], y_test[:,1]))
submission = input_test_data.loc[:,['PassengerId','Survived']].round().astype(int)
submission.to_csv('titanic_csv_submission.csv', index=False)