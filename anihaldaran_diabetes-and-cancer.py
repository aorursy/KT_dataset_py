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
main_file_path = '../input/diabetes/diabetes.csv' # this is the path to the diabetic table that you will use
data = pd.read_csv(main_file_path)
data.columns #outputs the columns of the table diabetes.csv
main_file_path1 = '../input/diabetes-dataset/diabetic_data.csv' # this is the path to the second diabetic table that you will use
data1 = pd.read_csv(main_file_path1)
data1.columns #outputs the columns of the table diabetic_data.csv
main_file_path2 = '../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv' # this is the path to the second diabetic table that you will use
data2 = pd.read_csv(main_file_path2)
data2.columns #outputs the columns of the table diabetic_data.csv
from sklearn.tree import DecisionTreeRegressor # used to make predictions from certain data
#factors that will predict Outcome for diabetes
desired_factors = ['Age']

#set my model to DecisionTree
model = DecisionTreeRegressor()

#set prediction data to factors that will predict, and set target to Outcome
train_data = data[desired_factors]
test_data = data2[desired_factors]
target = data.Outcome

#fitting model with prediction data and telling it my target for the cancer data 
model.fit(train_data, target)

model.predict(test_data)
import seaborn as sns #for seaborn plotting
sns.countplot(data2['Age'].head(30)) #seaborn countplot for the first 5 pieces of SalePrices data (shows the probability of each section of prices occuring)
submission = pd.DataFrame({'Age': data2.Age ,'cancer outcome':data2.Dx,'diabetic outcome': model.predict(test_data)})

submission.to_csv('MySubmission.csv', index=False)