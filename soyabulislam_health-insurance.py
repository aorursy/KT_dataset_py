# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train= pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

train
test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

test
sample=pd.read_csv("../input/health-insurance-cross-sell-prediction/sample_submission.csv")

sample
train.isnull().sum()
train.info()
train_data= pd.get_dummies(train)

train_data
test_data=pd.get_dummies(test)

test_data
train.hist(bins=50, figsize=(20,15))

plt.show()
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import  Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier

sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
for i in train.columns:

    plt.figure()

    plt.hist(train[i])
train_data.dropna(axis=0, subset=['Response'], inplace=True)

val = train_data["Response"]

train_data.drop(['Response'], axis=1, inplace=True)
x_train, x_test, y_train, y_test= train_test_split(train_data, val, test_size=0.25, random_state=42)

x_train
Grad1=GradientBoostingRegressor(learning_rate=0.5,criterion='mse',n_estimators=200,

                               max_depth=6, random_state=42, verbose=1)

model1= Grad1.fit(x_train, y_train)

prediction1=model1.predict(x_test)

model1.score(x_test, y_test)

Grad2=GradientBoostingClassifier(learning_rate=0.3,criterion='mse',n_estimators=200,

                               max_depth=6, random_state=10, verbose=1)

model2= Grad2.fit(x_train, y_train)

prediction2=model2.predict(x_test)

model2.score(x_test, y_test)
reg= RandomForestRegressor(n_estimators=500, max_depth=6,

                          random_state=42, verbose=1)

model3= reg.fit(x_train, y_train)

prediction3=model3.predict(x_test)

model3.score(x_test, y_test)
fclass= RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=6,

                          random_state=42, verbose=1)

model4= fclass.fit(x_train, y_train)

prediction4=model4.predict(x_test)

model4.score(x_test, y_test)




from sklearn.linear_model import  Ridge, Lasso

ridge = Ridge(alpha = 1.5)  # sets alpha to a default value as baseline  

model5=ridge.fit(x_train, y_train)

prediction5=model5.predict(x_test)

model5.score(x_test, y_test)





lasso= Lasso(alpha = .001)  # sets alpha to a default value as baseline  

model6= lasso.fit(x_train, y_train)

prediction6=model6.predict(x_test)

model6.score(x_test, y_test)
final_prediction= model2.predict(test_data)

final_prediction


print ('MSE is: \n', mean_squared_error(y_test, prediction1))

print ('MAE is: \n', mean_absolute_error(y_test, prediction1))
output = pd.DataFrame({'id': test_data['id'],

                       'Response': final_prediction})

output

output.to_csv('submission.csv', index=False)
sub= pd.read_csv('submission.csv')

sub