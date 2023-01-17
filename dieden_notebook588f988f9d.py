# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import IPython



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv')

y=data.pop('Survived')

data['Age'].fillna(data['Age'].mean(),inplace=True)

data.describe()
num_var=list(data.dtypes[data.dtypes != 'object'].index)

data[num_var].head()
model=RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

model.fit(data[num_var],y)
model.oob_score_
y_oob=model.oob_prediction_

print ('c-stat:' , roc_auc_score(y,y_oob))
def describe_categorical(data):

    

    from IPython.display import display ,HTML

    display(HTML(data[data.columns[data.dtypes=='object']].describe().to_html()))
describe_categorical(data)
data.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)