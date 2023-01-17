# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=43)

print(os.listdir("../input"))
data=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
x=data.drop('SalePrice', axis='columns')
y=data.SalePrice
test.tail(10)
#model.fit(x,y)

# Any results you write to the current directory are saved as output.
#print(x.dtypes.sample(20));

x['label'] = 1

test['label'] = 2

# Concat
concat_df = pd.concat([x , test])

# Create your dummies
features_df = pd.get_dummies(concat_df)

# Split your data
x = features_df[features_df['label'] == 1]
test = features_df[features_df['label'] == 2]

# Drop your labels
x = x.drop('label', axis=1)
test = test.drop('label', axis=1)
#print(x.isnull().sum())
#cols_with_missing = [col for col in x.columns 
                           #      if x[col].isnull().any()]
#x = x.drop(cols_with_missing, axis=1)
#test = test.drop(cols_with_missing, axis=1)


from sklearn.preprocessing import Imputer
my_imputer = Imputer()
x = my_imputer.fit_transform(x)
test= my_imputer.fit_transform(test)
model.fit(x,y)
predictions= model.predict(test)
print(model.score(x,y))



my_submission = pd.DataFrame({'Id':1 , 'SalePrice': predictions})
my_submission.Id= my_submission.index+1461

# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)