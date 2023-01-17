# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_full = pd.read_csv('../input/dc-residential-properties/DC_Properties.csv')

data_full

data_full.columns

data_full['PRICE']
#null_values=data_full['PRICE'].isnull()

unknown_data= data_full.loc[data_full['PRICE'].isnull()==True]

unknown_data



#Just for code

#null_columns=data_full.columns[data_full.isnull().any()]

#print(data_full[data_full.isnull().any(axis=1)][null_columns])

data_withoutna = data_full[np.isfinite(data_full['PRICE'])]

data_withoutna
#y= data_withoutna.PRICE

#X= data_withoutna.drop('PRICE',axis=1) 

#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] another way of using only int features 



newdf = data_withoutna.select_dtypes(exclude= object)

newdf

y= newdf.PRICE

X= newdf.drop('PRICE',axis=1) 



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=.2,random_state=42)

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_test = my_imputer.transform(X_test)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score





model1 = RandomForestRegressor()

model1.fit(imputed_X_train, y_train)

preds = model1.predict(imputed_X_test)

MAE = mean_absolute_error(y_test, preds)



print("Mean Absolute Error from Imputation:")

print('MAE : ', MAE)



r_square= r2_score(y_test,preds) 

print('R square: ',r_square)
from sklearn.linear_model import LinearRegression as lm 

from sklearn.metrics import r2_score





model = lm().fit(imputed_X_train,y_train)

model.fit(imputed_X_train, y_train)

preds = model.predict(imputed_X_test)



MAE=mean_absolute_error(y_test, preds)





print("Mean Absolute Error from Imputation:")

print('MAE :', MAE)



r_square= r2_score(y_test,preds) 

print('R square : ',r_square)



#import matplotlib.pyplot as plt

#plt.scatter(y_test,predictions)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

cor = X.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()