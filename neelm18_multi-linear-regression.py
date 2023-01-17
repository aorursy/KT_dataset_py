# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)

print ("Skew is:", df_train.SalePrice.skew())

plt.hist(df_train.SalePrice, color='blue')

plt.show()



target = np.log(df_train.SalePrice)

print ("Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()



y = target #df_train['SalePrice']

y
nulls = pd.DataFrame(df_train.isnull().sum().sort_values(ascending=False)[:35])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls

df_train = df_train.dropna(axis='columns')



nulls = pd.DataFrame(df_train.isnull().sum().sort_values(ascending=False)[:35])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
dummies_Street = pd.get_dummies(df_train.Street)

merged_df_train = pd.concat([df_train,dummies_Street],axis='columns')

merged_df_train = merged_df_train.drop('Street',axis='columns')



merged_df_train.dtypes



obj_df = merged_df_train.select_dtypes(include=['object']).copy()

obj_df.head()

obj_df[obj_df.isnull().any(axis=1)]
dummies_obj_df = pd.get_dummies(obj_df, columns=obj_df.columns)

dummies_obj_df
finalMerged_df_train = pd.concat([merged_df_train,dummies_obj_df],axis="columns")

finalMerged_df_train = finalMerged_df_train.drop(obj_df.columns,axis='columns')

finalMerged_df_train = finalMerged_df_train.dropna(axis='columns')

finalMerged_df_train = finalMerged_df_train.drop(columns = ['Id'])

finalMerged_df_train
#### Preparing the Test Data ####



df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



### Dropping the columns with null values ###

df_test = df_test.dropna(axis='columns')



### Creating dummy categories ###

obj_df = df_test.select_dtypes(include=['object']).copy()

dummies_obj_df = pd.get_dummies(obj_df, columns=obj_df.columns)

dummies_obj_df



finalMerged_df_test = pd.concat([df_test,dummies_obj_df],axis="columns")

finalMerged_df_test = finalMerged_df_test.drop(obj_df.columns,axis='columns')

finalMerged_df_test = finalMerged_df_test.dropna(axis='columns')

finalMerged_df_test



### Separating the test ID ###

submission = pd.DataFrame()

submission['Id'] = df_test.Id

finalMerged_df_test=finalMerged_df_test.drop(columns = ['Id'])



### select only the columns in the test data ###

testCols = finalMerged_df_test.columns

finalMerged_df_test = finalMerged_df_test[finalMerged_df_train.columns & testCols]

finalMerged_df_test

### Making the predictions ###

#predictions = model.predict(finalMerged_df_test)

#final_predictions = np.exp(predictions)



#print ("Final predictions are: \n", final_predictions[:5])

                                
from sklearn.model_selection import train_test_split



### select only the columns in the test data ###



model = LinearRegression()

X = finalMerged_df_train[finalMerged_df_train.columns & testCols]



X_train, X_test, y_train, y_test = train_test_split(

                          X, y, random_state=42, test_size=.33)



model.fit(X_train,y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))
### Making the predictions ###

predictions = model.predict(finalMerged_df_test)

final_predictions = np.exp(predictions)



print ("Final predictions are: \n", final_predictions[:5])

                                
submission['SalePrice'] = final_predictions

submission.head()
submission.to_csv('submission1.csv', index=False)