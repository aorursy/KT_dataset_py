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
import seaborn as sns
med_df=pd.read_csv("/kaggle/input/insurance/insurance.csv")

med_df.head()
sns.countplot(med_df['region'])
dummies = pd.get_dummies(med_df[['sex','smoker','region']],drop_first=True) #changing categorical (alphabetical) data into binary 

med_df.drop(['sex','smoker','region'],axis=1,inplace=True)

med_df = pd.concat([med_df,dummies],axis=1)

med_df.head(2)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

#from sklearn.linear_model import LinearRegression

#from sklearn.metrics import r2_score
def data(X,y,new_data_set): #new_data_set to actually see numbers by comparing 

    #poly = PolynomialFeatures() # default degree is 2

    #X = poly.fit_transform(X) # transforming features into PolynomialFeatures

    #new_X_test = poly.fit_transform(new_data_set.drop('charges',axis=1)) # new_data_set transforming features into PolynomialFeatures

    new_X_test = new_data_set.drop('charges',axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101) # normal splitting

    lr = RandomForestRegressor(n_estimators=13) # making a Ridge Regression variable

    lr.fit(X_train,y_train)

    predict = lr.predict(X_test) # prediction on splitted test data

    predict_on_new_data = lr.predict(new_X_test) # prediction on new data set

    print('MAE on Dataset: ', mean_absolute_error(y_test,predict))

    print('MSE on Dataset: ', mean_squared_error(y_test,predict))

    print('RMSR on Dataset: ', np.sqrt(mean_squared_error(y_test,predict)))

    #print('R2 Score on Dataset: ', r2_score(y_test,predict))

    #print('(poly deg 2 + ridge) linear model coeff (w):\n{}'.format(lr.coef_))

    #print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'.format(lr.intercept_))

    print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'

         .format(lr.score(X_train, y_train)))

    print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'

         .format(lr.score(X_test, y_test)))

    print('\n')

    print('MAE on New Dataset: ', mean_absolute_error(new_data_set['charges'],predict_on_new_data))

    print('MSE on New Dataset: ', mean_squared_error(new_data_set['charges'],predict_on_new_data))

    print('RMSR on New Dataset: ', np.sqrt(mean_squared_error(new_data_set['charges'],predict_on_new_data)))

    #print('R2 Score New Dataset: ', r2_score(new_data_set['charges'],predict_on_new_data))

    print('R-squared score (New test Dataset): {:.3f}'

     .format(lr.score(new_X_test, new_data_set['charges'])))

    print(predict_on_new_data)
new_data_set = med_df.iloc[[43,91]]

data(med_df.drop('charges',axis=1),med_df['charges'],new_data_set)
new_data_set