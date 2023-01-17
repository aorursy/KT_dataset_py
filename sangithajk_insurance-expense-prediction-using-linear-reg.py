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
def data_exploration(data1,y_value):

    from sklearn.preprocessing import LabelEncoder

    import seaborn as sns



    x_value=[]

    out_data=pd.DataFrame()

    data_cols=data1.columns

    for i in data_cols:

        if data1[i].dtype == object:

            print('converting to numerical value  :',i)

            LE = LabelEncoder()

            colname = i+'_code'

            out_data[colname] = LE.fit_transform(data1[i])

            if i != y_value:

                x_value.append(colname)

        else:

            out_data[i]=data1[i]

            if i != y_value:

                x_value.append(i)

    print('============================CORRELATION==========================')

    print(out_data.corr())

    sns.pairplot(data=out_data,x_vars=x_value,y_vars=y_value)

    return out_data

def drop_low_correlation(out_data1,col_names):

    out_data1.drop(columns=col_names,inplace=True)

    

def linear_reg(out_data1,y_value):

    import numpy as np

    import sklearn

    from sklearn.model_selection import train_test_split

    from sklearn.linear_model import LinearRegression

    from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error, r2_score

    

    y=out_data1[y_value]

    X=out_data1.drop(columns=y_value)

    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=1)

    print('Train X shape: ',train_X.shape, 'Train Y shape:',train_y.shape)

    model = LinearRegression()

    model.fit(train_X,train_y)

    print('**** Linear Regression Output  ****')

    print('Intercept:',model.intercept_,'Slope:',model.coef_)

    train_predict = model.predict(train_X)

    test_predict=model.predict(test_X)

    print('****** ERRORS ********')

    print('MAE train', mean_absolute_error(train_y,train_predict))

    print('MAE test', mean_absolute_error(test_y,test_predict))

    print(' ')

    print('MSE train',mean_squared_error(train_y,train_predict))

    print('MSE test',mean_squared_error(test_y,test_predict))

    print(' ')

    print('RMSE train',np.sqrt(mean_squared_error(train_y,train_predict)))

    print('RMSE test',np.sqrt(mean_squared_error(test_y,test_predict)))

    print(' ')

    print('r2 train',r2_score(train_y,train_predict))

    print('r2 test',r2_score(test_y,test_predict))
import pandas as pd

insurance=pd.read_csv('../input/insurance.csv')

print(insurance.info())

out_data1=data_exploration(insurance,'expenses')



# if there are null values try to replace them

# drop low correlation rows

drop_cols=[]

drop_low_correlation(out_data1,drop_cols)

# if x columns have correlation do feature engineering

eeng_cols = []

# execute Linear Regression

linear_reg(out_data1,'expenses')