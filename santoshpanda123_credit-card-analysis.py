

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



retail_dataset=pd.read_excel("../input/Credit Card _Dataset.xls")

retail_dataset.sample(5)
retail_dataset.reset_index(inplace=True)
retail_dataset.drop(columns='ID',inplace=True)

retail_dataset.drop(columns='index',inplace=True)
retail_dataset.head()
retail_dataset.describe()
retail_dataset.isna().sum()
y=retail_dataset['LIMIT_BAL']

x=retail_dataset.drop(['LIMIT_BAL'],axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.ensemble import RandomForestRegressor

lm=RandomForestRegressor()

lm.fit(x_train,y_train)

lm.predict(x_test)

pred_values_1=lm.predict(x_test)
import numpy as np

def regression_model(predicted_values,y_test):

    

    from sklearn.metrics import mean_absolute_error,r2_score

    from sklearn.metrics import mean_squared_error

    total_error=sum(abs(predicted_values-y_test.values))

    MSE=mean_absolute_error(y_test.values,predicted_values)

    MAE=mean_squared_error(y_test.values,predicted_values)

    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))

    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]

    r2=r2_score(predicted_values,y_test)

    print("total error",total_error)

    print("MSE",MSE)

    print("MAE",MAE)

    print("RMSE",RMSE)

    print("MAPE",MAPE)

    print("R2",r2)

    return [MSE, MAE,RMSE,MAPE,r2]
erros_rf=regression_model(pred_values_1,y_test)
from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(x_train,y_train)

lm.predict(x_test)

pred_values_2=lm.predict(x_test)
import numpy as np

def regression_model(predicted_values,y_test):

    

    from sklearn.metrics import mean_absolute_error,r2_score

    from sklearn.metrics import mean_squared_error

    total_error=sum(abs(predicted_values-y_test.values))

    MSE=mean_absolute_error(y_test.values,predicted_values)

    MAE=mean_squared_error(y_test.values,predicted_values)

    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))

    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]

    r2=r2_score(predicted_values,y_test)

    print("total error",total_error)

    print("MSE",MSE)

    print("MAE",MAE)

    print("RMSE",RMSE)

    print("MAPE",MAPE)

    print("R2",r2)

    return [MSE, MAE,RMSE,MAPE,r2]
erros_reg=regression_model(pred_values_2,y_test)
comparision=pd.DataFrame(pred_values_1,pred_values_2)

comparision.reset_index(inplace=True)

comparision.columns=['Linear','Random Forest']

comparision['actual']=y_test.values

comparision
from sklearn.tree import DecisionTreeRegressor

lm=DecisionTreeRegressor()

lm.fit(x_train,y_train)

lm.predict(x_test)

pred_values_3=lm.predict(x_test)
import numpy as np

def regression_model(predicted_values,y_test):

    

    from sklearn.metrics import mean_absolute_error,r2_score

    from sklearn.metrics import mean_squared_error

    total_error=sum(abs(predicted_values-y_test.values))

    MSE=mean_absolute_error(y_test.values,predicted_values)

    MAE=mean_squared_error(y_test.values,predicted_values)

    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))

    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]

    r2=r2_score(predicted_values,y_test)

    print("total error",total_error)

    print("MSE",MSE)

    print("MAE",MAE)

    print("RMSE",RMSE)

    print("MAPE",MAPE)

    print("R2",r2)

    return [MSE, MAE,RMSE,MAPE,r2]
err_decision=regression_model(pred_values_3,y_test)
from sklearn.neighbors import KNeighborsRegressor

lm=KNeighborsRegressor()

lm.fit(x_train,y_train)

lm.predict(x_test)

pred_values_4=lm.predict(x_test)
import numpy as np

def regression_model(predicted_values,y_test):

    

    from sklearn.metrics import mean_absolute_error,r2_score

    from sklearn.metrics import mean_squared_error

    total_error=sum(abs(predicted_values-y_test.values))

    MSE=mean_absolute_error(y_test.values,predicted_values)

    MAE=mean_squared_error(y_test.values,predicted_values)

    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))

    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]

    r2=r2_score(predicted_values,y_test)

    print("total error",total_error)

    print("MSE",MSE)

    print("MAE",MAE)

    print("RMSE",RMSE)

    print("MAPE",MAPE)

    print("R2",r2)

    return [MSE, MAE,RMSE,MAPE,r2]
error_knn=regression_model(pred_values_4,y_test)
from sklearn.ensemble import BaggingRegressor

lm=BaggingRegressor()

lm.fit(x_train,y_train)

lm.predict(x_test)

pred_values_5=lm.predict(x_test)
import numpy as np

def regression_model(predicted_values,y_test):

    

    from sklearn.metrics import mean_absolute_error,r2_score

    from sklearn.metrics import mean_squared_error

    total_error=sum(abs(predicted_values-y_test.values))

    MSE=mean_absolute_error(y_test.values,predicted_values)

    MAE=mean_squared_error(y_test.values,predicted_values)

    RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))

    MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/x_test.shape[0]

    r2=r2_score(predicted_values,y_test)

    print("total error",total_error)

    print("MSE",MSE)

    print("MAE",MAE)

    print("RMSE",RMSE)

    print("MAPE",MAPE)

    print("R2",r2)

    return [MSE, MAE,RMSE,MAPE,r2]
error_bagging=regression_model(pred_values_5,y_test)
error_knn
err_decision
error_bagging
error_knn
erros_reg
erros_rf
comparision[['Random Forest','actual']][0:100].plot(figsize=(20,10))