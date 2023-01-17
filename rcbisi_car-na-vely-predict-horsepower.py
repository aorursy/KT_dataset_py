import os
os.listdir("../input")
import pandas as pd
car=pd.read_csv('../input/Automobile_data.csv')
# we read that: 
# horsepower = torque * rmp, 
# Longer the stroke higher the torque, shorter the stroke lower the torque.
# So we absolutely can't drop 'stroke' or 'peak-rpm'
not_to_drop = ['bore','stroke','peak-rpm']
car.shape
car = car.replace({'?': None})
car.isnull().sum().sum()
if (car.dtypes['num-of-cylinders'] == 'O'):
    car['num-of-cylinders'].replace({'two':2, 'four':4 , 'three':3, 'eight':8, 'six':6, 'five':5, 'twelve':12}, inplace=True)
car.dtypes['num-of-cylinders']
dropped = pd.DataFrame()
try: 
    dropped
except:
    dropped = pd.DataFrame()
else:
    if (len(dropped) == 0):        
        for i in car.columns:
            if((len(car[i].value_counts()) < 51) and (car.dtypes[i] == 'O') and (not(i in not_to_drop))):
                dropped[i] = car[i]  
                car = car.drop([i],axis=1)
        #        dropped.append(i) ... or we could append the name i to a list of dropped columns..
            else: 
                car[i] = car[i].astype(float) 
                b = car[i].mean()
                print(i, len(car[i].value_counts()), b)
                car[i] = car[i].replace({None : b})        
dropped.sample(3)
import numpy as np
car['torque'] = car['horsepower']/car['peak-rpm']*5252
torque_predictors = ['num-of-cylinders', 'engine-size', 'bore', 'stroke', 'compression-ratio']
Xr = car[torque_predictors]
yr = car['torque']
X_array = np.array(Xr) #r stands for 'regression'
y_array = np.array(yr)
# We should not ignore engine_type and fuel_system here ... 
#car.columns
# we read that: 
# HP = Torque x RPM ÷ 5252
# Longer the stroke higher the torque, shorter the stroke lower the torque.
# So let's predict torque too. 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
loo = LeaveOneOut()
ytests = []
ypreds = []
for train_idx, test_idx in loo.split(Xr):
    X_train, X_test = X_array[train_idx], X_array[test_idx] #requires arrays
    y_train, y_test = y_array[train_idx], y_array[test_idx]
    
    model = LinearRegression()
    model.fit(X = X_train, y = y_train) 
    y_pred = model.predict(X_test)
        
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
        
    ytests += list(y_test)
    ypreds += list(y_pred)
from sklearn import metrics        
rr = metrics.r2_score(ytests, ypreds)
ms_error = metrics.mean_squared_error(ytests, ypreds)
print("Leave One Out Cross Validation")
print("R^2: {:.5f}%, MSE: {:.5f}".format(rr*100, ms_error))
