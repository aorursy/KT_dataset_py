import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
df = pd.read_csv("/kaggle/input/craigslist-carstrucks-data/vehicles.csv")
df.head()
df.shape
df.info()
# Clean data
df= df.drop(columns=['id','url', 'region_url', 'vin', 'image_url', 'description', 'lat', 'long','county','region'], axis=1)
df.head()
is_bmw = df["manufacturer"] =="bmw"
df_bmw = df[is_bmw]
df_bmw.head()
is_clean = df_bmw["title_status"] == "clean"
df_clean = df_bmw[is_clean]
df_clean.head()
df_clean = df_clean.drop(columns=['title_status'], axis=1)
df_clean.head()
# Dropping more data
df_clean = df_clean.drop(columns=['size', 'type', 'paint_color', 'state'])
df_clean.head()
df_clean = df_clean.drop(columns=['manufacturer'])
df_clean.head()
df_clean1 = df_clean.groupby('model').filter(lambda x: len(x) >= 50)
df_clean1['model'].value_counts()
rr = sorted(df_clean1["price"])
quantile1, quantile3= np.percentile(rr,[10,90])
print(quantile1,quantile3)
# Perform filtering
df_clean2 = df_clean1[(df_clean1.price < 19950) & (df_clean1.price >= 4900)]
df_clean2.shape
# Onehot cyl
oh = pd.get_dummies(df_clean2.cylinders)
df_oh = df_clean2.join(oh)
df_oh.head()
df_oh = df_oh.drop(columns=['cylinders'], axis=1)
df_oh.head()
df_oh.shape
oh = pd.get_dummies(df_oh.condition)
df_oh = df_oh.join(oh)
df_oh = df_oh.drop(columns=['condition'], axis=1)
df_oh.head()
df_oh.shape
oh = pd.get_dummies(df_oh.transmission)
oh = oh.drop(columns=["other"], axis=1)
df_oh = df_oh.join(oh)
df_oh = df_oh.drop(columns=['transmission'], axis=1)
df_oh.head()
oh = pd.get_dummies(df_oh.fuel)
oh = oh.drop(columns=["other"], axis=1)
df_oh = df_oh.join(oh)
df_oh = df_oh.drop(columns=['fuel'], axis=1)
df_oh.head()
oh = pd.get_dummies(df_oh.model)
df_oh = df_oh.join(oh)
df_oh = df_oh.drop(columns=['model'], axis=1)
df_oh.head()
oh = pd.get_dummies(df_oh.drive, prefix="drive")
df_oh = df_oh.join(oh)
df_oh = df_oh.drop(columns=['drive'], axis=1)
df_oh.head()
df_oh_final = df_oh[df_oh.odometer.notnull()]
print(df_oh_final.shape)
print(df_oh.shape)
# Modeling prep
target_name = 'price'
train_target0 = df_oh_final[target_name]
train0 = df_oh_final.drop([target_name], axis=1)
train0, test0, train_target0, test_target0 = train_test_split(train0, train_target0, test_size=0.2, random_state=0)
# For boosting model
train0b = train0
train_target0b = train_target0
# Synthesis valid as test for selection models
trainb, testb, targetb, target_testb = train_test_split(train0b, train_target0b, test_size=0.3, random_state=0)
train0.head()
train, test, target, target_test = train_test_split(train0, train_target0, test_size=0.3, random_state=0)
acc_train_r2 = []
acc_test_r2 = []
acc_train_d = []
acc_test_d = []
acc_train_rmse = []
acc_test_rmse = []
def acc_d(y_meas, y_pred):
    # Relative error between predicted y_pred and measured y_meas values
    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))

def acc_rmse(y_meas, y_pred):
    # RMSE between predicted y_pred and measured y_meas values
    return (mean_squared_error(y_meas, y_pred))**0.5
def acc_boosting_model(num,model,train,test,num_iteration=0):
    # Calculation of accuracy of boosting model by different metrics
    
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    if num_iteration > 0:
        ytrain = model.predict(train, num_iteration = num_iteration)  
        ytest = model.predict(test, num_iteration = num_iteration)
    else:
        ytrain = model.predict(train)  
        ytest = model.predict(test)

    print('target = ', targetb[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(targetb, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_testb[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_testb, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)
def acc_model(num,model,train,test):
    # Calculation of accuracy of model акщь Sklearn by different metrics   
  
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    ytrain = model.predict(train)  
    ytest = model.predict(test)

    print('target = ', target[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(target, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_test[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_test, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)
linreg = LinearRegression()
linreg.fit(train, target)
acc_model(0,linreg,train,test)
target_test.to_numpy()
preds = linreg.predict(test)
tp0 = target_test.to_numpy()
mae = 0
for pred in range(len(preds)):
    mae += np.abs(preds[pred] - tp0[pred])
print(mae/len(preds))

print(preds)
print(tp0)
linreg.coef_