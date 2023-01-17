import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from time import time
import os
#print(os.listdir("../input"))
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/womenintheloop-data-science-hackathon/train.csv')
test = pd.read_csv('/kaggle/input/womenintheloop-data-science-hackathon/test_QkPvNLx.csv')

train.head()
test.head()
train.describe()
train.describe(include='object')
test.describe()
test.describe(include='object')
# checking for null values
train.isnull().sum()
train.fillna(train['Competition_Metric'].median(),inplace=True)
test.isnull().sum()
test.fillna(train['Competition_Metric'].median(),inplace=True)
train['Day_No'].value_counts()
test['Day_No'].value_counts().head()
train['Course_ID'].value_counts()
test['Course_ID'].value_counts()
sns.countplot(train['Course_Domain'])
sns.countplot(train['Course_Type'])
sns.countplot(test['Course_Domain'])
sns.countplot(test['Course_Type'])
sns.countplot(train['Short_Promotion']) 
sns.countplot(test['Short_Promotion']) 
sns.countplot(train['Public_Holiday'])
sns.countplot(test['Public_Holiday'])
sns.countplot(train['Long_Promotion'])
sns.countplot(test['Long_Promotion'])
#target variable
sns.distplot(train['Sales'])
# it is skewed hence while performing model,log1p transformation is applied
sns.distplot(train['User_Traffic'])
sns.distplot(train['Competition_Metric'])
sns.distplot(test['Competition_Metric'])
# sales of course_ID 1 full timne
strain = train[train.Sales>0]
strain.loc[strain['Course_ID']==1 ,['Day_No','Sales']].plot(x='Day_No',y='Sales',title='Course_ID 1',figsize=(16,4))

# sales of course_ID 2  full timne
strain = train[train.Sales>0]
strain.loc[strain['Course_ID']==2 ,['Day_No','Sales']].plot(x='Day_No',y='Sales',title='Course_ID 2',figsize=(16,4))

#sales of course_id 1 ,day no 240 vs 605
strain = train[train.Sales>0]
strain.loc[strain['Course_ID']==1 ,['Day_No','Sales']]     .plot(x='Day_No',y='Sales',title='Course_ID 1',figsize=(8,2),xlim=[240,605])
strain.loc[strain['Course_ID']==1 ,['Day_No','Sales']]     .plot(x='Day_No',y='Sales',title='Course_ID 1',figsize=(8,2),xlim=[240,605])
# drop user_traffic variable as it is not present is test data
train1=train.copy()
train1.drop('User_Traffic',axis=1,inplace=True)
# Creating new feature Day_of_week in train and test data
train1['Day_of_week']=train1['Day_No'].apply(lambda x:x%7)
test1=test.copy()
test1['Day_of_week']=test1['Day_No'].apply(lambda x:x%7)
# one hot encoding for categorical columns
train1=pd.get_dummies(train1,columns=['Course_Domain','Course_Type'],drop_first=True)
test1=pd.get_dummies(test1,columns=['Course_Domain','Course_Type'],drop_first=True)
# splitting train data into train and test(for validating)
ho_test = train1[:6*7*600]

ho_train = train1[6*7*600:]
plt.subplots(figsize=(24,20))
sns.heatmap(ho_train.corr(),annot=True, vmin=-0.1, vmax=0.1,center=0)
# defining RMSLE metric (Root mean squared log error)
from sklearn.metrics import mean_squared_log_error
def rmsle(y, yhat):
    return (np.sqrt(mean_squared_log_error(y, yhat)))*1000

def rmsle_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmsle", rmsle(y,yhat)
# as the target variable is skewed applying log1p transformation
ho_xtrain = ho_train.drop(['Sales','ID'],axis=1 )
ho_ytrain = np.log1p(ho_train.Sales)
ho_xtest = ho_test.drop(['Sales','ID'],axis=1 )
ho_ytest = np.log1p(ho_test.Sales)
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.03,
          "max_depth": 14,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 10
          }
num_boost_round = 6000


dtrain = xgb.DMatrix(ho_xtrain, ho_ytrain)
dvalid = xgb.DMatrix(ho_xtest, ho_ytest)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
print("Train a XGBoost model")
start = time()
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, 
  early_stopping_rounds=100, feval=rmsle_xg, verbose_eval=True)
end = time()
print('Training time is {:2f} s.'.format(end-start))
print("validating")
ho_xtest.sort_index(inplace=True) 
ho_ytest.sort_index(inplace=True) 
yhat = gbm.predict(xgb.DMatrix(ho_xtest))
error = rmsle(np.expm1(ho_ytest), np.expm1(yhat))

print('RMSPE: {:.6f}'.format(error))
res = pd.DataFrame(data = ho_ytest)
res['Prediction']=yhat
res = pd.merge(ho_xtest,res, left_index= True, right_index=True)
res['Ratio'] = res.Prediction/res.Sales
res['Error'] =abs(res.Ratio-1)
res['Weight'] = res.Sales/res.Prediction
res.head()
for i in range(1,5):
    s1 = pd.DataFrame(res[res['Course_ID']==i],columns = ['Sales','Prediction'])
    s2 = pd.DataFrame(res[res['Course_ID']==i],columns = ['Ratio'])
    s1.plot(figsize=(12,4))
    s2.plot(figsize=(12,4))
    plt.show()
test2=test1.drop(['ID'],axis=1)
print("Make predictions on the test set")

dtest = xgb.DMatrix(test2)
test_probs = gbm.predict(dtest)

# model1  kaggle private score 0.12647
result = pd.DataFrame({"ID": test1['ID'], 'Sales': np.expm1(test_probs)})
result
print("weight correction")
W=[(0.995+(i/1000)) for i in range(30)]
S =[]
for w in W:
    error = rmsle(np.expm1(ho_ytest), np.expm1(yhat*w))
    print('RMSPE for {:.3f}:{:.6f}'.format(w,error))
    S.append(error)
Score = pd.Series(S,index=W)
Score.plot()
BS = Score[Score.values == Score.values.min()]
print ('Best weight for Score:{}'.format(BS))
result_w3 = pd.DataFrame({"ID": test1['ID'], 'Sales': np.expm1(test_probs*1.001)})
result_w3 #.to_csv('gbmm7.csv',index=False) 
# out of the different models XGBoost Worked well and also Random Forest gave relatively good results.
# The predictions can be more improved provided there are more features
# Features like Year and Week of year were added and different models are builded but none of them performed better.Hence they are removed and only dy of week feature is added
# All the features were important and ID feature has much correlation with Course_ID variable hence it is dropped

