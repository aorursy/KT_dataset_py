import pandas as pd
import numpy as np

%matplotlib inline
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
print(train.iloc[:,0].count())
print(test.iloc[:,0].count())
print(train.head())
print(test.head())
def data_preprocessing(df):
    dates = pd.DataFrame(df.date.str.split('-').tolist(),
                                   columns = ['year','month','day'])
    df=pd.concat([dates,df],axis=1)
    del dates

    df['date']=df['date'].apply(pd.to_datetime)
    df[['year','month','day']]=df[['year','month','day']].apply(pd.to_numeric)


    df=df.drop('date',axis=1)
    df=df.drop('year',axis=1)

    df=pd.get_dummies(df,columns=['month','store','item'])
    return df
train =data_preprocessing(train)
train.head()
from sklearn.model_selection import train_test_split
train_x , test_x , train_y,test_y =train_test_split(train.iloc[:,train.columns != 'sales'],train['sales'],test_size=0.3,random_state=50)
#print(train_x.head() )
#print(test_x.head() )
#print(train_y.head() )
#print(test_y.head() )
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor  , AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
mae = {}
mse = {}
colums=[]
predicts=pd.DataFrame()
time_ = {}
start=time.time()


LinearModel = linear_model.LinearRegression()
LinearModel.fit(train_x, train_y)
p_LinearModel = LinearModel.predict(test_x)
mae.update({'LinearModel':mean_absolute_error(p_LinearModel, test_y)})
mse.update({'LinearModel':mean_squared_error(p_LinearModel, test_y)})
time_.update({'LinearModel':time.time()- start})
predicts['LinearModel']= pd.Series(p_LinearModel)
print("Model Evaluated: LinearModel\n")
start=time.time()


#    Kernel_Ridge = KernelRidge(alpha=1.0)
#    Kernel_Ridge.fit(train_x, train_y)
#    p_Kernel_Ridge = Kernel_Ridge.predict(test_x)
#    mae.update({'Kernel_Ridge':mean_absolute_error(p_Kernel_Ridge, test_y)})
#    mse.update({'Kernel_Ridge':mean_squared_error(p_Kernel_Ridge, test_y)})
#    time_.update({'Kernel_Ridge':time.time()- start})
#    predicts += (p_LinearModel,)
#    start=time.time()

'''
SVM = svm.SVR()
SVM.fit(train_x, train_y)
p_SVM = SVM.predict(test_x)
mae.update({'SVM':mean_absolute_error(p_SVM, test_y)}) 
mse.update({'SVM':mean_squared_error(p_SVM, test_y)})
time_.update({'SVM':time.time()- start})
predicts['SVM']= pd.Series(p_SVM)
print("Model Evaluated: SVM\n")
start=time.time()

SGDR = linear_model.SGDRegressor()
SGDR.fit(train_x, train_y)
p_SGDR = SGDR.predict(test_x)
mae.update({'SGDR':mean_absolute_error(p_SGDR, test_y)})
mse.update({'SGDR':mean_squared_error(p_SGDR, test_y)})
time_.update({'SGDR':time.time()- start})
print("Model Evaluated: SGDR\n")
predicts['SGDR']= pd.Series(p_SGDR)
start=time.time()
'''
#    GPR = GaussianProcessRegressor()
#    GPR.fit(train_x, train_y)
#    p_GPR = GPR.predict(test_x)
#    mae.update({'GaussianProcessRegressor':mean_absolute_error(p_GPR, test_y)}) 
#    mse.update({'GaussianProcessRegressor':mean_squared_error(p_GPR, test_y)})
#    time_.update({'GaussianProcessRegressor':time.time()- start})
#    print("Model Evaluated: GaussianProcessRegressor\n")
#    predicts += (p_LinearModel,)
#    start=time.time()

kNN = MLPRegressor(hidden_layer_sizes=(10,15))
kNN.fit(train_x, train_y)
p_kNN = kNN.predict(test_x)
mae.update({'NeuralNetwork':mean_absolute_error(p_kNN, test_y)})
mse.update({'NeuralNetwork':mean_squared_error(p_kNN, test_y)})
time_.update({'NeuralNetwork':time.time()- start})
print("Model Evaluated: NeuralNetwork\n")
predicts['NeuralNetwork']= pd.Series(p_kNN)
start=time.time()

#KNN = KNeighborsRegressor()
#KNN.fit(train_x, train_y)
#p_KNN = KNN.predict(test_x)
#mae.update({'KNeighborsRegressor':mean_absolute_error(p_KNN, test_y)})
#mse.update({'KNeighborsRegressor':mean_squared_error(p_KNN, test_y)})
#time_.update({'KNeighborsRegressor':time.time()- start})
#print("Model Evaluated: KNeighborsRegressor\n")
#predicts['KNeighborsRegressor']= pd.Series(p_KNN)
#start=time.time()


DT = DecisionTreeRegressor()
DT.fit(train_x, train_y)
p_DT = DT.predict(test_x)
mae.update({'DecisionTreeRegressor':mean_absolute_error(p_DT, test_y)}) 
mse.update({'DecisionTreeRegressor':mean_squared_error(p_DT, test_y)})
time_.update({'DecisionTreeRegressor':time.time()- start})
print("Model Evaluated: DecisionTreeRegressor\n")
predicts['DecisionTreeRegressor']= pd.Series(p_DT)
start=time.time()

GBR = GradientBoostingRegressor(n_estimators = 100, random_state=42,loss = 'ls')
GBR.fit(train_x, train_y)
p_GBR = GBR.predict(test_x)
mae.update({'GradientBoostingRegressor':mean_absolute_error(p_GBR, test_y)}) 
mse.update({'GradientBoostingRegressor':mean_squared_error(p_GBR, test_y)})
time_.update({'GradientBoostingRegressor':time.time()- start})
print("Model Evaluated: GradientBoostingRegressor\n")
predicts['GradientBoostingRegressor']= pd.Series(p_GBR)
start=time.time()

RFR = RandomForestRegressor()
RFR.fit(train_x, train_y)
p_RFR = RFR.predict(test_x)
mae.update({'RandomForestRegressor':mean_absolute_error(p_RFR, test_y)}) 
mse.update({'RandomForestRegressor':mean_squared_error(p_RFR, test_y)})
time_.update({'RandomForestRegressor':time.time()- start})
print("Model Evaluated: RandomForestRegressor\n")
predicts['RandomForestRegressor']= pd.Series(p_RFR)

start=time.time()

ADA = AdaBoostRegressor()
ADA.fit(train_x, train_y)
p_ADA = ADA.predict(test_x)
mae.update({'AdaBoostRegressor':mean_absolute_error(p_ADA, test_y)}) 
mse.update({'AdaBoostRegressor':mean_squared_error(p_ADA, test_y)})
time_.update({'AdaBoostRegressor':time.time()- start})
print("Model Evaluated: AdaBoostRegressor\n")
predicts['AdaBoostRegressor']= pd.Series(p_ADA)

start=time.time()

XGB = XGBRegressor()
XGB.fit(train_x, train_y)
p_XGB = XGB.predict(test_x)
mae.update({'XGBRegressor':mean_absolute_error(p_XGB, test_y)}) 
mse.update({'XGBRegressor':mean_squared_error(p_XGB, test_y)})
time_.update({'XGBRegressor':time.time()- start})
print("Model Evaluated: XGBRegressor\n")
predicts['XGBRegressor']= pd.Series(p_XGB)

#predicts=np.concatenate([p_LinearModel, p_SVM, p_SGDR, p_kNN, p_KNN, p_DT , p_GBR, p_RFR],axis=1)
#predicts_model = pd.DataFrame(np.concatenate((predicts)),columns=list(time_.keys()) )
#predicts_models.to_csv(r'D:\ABG\P_2_Eff\New\\model_comp_res.csv')
met = [mae,mse,time_]    
met[0]
from heapq import nsmallest
three_smallest = nsmallest(3,met[0],key=mae.get)
print(three_smallest)
test=data_preprocessing(test)
test.head()
RFR_P =RFR.predict(test.iloc[:,test.columns != 'id'])
DT_P =DT.predict(test.iloc[:,test.columns != 'id'])
XGB_P =XGB.predict(test.iloc[:,test.columns != 'id'])
RFR_sub=pd.DataFrame()
RFR_sub['sales']=pd.Series(RFR_P)
RFR_sub.to_csv('submission_RFR.csv',index_label='id')
DT_sub=pd.DataFrame()
DT_sub['sales']=pd.Series(DT_P)
DT_sub.to_csv('submission_DT.csv',index_label='id')
XGB_sub=pd.DataFrame()
XGB_sub['predicts']=pd.Series(XGB_P)
XGB_sub.to_csv('submission_XGB.csv',index_label='id')


