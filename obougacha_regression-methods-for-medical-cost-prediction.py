#let's start with importing the libraries we need to:

# 1- read and manipulate data:

import pandas as pd

import numpy as np

# 2- visualize data and graphs

import matplotlib.pyplot as plt

import seaborn as sns
raw_data=pd.read_csv('../input/insurance/insurance.csv')

X=raw_data.drop(["charges"],axis=1)

y=raw_data["charges"]

raw_data.head()
raw_data.isnull().sum()
plt.figure(figsize=(16,16))

i=1

for elt in X.columns:

    plt.subplot(3,2,i)

    plt.scatter(X[elt],y)

    plt.xlabel(elt)

    plt.ylabel('Medical Cost (u.m.)')

    i+=1

plt.tight_layout(0.05)

plt.show()
plt.figure(figsize=(8,5))

plt.scatter(X['age'],y)

plt.plot([17,65,65,17,17],[4000,18000,10000,0,4000],color='red')

plt.plot([17,65,65,17,17],[25000,36000,20000,10000,25000],color='red')

plt.plot([17,65,65,17,17],[40000,55000,40000,30000,40000],color='red')

plt.xlabel('Age (years)')

plt.ylabel('Medical Cost (u.m.)')

plt.show()
from mpl_toolkits.mplot3d import Axes3D

threedee = plt.figure(figsize=(10,8)).gca(projection='3d')

threedee.scatter(X["age"], X['bmi'],y)

threedee.set_xlabel('Age (years)')

threedee.set_ylabel('BMI')

threedee.set_zlabel('Medical Cost (u.m.)')

plt.show()
from mpl_toolkits.mplot3d import Axes3D

threedee = plt.figure(figsize=(10,8)).gca(projection='3d')

threedee.scatter(X["age"], X['children'],y)

threedee.set_xlabel('Age (years)')

threedee.set_ylabel('children')

threedee.set_zlabel('Medical Cost (u.m.)')

plt.show()
plt.figure(figsize=(10,10))

cat_col=['smoker','sex','region','children']

i=1

for elt in cat_col:

    plt.subplot(2,2,i)

    sns.boxplot(x=X[elt],y=y)

    i+=1

plt.show()
hist = raw_data.hist(bins=100,color='red',figsize=(16, 16))
X['smoker']=X['smoker'].apply(lambda x : 1 if x=="yes" else 0)

X.head(5)
X['sex']=X['sex'].apply(lambda x : 1 if x=='male' else 0)

X.head(5)
for region_name in ['southwest', 'southeast', 'northwest', 'northeast']:

    X[region_name]=X['region'].apply(lambda x : 1 if x==region_name else 0)

X= X.drop('region',axis=1)

X.head(5)
f = plt.figure(figsize=(19, 15))

df=pd.concat([X, y], axis=1)

plt.matshow(df.corr(), fignum=f.number)

plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)

plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16,y=-0.08)

plt.show()
from sklearn.metrics import mean_squared_error,r2_score

from math import sqrt
Performance={}

Performance["Method"]=[]

Performance["R_squared (train)"]=[]

Performance["R_squared (test)"]=[]

Performance["RMSE (train)"]=[]

Performance["RMSE (test)"]=[]
Predictions=pd.DataFrame()

Training_Preds=pd.DataFrame()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.drop('southwest',axis=1).values,y.values,

                                                    test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression



SLR = LinearRegression()

SLR.fit(X_train,y_train)

y_pred=SLR.predict(X_test)

y_p_t =SLR.predict(X_train)

Performance["Method"].append('SLR')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

print('SLR',r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,y_pred)))

Predictions["Ground Truth"]=y_test

Training_Preds["Ground Truth"]=y_train

Predictions['SLR']=y_pred

Training_Preds["SLR"]=y_p_t
#we are going to eliminate variables that won't make a difference 

#we add the constant variable x_0

X_Ttrain = np.append(np.ones((len(X_train),1)).astype(int),X_train,1)

X_Ttest = np.append(np.ones((len(X_test),1)).astype(int),X_test,1)
import statsmodels.api as sm

#our X_optimal is initialized to X_Ttrain

X_opt=X_Ttrain[:,:]
#Step 1 :Fit the ALL IN model

model_MLR=sm.OLS(endog=y_train,exog=X_opt).fit()

model_MLR.summary()
Column_to_delete=2

columns_to_keep=[]

for elt in range(X_opt.shape[1]):

    if elt != Column_to_delete :

        columns_to_keep.append(elt)

X_opt = X_opt[:,columns_to_keep]

X_Ttest = X_Ttest[:,columns_to_keep]

X_opt.shape
model_MLR=sm.OLS(endog=y_train,exog=X_opt).fit()

model_MLR.summary()
Column_to_delete=5

columns_to_keep=[]

for elt in range(X_opt.shape[1]):

    if elt != Column_to_delete :

        columns_to_keep.append(elt)

X_opt = X_opt[:,columns_to_keep]

X_Ttest = X_Ttest[:,columns_to_keep]

X_opt.shape
model_MLR=sm.OLS(endog=y_train,exog=X_opt).fit()

model_MLR.summary()
Column_to_delete=5

columns_to_keep=[]

for elt in range(X_opt.shape[1]):

    if elt != Column_to_delete :

        columns_to_keep.append(elt)

X_opt = X_opt[:,columns_to_keep]

X_Ttest = X_Ttest[:,columns_to_keep]

X_opt.shape
model_MLR=sm.OLS(endog=y_train,exog=X_opt).fit()

model_MLR.summary()
Column_to_delete=5

columns_to_keep=[]

for elt in range(X_opt.shape[1]):

    if elt != Column_to_delete :

        columns_to_keep.append(elt)

X_opt = X_opt[:,columns_to_keep]

X_Ttest = X_Ttest[:,columns_to_keep]

X_opt.shape
model_MLR=sm.OLS(endog=y_train,exog=X_opt).fit()

model_MLR.summary()
from sklearn.linear_model import LinearRegression

MLR=LinearRegression()

X_Ttrain = X_opt[:,1:]

X_Ttest = X_Ttest[:,1:]

MLR.fit(X_Ttrain,y_train)

y_pred = MLR.predict(X_Ttest)

y_p_t = MLR.predict(X_Ttrain)

Performance["Method"].append('MLR')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

print('MLR',r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,y_pred)))

Predictions['MLR']=y_pred

Training_Preds["MLR"]=y_p_t
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

degrees=[2,3,4,5,6]

for deg in degrees:

    Poly_Trans = PolynomialFeatures(degree=deg)

    X_poly_train = Poly_Trans.fit_transform(X_train)

    X_poly_test = Poly_Trans.transform(X_test)

    Poly_SLR=LinearRegression()

    Poly_SLR.fit(X_poly_train,y_train)

    y_pred = Poly_SLR.predict(X_poly_test)

    y_p_t = Poly_SLR.predict(X_poly_train)

    Performance["Method"].append('Poly_SLR_d={}'.format(deg))

    Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

    Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

    Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

    Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

    print('Poly_SLR_d={}'.format(deg),r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,y_pred)))

    Predictions['Poly_SLR_d={}'.format(deg)]=y_pred

    Training_Preds['Poly_SLR_d={}'.format(deg)]=y_p_t
from sklearn.tree import DecisionTreeRegressor

DTR = DecisionTreeRegressor(random_state=0)

DTR.fit(X_train,y_train)

y_pred = DTR.predict(X_test)

y_p_t = DTR.predict(X_train)

Performance["Method"].append('DTR')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

print('DTR',r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,y_pred)))

Predictions['DTR']=y_pred

Training_Preds['DTR']=y_p_t
from sklearn.svm import SVR

L_SVR = SVR(kernel='linear',epsilon=0.1,C=1e4)

L_SVR.fit(X_train,y_train)

y_pred = L_SVR.predict(X_test)

y_p_t = L_SVR.predict(X_train)

Performance["Method"].append('L_SVR')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

print('L_SVR',r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,y_pred)))

Predictions['L_SVR']=y_pred

Training_Preds['L_SVR']=y_p_t
from sklearn.preprocessing import MinMaxScaler



X_Scaler = MinMaxScaler()

X_train_s=X_Scaler.fit_transform(X_train)

X_test_s = X_Scaler.transform(X_test)



y_Scaler = MinMaxScaler()

y_train_s=y_Scaler.fit_transform(y_train.reshape(-1,1))

y_train_s=y_train_s.reshape(len(y_train),)

y_test_s=y_Scaler.transform(y_test.reshape(-1,1))

y_test_s=y_test_s.reshape(len(y_test),)
from sklearn.svm import SVR

RBF_SVR = SVR(kernel='rbf',C=1e4,gamma='auto')

RBF_SVR.fit(X_train_s,y_train_s)

y_pred_s = RBF_SVR.predict(X_test_s)

y_pred = y_Scaler.inverse_transform(y_pred_s.reshape(-1,1))

y_p_t = y_Scaler.inverse_transform(RBF_SVR.predict(X_train_s).reshape(-1,1))

Performance["Method"].append('RBF_SVR')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

print('RBF_SVR',r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,y_pred)))

Predictions['RBF_SVR']=y_pred

Training_Preds['RBF_SVR']=y_p_t
from sklearn.neighbors import KNeighborsRegressor

#for different k values example [1,10,50]

k_values =[1,10,20,50]

for k in k_values:

    k_nn=KNeighborsRegressor(n_neighbors = k)

    k_nn.fit(X_train_s, y_train_s)  

    y_pred_s=k_nn.predict(X_test_s)

    y_pred = y_Scaler.inverse_transform(y_pred_s.reshape(-1,1))

    y_p_t = y_Scaler.inverse_transform(k_nn.predict(X_train_s).reshape(-1,1))

    Performance["Method"].append('{}-NN'.format(k))

    Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

    Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

    Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

    Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

    print('{}-NN'.format(k),r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,

                                                                                                    y_pred)))

    Predictions['{}-NN'.format(k)]=y_pred

    Training_Preds['{}-NN'.format(k)]=y_p_t
from sklearn.neighbors import RadiusNeighborsRegressor

R=[0.4,0.5,1,2,3]

for rad in R:

    RN = RadiusNeighborsRegressor(radius=rad)

    RN.fit(X_train_s,y_train_s)

    y_pred_s=RN.predict(X_test_s)

    y_pred = y_Scaler.inverse_transform(y_pred_s.reshape(-1,1))

    y_p_t = y_Scaler.inverse_transform(RN.predict(X_train_s).reshape(-1,1))

    Performance["Method"].append('{}-N'.format(rad))

    Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

    Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

    Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

    Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

    print('{}-N'.format(rad),r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,

                                                                                                     y_pred)))

    Predictions['{}-N'.format(rad)]=y_pred

    Training_Preds['{}-N'.format(rad)]=y_p_t
n_trees=100

from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(n_trees,random_state=0)

RFR.fit(X_train,y_train)

y_pred=RFR.predict(X_test)

y_p_t=RFR.predict(X_train)

Performance["Method"].append('RFR')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

print('RFR',r2_score(y_test,y_pred),sqrt(mean_squared_error(y_test,y_pred)))

Predictions['RFR']=y_pred

Training_Preds['RFR']=y_p_t
from sklearn.ensemble import VotingRegressor

unscaling_Regressors=[("SLR",SLR), ("L_SVR",L_SVR), ("DTR",DTR), ("RFR",RFR)]

unscaling_Ensemble = VotingRegressor(unscaling_Regressors)

unscaling_Ensemble.fit(X_train, y_train)

y_pred = unscaling_Ensemble.predict(X_test)

y_p_t = unscaling_Ensemble.predict(X_train)

Performance["Method"].append('Unscaled_Ensemble')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

Predictions["Unscaled_Ensemble"]=y_pred

Training_Preds['Unscaled_Ensemble']=y_p_t
from sklearn.ensemble import VotingRegressor

k_nn = KNeighborsRegressor(n_neighbors = 10)

RN = RadiusNeighborsRegressor(radius=0.5)

scaling_Regressors=[('RBF_SVR',RBF_SVR), ('K-NN',k_nn), ('R-N',RN)]

scaling_Ensemble = VotingRegressor(scaling_Regressors)

scaling_Ensemble.fit(X_train_s,y_train_s)

y_pred_s = scaling_Ensemble.predict(X_test_s)

y_pred = y_Scaler.inverse_transform(y_pred_s.reshape(-1,1))

y_p_t = y_Scaler.inverse_transform(scaling_Ensemble.predict(X_train_s).reshape(-1,1))

Performance["Method"].append('scaled_Ensemble')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

Predictions["scaled_Ensemble"]=y_pred

Training_Preds['scaled_Ensemble']=y_p_t
regressors_models=[('RBF_SVR',RBF_SVR), ('K-NN',k_nn), ('R-N',RN), ("SLR",SLR), ("L_SVR",L_SVR),

                   ("DTR",DTR), ("RFR",RFR)]

All_Ensemble = VotingRegressor(regressors_models)

All_Ensemble.fit(X_train_s,y_train_s)

y_pred_s = All_Ensemble.predict(X_test_s)

y_pred = y_Scaler.inverse_transform(y_pred_s.reshape(-1,1))

y_p_t = y_Scaler.inverse_transform(All_Ensemble.predict(X_train_s).reshape(-1,1))

Performance["Method"].append('ALL_Ensemble')

Performance["R_squared (train)"].append(r2_score(y_train,y_p_t))

Performance["R_squared (test)"].append(r2_score(y_test,y_pred))

Performance["RMSE (train)"].append(sqrt(mean_squared_error(y_train,y_p_t)))

Performance["RMSE (test)"].append(sqrt(mean_squared_error(y_test,y_pred)))

Predictions["ALL_Ensemble"]=y_pred

Training_Preds['ALL_Ensemble']=y_p_t
Performance = pd.DataFrame(Performance)

Performance
Performance = Performance.drop(6).reset_index()
plt.figure(figsize=(7,7))

Performance["R_squared (train)"].plot(color="blue",label="R² Train")

Performance["R_squared (test)"].plot(color="red",label="R² Test")

plt.xticks(Performance.index,Performance["Method"].tolist(),rotation=90)

plt.ylabel("R² Value")

plt.legend()

plt.show()
plt.figure(figsize=(7,7))

Performance["RMSE (train)"].plot(color="blue",label="RMSE Train")

Performance["RMSE (test)"].plot(color="red",label="RMSE Test")

plt.xticks(Performance.index,Performance["Method"].tolist(),rotation=90)

plt.ylabel("RMSE Value")

plt.legend()

plt.show()
plt.figure(figsize=(16,100))

i=1

Cols = Predictions.columns[1:]

for elt in Cols:

    plt.subplot(24,2,i)

    plt.scatter(Training_Preds.index,Training_Preds['Ground Truth'],color = 'blue', label='Ground Truth')

    Training_Preds[elt].plot(color='red',label="Training with {}".format(elt))

    plt.legend()

    i+=1

    plt.subplot(24,2,i)

    plt.scatter(Predictions.index,Predictions['Ground Truth'],color = 'blue', label='Ground Truth')

    Predictions[elt].plot(color='red',label="Predictions with {}".format(elt))

    plt.legend()

    i+=1

plt.show()