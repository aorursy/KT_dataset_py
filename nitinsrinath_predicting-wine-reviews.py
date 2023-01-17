import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
wine=pd.read_csv('wine.csv')
wine=wine[['country', 'points', 'price', 'region_1', 'variety']]
wine=wine.dropna()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in ['country', 'region_1', 'variety']:
    print(i, end=" ")
    wine[str(i)+'enc']=label_encoder.fit_transform(wine[i])
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(le_name_mapping)
wine.corr()
wine[['price','countryenc', 'region_1enc', 'varietyenc']].hist(figsize=(20,10))
X=wine.drop(columns=['country', 'region_1', 'variety', 'points'])
Y=wine['points']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
PRTest=pd.DataFrame()
PR = LinearRegression()
for i in range(1, 7):
    pf= PolynomialFeatures(degree=i)
    X_train_poly = pf.fit_transform(X_train)
    X_test_poly=pf.fit_transform(X_test)
    PR.fit(X_train_poly, Y_train)
    y_pred_test = PR.predict(X_test_poly)
    y_pred_train=PR.predict(X_train_poly)
    PRTest.at[i, 'TrainAcc']=r2_score(Y_train, y_pred_train)
    PRTest.at[i, 'TestAcc']=r2_score(Y_test, y_pred_test)
plt.plot(PRTest.index, PRTest.TrainAcc, label='Train Acc')
plt.plot(PRTest.index, PRTest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.grid()
pf= PolynomialFeatures(degree=5)
X_train_poly = pf.fit_transform(X_train)
X_test_poly=pf.fit_transform(X_test)
PR.fit(X_train_poly, Y_train)
y_pred_test = PR.predict(X_test_poly)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
KNtest=pd.DataFrame()
for i in range(1, 41):
    KNR = KNeighborsRegressor(n_neighbors=i)
    KNR.fit(X_train, Y_train)
    y_train_pred=KNR.predict(X_train)
    y_test_pred=KNR.predict(X_test)
    KNtest.at[i, 'TrainAcc']=r2_score(Y_train, y_train_pred)
    KNtest.at[i, 'TestAcc']=r2_score(Y_test, y_test_pred)
plt.plot(KNtest.index, KNtest.TrainAcc, label='Train Acc')
plt.plot(KNtest.index, KNtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.grid()
KNR = KNeighborsRegressor(n_neighbors=40)

KNR.fit(X_train, Y_train)
y_pred=KNR.predict(X_test)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
RFtest=pd.DataFrame()
for i in range(1, 21):
    RFR = RandomForestRegressor(max_depth=i, random_state=0)
    RFR.fit(X_train, Y_train)
    y_train_pred=RFR.predict(X_train)
    y_test_pred=RFR.predict(X_test)
    RFtest.at[i, 'TrainAcc']=r2_score(Y_train, y_train_pred)
    RFtest.at[i, 'TestAcc']=r2_score(Y_test, y_test_pred)
plt.plot(RFtest.index, RFtest.TrainAcc, label='Train Acc')
plt.plot(RFtest.index, RFtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.grid()
RFR = RandomForestRegressor(max_depth=15, random_state=0)

RFR.fit(X_train, Y_train)
y_pred=RFR.predict(X_test)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
NNtest=pd.DataFrame()
for i in range(1, 16):
    temp=tuple([4 for j in range(i)])
    NNR = MLPRegressor(hidden_layer_sizes=temp, random_state=0)
    NNR.fit(X_train, Y_train)
    y_train_pred=NNR.predict(X_train)
    y_test_pred=NNR.predict(X_test)
    NNtest.at[i, 'TrainAcc']=r2_score(Y_train, y_train_pred)
    NNtest.at[i, 'TestAcc']=r2_score(Y_test, y_test_pred)
    print(i, end=' ')
plt.plot(NNtest.index, NNtest.TrainAcc, label='Train Acc')
plt.plot(NNtest.index, NNtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Number of layers')
plt.ylabel('Accuracy')
plt.grid()
temp=tuple([4 for j in range(9)])
NNR = MLPRegressor(hidden_layer_sizes=temp, random_state=0)
NNR.fit(X_train, Y_train)
y_pred=NNR.predict(X_test)
print('R2=', r2_score(Y_test, y_pred))
print('MSE=', mean_squared_error(Y_test, y_pred))
cvscoresRFC=cross_val_score(KNR, X, Y, cv=5)
cvscoresDTC=cross_val_score(RFR, X, Y, cv=5)
cvscoresSVM=cross_val_score(NNR, X, Y, cv=5)

fig, ax=plt.subplots(1,3, figsize=(20,5))
ax[0].boxplot(cvscoresRFC)
ax[1].boxplot(cvscoresDTC)
ax[2].boxplot(cvscoresSVM)

ax[0].set_ylabel('Test Accuracy')
ax[1].set_ylabel('Test Accuracy')
ax[2].set_ylabel('Test Accuracy')
ax[0].set_xlabel('KNN')
ax[1].set_xlabel('RF')
ax[2].set_xlabel('NN')
cross_val_score(RFR, X, Y, cv=10)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40, random_state=42)
RFR.fit(X_train, Y_train)
y_train_pred=RFR.predict(X_train)
y_test_pred=RFR.predict(X_test)
print(r2_score(Y_train, y_train_pred))
print(r2_score(Y_test, y_test_pred))
sample=X.sample(frac=1).head(10)
sample['prediction']=RFR.predict(sample)
sample
