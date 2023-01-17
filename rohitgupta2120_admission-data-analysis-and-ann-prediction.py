import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library

import matplotlib.pyplot as plt # mathematical plotting library
#read the dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename));

        

admission_df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv',index_col="Serial No.")

admission_df
admission_df.info()
admission_df.nunique()
admission_df.columns
admission_df.describe()
df_univ = admission_df.groupby(by  = 'University Rating').mean()

df_univ
matrix = admission_df.corr()

plt.figure(figsize=(8,6))

#plot heat map

g=sns.heatmap(matrix,annot=True,cmap="RdYlGn_r")

plt.title("Correltion Matrix");
#correlatoin pair plots

sns.pairplot(admission_df.drop(columns=["LOR ","SOP","Research"]));
sns.jointplot(x="CGPA",y="Chance of Admit ",data=admission_df);
num = [304.91176471, 309.13492063, 315.0308642 , 323.3047619 ,

       327.89041096]
plt.figure(figsize=(10,8))

sns.barplot(y="GRE Score",x="University Rating",data=admission_df)

plt.ylim([300,330])

li = 0.1

for i in range(5):

    plt.text(li , num[i]+0.5, np.round(num[i],2) )

    li+=1

plt.title("Expected GRE score vs University Rating");
sns.boxplot(x="LOR ",y="Chance of Admit ",data=admission_df)

plt.title("Chance of admission depending on Letter of Recommendation");
sns.boxplot(x="SOP",y="Chance of Admit ",data=admission_df)

plt.title("Chance of admission depending on Letter of Recommendation");
plt.figure(figsize=(12,8))

sns.lineplot(x="SOP",y="Chance of Admit ",data=admission_df, label="SOP")

sns.lineplot(x="LOR ",y="Chance of Admit ",data=admission_df, label="LOR")

sns.lineplot(x="University Rating",y="Chance of Admit ",data=admission_df, label="Research")

plt.legend()

plt.title("features affecting admission on Scale of 0-5")

plt.xlabel("Features Scale")

plt.show()
#creating histograms

admission_df.hist(bins = 30, figsize=(10,10), color= 'orange');
#create the dependent and independent dataset

#splitting training and test set 

#test set is last 100 observations



X_train=admission_df.iloc[0:400,:-1].values

y_train= admission_df.iloc[0:400,-1].values

X_test=admission_df.iloc[400:500,:-1].values

y_test= admission_df.iloc[400:500,-1].values
#feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
GRE=[]

TOEFL=[]



for i in range(X_train.shape[0]):

    GRE.append(X_train[i][1])

    TOEFL.append(X_train[i][0])
sns.kdeplot(GRE, shade=True, label="GRE")

sns.kdeplot(TOEFL, shade=True, label="TOEFL")

plt.title("Density chart of GRE vs TOEFL")
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)
y_lin_pred = linear_reg.predict(X_test)
from sklearn.tree import DecisionTreeRegressor

Decision_regressor = DecisionTreeRegressor(random_state = 0)

Decision_regressor.fit(X_train, y_train)
y_decision_pred = Decision_regressor.predict(X_test)
from sklearn.ensemble import RandomForestRegressor

Forest_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

Forest_regressor.fit(X_train, y_train)
y_forest_pred = Forest_regressor.predict(X_test)
#Libraries to train Neural network

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.optimizers import Adam
ANN_model = keras.Sequential()

ANN_model.add(Dense(50, input_dim=7))

ANN_model.add(Activation('relu'))



ANN_model.add(Dense(150))

ANN_model.add(Activation('relu'))

ANN_model.add(Dropout(0.5))



ANN_model.add(Dense(150))

ANN_model.add(Activation('relu'))

ANN_model.add(Dropout(0.5))



ANN_model.add(Dense(50))

ANN_model.add(Activation('linear'))

ANN_model.add(Dense(1))





ANN_model.compile(loss = 'mse', optimizer = 'adam')

ANN_model.summary()

#Using Adam optimizer

ANN_model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
epochs_hist = ANN_model.fit(X_train, y_train, epochs = 100, batch_size = 20);
y_ann_pred = ANN_model.predict(X_test)

result = ANN_model.evaluate(X_test, y_test)
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])

plt.title('Model Loss Progreess During Training')

plt.xlabel('Epoch')

plt.ylabel('Training Loss')

plt.legend(['Training Loss'])
from sklearn.metrics import accuracy_score

acc_lin = linear_reg.score(X_test, y_test)

print("Liner Accuracy : {}".format(acc_lin))
acc_decision = Decision_regressor.score(X_test, y_test)

print("Decision Accuracy : {}".format(acc_decision))
acc_forest = Forest_regressor.score(X_test, y_test)

print("Forest Accuracy : {}".format(acc_forest))
acc_ANN = 1 - ANN_model.evaluate(X_test, y_test)

print("ANN Accuracy : {}".format(acc_ANN))
plt.figure(figsize= (14,10))

#y_test on x axis

#y_pred on y axis

plt.subplot(221)

plt.plot(y_test, y_lin_pred,'o', color = 'b')

plt.title('Linear plot')

plt.ylabel("linear predict")

plt.xlabel("test cases")



plt.subplot(222)

plt.plot(y_test, y_decision_pred, '^', color = 'r')

plt.title('Decision plot')



plt.subplot(223)

plt.plot(y_test, y_forest_pred, 'v', color = 'g')

plt.title('Forest plot')



plt.subplot(224)

plt.plot(y_test, y_decision_pred, '*', color = 'aqua')

plt.title('ANN plot')

k = X_test.shape[1]

n= len(X_test)

k,n 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from math import sqrt



r2 = r2_score(y_test, y_lin_pred)

adj_r2 = 1- (1-r2)*(n-1)/(n-k-1)

MAE = mean_absolute_error(y_test, y_lin_pred)

MSE = mean_squared_error(y_test, y_lin_pred)

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_lin_pred)),'.3f'))



print('R2 - ', r2, '\nAdjusted R2 - ', adj_r2, '\nMAE - ', MAE, '\nMSE - ', MSE, '\nRMSE - ', RMSE)
from statsmodels.api import OLS

summ=OLS(y_train,X_train).fit()

summ.summary()