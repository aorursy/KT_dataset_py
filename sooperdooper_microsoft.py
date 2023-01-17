# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv(r"../input/train.csv",nrows=1000000,index_col='MachineIdentifier')
test_data = pd.read_csv(r"../input/test.csv",index_col='MachineIdentifier')
set(test_data.columns)-set(train_data.columns)
train_data = train_data.dropna(axis=1)
test_data = test_data.dropna(axis=1)
import seaborn as sns
plt.figure(figsize=(15,15))
sns.heatmap(train_data.corr().abs(),cmap='Dark2')
plt.show()
len(train_data.columns)
p = pd.DataFrame()
for i in train_data.columns:
    if train_data[i].dtype == 'object':
        train_data[i] = train_data[i].astype('category')
        p[i+'_cat'] = train_data[i].cat.codes
    else:
        p[i] = train_data[i]
        
T = pd.DataFrame()
for j in test_data.columns:
    if test_data[j].dtype == 'object':
        test_data[j] = test_data[j].astype('category')
        T[j+'_cat'] = test_data[j].cat.codes
    else:
        T[j] = test_data[j]
# plt.figure(figsize=(25,25))
p.hist(figsize=(25,25))
plt.show()
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
X = p.drop(['HasDetections','Census_GenuineStateName_cat', 'Census_OSEdition_cat'],axis=1).values
y = p['HasDetections'].values
model.add(layer=Dense(units=X.shape[1]+10,input_dim=X.shape[1],activation='relu'))
model.add(layer=Dense(units=X.shape[1]-10,activation='relu'))
model.add(layer=Dense(units=X.shape[1]-20,activation='relu'))
model.add(layer=Dense(units=1,activation='relu'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
model_mlp = MLPClassifier()
model_Kn = KNeighborsClassifier()
model_svc = SVC()
model_GPC = GaussianProcessClassifier()
model_DC = DecisionTreeClassifier()
model_RF = RandomForestClassifier()
model_ABC = AdaBoostClassifier()
# res_ABC = model_ABC.fit(X,y)
# res_mlp = model_mlp.fit(X,y)
# res_Kn = model_Kn.fit(X,y)
# res_GPC = model_GPC.fit(X,y)
# res_DC = model_DC.fit(X,y)

# res_svc = model_svc.fit(X,y)
# res_RF = model_RF.fit(X,y)
# pre_RF = res_RF.predict(X)

# tn, fp, fn, tp = confusion_matrix(y, pre_RF).ravel()

# plt.figure(figsize=(15,7))
# plt.style.use('Solarize_Light2')
# plt.plot(pre_RF[:100],c='palevioletred',linewidth=3)
# plt.plot(y[:100],c='slategray',linewidth=3,alpha=0.6)
# plt.title('Actual Vs Predicted 100 sample RandomForestClassifier',fontsize=20)


# plt.annotate('Accuracy = '+str(round((tp+tn)/(tn+tp+fn+fp),4))+' \nSensitivity = '+str(round(tp/(tp+fn),4))+
#              ' \nError = '+str(round((fp+fn)/(tn+tp+fn+fp),4)),
#              (20,0.6),fontsize=20,bbox=dict(boxstyle="round", fc="cadetblue"),alpha=0.5)

# plt.show()
# res_DC = model_DC.fit(X,y)
# pre_DC = res_DC.predict(X)

# tn, fp, fn, tp = confusion_matrix(y, pre_DC).ravel()

# plt.figure(figsize=(15,7))
# plt.style.use('Solarize_Light2')
# plt.plot(pre_DC[:100],c='palevioletred',linewidth=3)
# plt.plot(y[:100],c='slategray',linewidth=3,alpha=0.6)
# plt.title('Actual Vs Predicted 100 sample DecisionTreeClassifier',fontsize=20)


# plt.annotate('Accuracy = '+str(round((tp+tn)/(tn+tp+fn+fp),4))+' \nSensitivity = '+str(round(tp/(tp+fn),4))+
#              ' \nError = '+str(round((fp+fn)/(tn+tp+fn+fp),4)),
#              (20,0.6),fontsize=20,bbox=dict(boxstyle="round", fc="cadetblue"),alpha=0.5)

# plt.show()
# 2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))
# tp/(tp+fn) , tp/(tp+fp)
# res_Kn = model_Kn.fit(X,y)
# pre_Kn = res_Kn.predict(X)

# tn, fp, fn, tp = confusion_matrix(y, pre_Kn).ravel()

# plt.figure(figsize=(15,7))
# plt.style.use('Solarize_Light2')
# plt.plot(pre_Kn[:100],c='palevioletred',linewidth=3)
# plt.plot(y[:100],c='slategray',linewidth=3,alpha=0.6)
# plt.title('Actual Vs Predicted 100 sample Kn',fontsize=20)


# plt.annotate('Accuracy = '+str(round((tp+tn)/(tn+tp+fn+fp),4))+' \nSensitivity = '+str(round(tp/(tp+fn),4))+
#              ' \nError = '+str(round((fp+fn)/(tn+tp+fn+fp),4)),
#              (20,0.6),fontsize=20,bbox=dict(boxstyle="round", fc="cadetblue"),alpha=0.5)

# plt.show()
# c = res_DC.predict_proba(T)*100
# res_RF.predict_proba(X)
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X,y)
pre = xg_reg.predict(X)
np.sqrt(mean_squared_error(y, pre))
import matplotlib.pyplot as plt
plt.figure(figsize=(250,250))
xgb.plot_tree(xg_reg,num_trees=0)
# plt.rcParams['figure.figsize'] = [50, 50]
plt.show()

T['HasDetections'] = xg_reg.predict(T.values)
T[['HasDetections']]
T[['HasDetections']].to_csv(r'submission.csv')

