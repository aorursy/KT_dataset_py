# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import itertools
from itertools import chain
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score,mean_squared_error
import keras
from keras.layers import Dense
from keras.models import Sequential
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

data = pd.read_csv("../input/kc_house_data.csv")
data.head()
print(data.shape)
## Column years passed after recent built or renovate is added ##

data['year_pass'] = 0
data['year_pass'] = data.yr_renovated-data.yr_built
data.year_pass = pd.Series([x+2018 if x <0 else x for x in data.year_pass])
## Column 'id' and 'zipcode' are withdrawn since I thought that they don't have meaningful implication ## 

data.drop(columns=['date','id','zipcode'],inplace=True)
def plot_distribution(column1,column2, size_bin) :  
    tmp1 = data[column1].head()
    tmp2 = data[column2].head()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['malignant', 'benign']
    colors = ['#FFD700', '#7EC0EE']

    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    
    fig['layout'].update(title = column1)

    py.iplot(fig, filename = 'Density plot')
import random
def dist_plot(column,ran):
    
    color = ['c','orange','lightgrey']
    data[column].plot.hist(color=color[ran],figsize=(10,6),bins=100)
    plt.title(column+" Distribution")
    plt.show()
for i,j in enumerate(data.columns[1:]):
    col_index = int(i) % 3
    dist_plot(j,col_index)
### Showing correlation matrix and heatmap figure

corr_mat = data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_mat)
target_pca = data['price']
data_pca = data.drop('price', axis=1)

target_pca = pd.DataFrame(target_pca)

### normalizing data
X_pca = data_pca.values
X_std = StandardScaler().fit_transform(X_pca)

pca = PCA(svd_solver='full')
pca_std = pca.fit(X_std, target_pca).transform(X_std)

pca_std = pd.DataFrame(pca_std)
pca_std = pca_std.merge(target_pca, left_index = True, right_index = True, how = 'left')
var_pca = pd.DataFrame(pca.explained_variance_ratio_)
var_pca = var_pca.T

#----------SUM AND DROP COMP [7:30]
col_list = list(v for v in chain(pca_std.columns[10:18])) 
var_pca['OTHERS_COMP'] = var_pca[col_list].sum(axis=1)
var_pca.drop(var_pca[col_list],axis=1,inplace=True)
var_pca = var_pca.T
### table of variances explained by each components
var_pca
labels = ['Component1','Component2','Component3','Component4','Component5','Component6','Component7','Component8','Component9','Component10','other Components']
trace = go.Pie( labels=labels, values=var_pca[0],
              opacity=1,
              textfont=dict(size=15)
              )
layout = dict(title="Variances explained by Each Componets: " + "10 Components among 17 are explaining 89.6%")

fig = dict(data=[trace], layout=layout)
py.iplot(fig)
train_data, test_data = train_test_split(data, train_size=0.8, random_state=3)
Y_train = train_data.price 
X_train = train_data.iloc[:,2:]
Y_test  = test_data.price
X_test  = test_data.iloc[:,2:]
Y_train = np.array(Y_train, dtype=pd.Series).reshape(-1,1)
Y_test = np.array(Y_test, dtype=pd.Series).reshape(-1,1)

lm = LinearRegression()
lm.fit(np.array(X_train), np.array(Y_train))
pred = lm.predict(X_test)
print( "Mean Squared Error is : "+ str(np.sqrt(mean_squared_error(Y_test,pred))))
print( "Linear Regression Score is : " + str(lm.score(X_test,Y_test)))
lasso = Lasso()
lasso.fit(np.array(X_train), np.array(Y_train))
pred1 = lasso.predict(X_test)

print( "Mean Squared Error is : "+ str(np.sqrt(mean_squared_error(Y_test,pred1))))
print( "Linear Regression Score is : " + str(lasso.score(X_test,Y_test)))

data.iloc[:,1:].shape
### model fitting using deep learning
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=5)

predictors = X_train
target = Y_train
# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# Compile the model

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])

# Fit the model
hist = model.fit(predictors,target, batch_size=100, epochs=100, validation_split=0.2,verbose=False)

hist.history.keys()
plt.figure(figsize=(10,8))
plt.plot(hist.history['mean_absolute_error'])
plt.plot(hist.history['val_mean_absolute_error'])
plt.xticks(range(0,100,5))
plt.xlabel("epoch",fontsize=15)
#plt.plot(hist.history['val_loss'])
plt.legend(['Train','Test'],loc='upper right',fontsize=13)
plt.show()
print("Mean Absolute Error(MAE) for Training set is : " + str(hist.history['mean_absolute_error'][99]))
print("Mean Absolute Error(MAE) for Validation set is : " + str(hist.history['val_mean_absolute_error'][99]))