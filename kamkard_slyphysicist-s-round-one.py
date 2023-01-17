# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib notebook

#Needed for feature engineering
from itertools import product

#Model selection with multioutput regression
from sklearn import linear_model
#from sklearn.multioutput import MultiOutputRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
print(os.listdir("../input"))
print("Number of files = ",len(os.listdir("../input")))
fileList = os.listdir("../input")
fileList = sorted(fileList)
# Any results you write to the current directory are saved as output.
#Initialize dataframe
data=pd.DataFrame()

#Create DataFrame with all times and id ti.
for i in range(len(fileList)):
    pstr = ("../input/"+fileList[i])
    df1 = pd.read_csv(pstr)
    df1['t'] = i*100
    data = data.append(df1)

data=data.reset_index(drop=True)
data.describe()
# data.dtypes
#Creates Features and Target data

#take out last time value for features since don't have next value to predict
featureIndex = data.groupby(['id'], sort=False)['t'].idxmax()
featureData = data.drop(featureIndex.tolist())
featureData = featureData.sort_values(by=['id','t'])
featureData.reset_index(drop=True, inplace=True)

#targets take out first time value since no feature to predict to
targetFull = data[data.t != 0]
targetFull = targetFull.sort_values(by=['id','t'])
targetData = targetFull[['x','y','z']]
targetData.reset_index(drop=True, inplace=True)
targetPositions = targetData.rename(columns={'x':'yX','y':'yY','z':'yZ'})

#combine for easy split later on
allData = pd.concat([featureData,targetPositions],axis=1)
allData.describe()
#Engineer Feature of position bins

xBin = np.quantile(data.x,[0,.3333,.6666,1])
yBin = np.quantile(data.y,[0,.3333,.6666,1])
zBin = np.quantile(data.z,[0,.3333,.6666,1])

#Add columns for each 3D space bin 
#Set to 3 bins [-1,0,1]
label_col = list(product(range(-1,2),repeat=3))
label_string = [str(word) for word in label_col]

#Create Featured Engineered DataFrame
feature_eng = pd.concat([featureData,pd.DataFrame(0,index = np.arange(featureData.shape[0]), columns=label_string)],axis=1, sort=False)

#Groupby to update position of other stars at each time interval
grouped = feature_eng.groupby('t', sort=True)

for name, group in grouped:
    xBinned = pd.cut(group.x,xBin, labels = ['-1','0','1'], include_lowest=True) 
    yBinned = pd.cut(group.y,yBin, labels = ['-1','0','1'], include_lowest=True)
    zBinned = pd.cut(group.z,zBin, labels = ['-1','0','1'], include_lowest=True)
    
    totalBinned = pd.DataFrame([xBinned,yBinned,zBinned])
    testBinned = totalBinned.apply(lambda x: '(' + ','.join(x) + ')',axis=0)
    
    star_clusters_binned = pd.DataFrame(testBinned.value_counts()).T
    star_clusters_binned.sort_index(axis=1,inplace=True)
    total_star_clusters_binned = pd.concat([star_clusters_binned]*testBinned.shape[0],ignore_index=True)
    
    for index, row in total_star_clusters_binned.iterrows():
        row.loc[testBinned.iloc[index]] -= 1

    total_star_clusters_binned['Index'] = group.index.values
    total_star_clusters_binned.set_index('Index',inplace = True)
    group = pd.concat([group.iloc[:,0:9],total_star_clusters_binned],axis=1)
    feature_eng[feature_eng['t']==name] = group.values
    
#     group.iloc[:,9:] = total_star_clusters_binned.values
#     feature_eng[feature_eng['t']==name] = group
    
feature_eng.describe()
#Target data in buckets to use for classification

yxBinned = pd.cut(allData.yX,xBin, labels = ['-1','0','1'], include_lowest=True)
yyBinned = pd.cut(allData.yY,yBin, labels = ['-1','0','1'], include_lowest=True)
yzBinned = pd.cut(allData.yZ,zBin, labels = ['-1','0','1'], include_lowest=True)

totalYBinned = pd.DataFrame([yxBinned,yyBinned,yzBinned])
testYBinned = totalYBinned.apply(lambda x: '(' + ','.join(x) + ')',axis=0)

classData = pd.concat([feature_eng,testYBinned.rename('TargetBin')],axis = 1)
#Goal: save data in 2 formats, 1 for regression (regData) & 1 for classification (classData)

feature_eng.to_csv('featureData.csv', index = False)
targetPositions.to_csv('regTarget.csv', index = False)
classData.to_csv('classTarget.csv',index = False)

feature_eng.shape,targetPositions.shape,classData.shape
# xAll = allData[['x','y','z','vx','vy','vz']]
# yAll = allData[['yX','yY','yZ']]
# train_size = list(range(1,int(allData.shape[0]/5*4),int(allData.shape[0]/10)))
# X_train, X_test, y_train, y_test = train_test_split(xAll, yAll, test_size=0.3, shuffle = True)

# reg = linear_model.LinearRegression(normalize=True)
# reg.fit(X_train,y_train)

# y_pred = pd.DataFrame(reg.predict(X_test),columns=['yX','yY','yZ'])

# y_indexes = allData.loc[allData['id']==100].index.values
# y_100 = yAll.iloc[y_indexes]
# y_pred_100 = y_pred.iloc[y_indexes]

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(y_100.yX,y_100.yY,y_100.yZ, c='red')
# ax.plot(y_pred_100.yX,y_pred_100.yY,y_pred_100.yZ, c='blue')
# plt.show()
# from sklearn.svm import SVR
# from sklearn.multioutput import MultiOutputRegressor

# regSvm = SVR()
# regr= MultiOutputRegressor(regSvm)

# regr.fit(X_train,y_train)

# y_predSvm = regr.predict(X_test)

# y_predSvmPD = pd.DataFrame(y_predSvm,columns=['yX','yY','yZ'])

# y_indexes = allData.loc[allData['id']==100].index.values
# y_100 = yAll.iloc[y_indexes]
# y_pred_100 = y_predSvmPD.iloc[y_indexes]

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(y_100.yX,y_100.yY,y_100.yZ, c='red')
# ax.plot(y_pred_100.yX,y_pred_100.yY,y_pred_100.yZ, c='blue')
# plt.show()
# ax1 = plt.axes(projection='3d')
# ax1.scatter3D(X_test.x,X_test.y,X_test.z)
# ax1.plot3D(y_predSvmPD.yX,y_predSvmPD.yY,y_predSvmPD.yZ)
# from sklearn.model_selection import learning_curve

# train_sizes, train_scores, valid_scores = learning_curve(SVR(), x_all, yZ)

# plt.figure()

# plt.xlabel("Training examples")
# plt.ylabel("Score")
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(valid_scores, axis=1)
# test_scores_std = np.std(valid_scores, axis=1)
# plt.grid()

# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#   train_scores_mean + train_scores_std, alpha=0.1,
#   color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#   test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#   label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#   label="Cross-validation score")

# plt.legend(loc="best")
