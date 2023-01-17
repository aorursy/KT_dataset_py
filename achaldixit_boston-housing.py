import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import r2_score
import sklearn
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
%matplotlib inline


boston_df = pd.DataFrame(load_boston().data,columns = load_boston().feature_names)

boston = sklearn.datasets.load_boston(return_X_y=False)
dataf = pd.DataFrame(boston.data)

y = load_boston().target
x = load_boston().data

boston_df.head(5)
boston_df["Price"] = y
boston_df.head()
print(boston.DESCR)
boston_df.describe()
fig,ax = plt.subplots(figsize= (7,5))
ax.scatter(boston_df.CRIM,boston_df.Price,s=3)
boston_df[boston_df.CRIM >30]
for i in range(len(x)-9):
    if x[i][0] > 30:
        x = np.delete(x,i,0)
        y = np.delete(y,i,0)
#outliers deleted
for i in range(len(x)-9):
    if x[i][0] > 30:
        print("True")
boston_df.drop(index=[380,398,404,405,410,414,418,427],inplace=True)
fig,ax = plt.subplots(figsize= (7,5))
ax.scatter(boston_df.Price,boston_df.ZN,s=3)
len(boston_df[boston_df.ZN > 80])
fig,ax = plt.subplots(figsize= (7,5))
ax.scatter(boston_df.Price,boston_df.B,s=3)
len(boston_df[boston_df.B < 150])
fig,ax = plt.subplots(figsize= (7,5))
ax.scatter(boston_df.Price,boston_df.INDUS,s=3)
boston_df[boston_df.INDUS > 27.5]
x = np.delete(x,489,0)
y = np.delete(y,490,0)
boston_df.drop(index=[489,490],inplace=True)
boston_df.corr('pearson')
kendall = abs(boston_df.corr(method="kendall").Price).to_dict()
pearson = abs(boston_df.corr(method="pearson").Price).to_dict()
spearman = abs(boston_df.corr(method="spearman").Price).to_dict()

print("\t\tPearson\t\tKendall\t\tSpearman")
for p,k,s in zip(pearson.items(),kendall.items(),spearman.items()):
    print("\t{}\t{:.2f}\t\t{:.2f}\t\t{:.2f}".format(p[0],p[1],k[1],s[1]))
kendall = abs(boston_df.corr(method="kendall").Price).to_dict()
pearson = abs(boston_df.corr(method="pearson").Price).to_dict()
spearman = abs(boston_df.corr(method="spearman").Price).to_dict()

print("\t\tPearson\t\tKendall\t\tSpearman")
for p,k,s in zip(pearson.items(),kendall.items(),spearman.items()):
    if p[1] >.50 or s[1] > .50 or k[1] > .50:
        print("\t{}\t{:.2f}\t\t{:.2f}\t\t{:.2f}".format(p[0],p[1],k[1],s[1]))
    
temp = abs(boston_df.corr('pearson').Price).to_dict()
from collections import Counter
var = Counter(temp)
top_features = var.most_common(4)
del top_features[0]
print(top_features)
fig1,ax1 = plt.subplots()
fig1.set_size_inches(12,7.5)
ax1.set_ylabel('Price')
ax1.set_xlabel('Low status Popolation %')
ax1.set_title('Relationship Between Price and Low status Population')
c = boston_df['Price']
plt.scatter(boston_df.LSTAT, boston_df.Price, c=c, cmap = 'copper_r', alpha =0.6)  
cbar = plt.colorbar()
cbar.set_label('Price')
boston_df["LSTAT"].to_frame().boxplot()
boston_df[boston_df.LSTAT > 4]
fig1,ax1 = plt.subplots()
fig1.set_size_inches(12,7.5)
ax1.set_ylabel('Price')
ax1.set_xlabel('Number of Rooms')
ax1.set_title('Relationship Between Price and Number of Rooms')
c = boston_df['Price']
plt.scatter(boston_df.RM, boston_df.Price,c=c, 
            cmap = 'autumn_r', alpha =0.5)  
cbar = plt.colorbar()
cbar.set_label('Price')
fig1,ax1 = plt.subplots()
fig1.set_size_inches(12,7.5)
ax1.set_ylabel('Price')
ax1.set_xlabel('Commercial Businesses')
ax1.set_title('Relationship Between Price and Industires in the town')
c = boston_df['Price']
plt.scatter(boston_df.INDUS, boston_df.Price,c=c, 
            cmap = 'copper_r', alpha =0.5)  
cbar = plt.colorbar()
cbar.set_label('Price')
#Train and Test set split into 80% and 20% respectively
x_train, x_rest, y_train, y_rest = train_test_split(x,y,test_size = .4,random_state =0)

#Rest of the 40% set split into equal parts of Train and Validation set 
x_test, x_val, y_test, y_val = train_test_split(x_train,y_train,test_size = .5,random_state = 0)

#Therefore : Train = 60%, Test = 20% and Validation = 20%
print(len(x_train),len(y_train))
print(len(x_val),len(y_val))
print(len(x_test),len(y_test))
#preprocessing is necessary for SDG Regressor

scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
y_val = y_val.reshape(-1)
clf_sdg = SGDRegressor(max_iter= 5000,eta0=0.0001,learning_rate='constant')
#learning rate is default = 0.01 as constant = eta0

clf_sdg.fit(x_train,y_train)

y_hat_test = clf_sdg.predict(x_test)
y_hat_val = clf_sdg.predict(x_val)
y_hat_train = clf_sdg.predict(x_train)

test_score = clf_sdg.score(x_test,y_test)
train_score = clf_sdg.score(x_train,y_train)
val_score = clf_sdg.score(x_val,y_val)
print("R2-score Train: \t\t%.2f" % r2_score(y_hat_train , y_train) )
print("R2-score Test: \t\t\t%.2f" % r2_score(y_hat_test , y_test) )
print("R2-score Validation: \t\t%.2f" % r2_score(y_hat_val , y_val) )
print("------------------------------------")
print("Train score : \t\t\t%.2f"%train_score)
print("Test score : \t\t\t%.2f"%test_score)
print("Validation Score : \t\t%.2f"%val_score)

print("RMSE test : %3f"%math.sqrt(mean_squared_error(y_test,y_hat_test)))
print("RMSE train : %3f"%math.sqrt(mean_squared_error(y_train,y_hat_train)))
print("RMSE val : %3f"%math.sqrt(mean_squared_error(y_val,y_hat_val)))
print("MAE test : %3f"%(mean_absolute_error(y_test,y_hat_test)))
print("MAE train : %3f"%(mean_absolute_error(y_train,y_hat_train)))
print("MAE val : %3f"%(mean_absolute_error(y_val,y_hat_val)))