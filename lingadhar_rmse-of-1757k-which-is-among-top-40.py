%matplotlib inline

# Data wrapper libraries
import pandas as pd
import numpy as np
from collections import Counter

#Data Visualization Libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.markers import MarkerStyle
import seaborn as sns

#Date time Libraries
import time
import datetime
TFI_data_train = pd.read_csv("C:/Users/IBM_ADMIN/Desktop/appliedai/TFI_Restaurant/train.csv")
TFI_data_test = pd.read_csv("C:/Users/IBM_ADMIN/Desktop/appliedai/TFI_Restaurant/test.csv")
print("size of train data",TFI_data_train.shape)
print("size of test data",TFI_data_test.shape)
TFI_data_train.info()
TFI_data_train.columns
TFI_data_train["Citygroup"]=TFI_data_train["City Group"]
TFI_data_train.drop("City Group",axis=1)
TFI_data_train=TFI_data_train[['Id', 'Open Date', 'City', 'Citygroup', 'Type', 'P1', 'P2', 'P3', 'P4',
       'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
       'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25',
       'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35',
       'P36', 'P37', 'revenue']]
TFI_data_test["Citygroup"]=TFI_data_test["City Group"]
TFI_data_test.drop("City Group",axis=1)
TFI_data_test=TFI_data_test[['Id', 'Open Date', 'City', 'Citygroup', 'Type', 'P1', 'P2', 'P3', 'P4',
       'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
       'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25',
       'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35',
       'P36', 'P37']]
plt.figure(figsize=(18,5))
sns.set_style("whitegrid")
(TFI_data_train.City.value_counts()/len(TFI_data_train)).plot(title="Distribution of City in Train",kind='bar',color='green')
plt.show()
plt.figure(figsize=(18,5))
(TFI_data_test.City.value_counts()/len(TFI_data_test)).plot(title="Distribution of City in Test",kind='bar',color='red')
plt.show()
cnotintest=[]
cnotintrain=[]
a=TFI_data_train.City.unique()
b=TFI_data_test.City.unique()
for i in a:
    if not i in b:
        cnotintest.append(i)

for i in b:
    if not i in a:
        cnotintrain.append(i)
print("Cities in Test but not in Train are",len(cnotintrain))
print(cnotintrain)
print("Cities in Train but not in Test are",len(cnotintest))
print(cnotintest)
TFI_data_train["Citygroup"].where(TFI_data_train["City"].isin(cnotintest)).unique()
TFI_data_test["Citygroup"].where(TFI_data_test["City"].isin(cnotintrain)).unique()
TFI_data_train["Type"].where(TFI_data_train["City"].isin(cnotintest)).unique()
TFI_data_test["Type"].where(TFI_data_test["City"].isin(cnotintrain)).unique()
a=TFI_data_test.where(TFI_data_test["City"].isin(cnotintrain))
len(a[(a["Type"]=='MB') | (a["Type"]=='DT')])
TFI_data_test.loc[TFI_data_test.City.isin(cnotintrain), 'City'] = 'UNK'
TFI_data_test.City.value_counts()
TFI_data_train.loc[TFI_data_train.City.isin(cnotintest), 'City'] = 'UNK'
TFI_data_train.City.value_counts()
plt.figure(figsize=(12,5))
sns.set_style("whitegrid")
(TFI_data_train.Citygroup.value_counts()/len(TFI_data_train)).plot(title="Distribution of City Group in Train",kind='bar',color='green')
plt.show()
plt.figure(figsize=(12,5))
(TFI_data_test.Citygroup.value_counts()/len(TFI_data_test)).plot(title="Distribution of City Group in Test",kind='bar',color='red')
plt.show()
plt.figure(figsize=(12,5))
sns.set_style("whitegrid")
(TFI_data_train.Type.value_counts()/len(TFI_data_train)).plot(title="Distribution of restaurant type in Train",kind='bar',color='green')
plt.show()
plt.figure(figsize=(12,5))
(TFI_data_test.Type.value_counts()/len(TFI_data_test)).plot(title="Distribution of restaurant type in Test",kind='bar',color='red')
plt.show()
TFI_data_test["Open Date"]=pd.to_datetime(TFI_data_test["Open Date"])
TFI_data_test["DayssinceInception"]=(datetime.date.today()-TFI_data_test["Open Date"]).dt.days
del TFI_data_test["Open Date"]
TFI_data_train["Open Date"]=pd.to_datetime(TFI_data_train["Open Date"])
TFI_data_train["DayssinceInception"]=(datetime.date.today()-TFI_data_train["Open Date"]).dt.days
del TFI_data_train["Open Date"]
TFI_data_train.head(3)
plt.figure(figsize=(12,5))
sns.set_style("whitegrid")
plt.scatter(x=TFI_data_train.DayssinceInception,y=TFI_data_train.revenue,c='r')
plt.show()

plt.figure(figsize=(12,5))
sns.set_style("whitegrid")
plt.scatter(x=np.log(TFI_data_train.DayssinceInception),y=TFI_data_train.revenue,c='g')
plt.show()
plt.figure(figsize=(10,6))
f, (ax1, ax2) = plt.subplots(1,2)
sns.boxplot(TFI_data_train.DayssinceInception,ax=ax1,orient='v',color='r')
ax1.set_title("DayssinceInception-Train")
sns.boxplot(TFI_data_test.DayssinceInception,ax=ax2,orient='v',color='g')
ax2.set_title("DayssinceInception-Test")
f.tight_layout()
np.log(TFI_data_test.DayssinceInception).describe()
np.log(TFI_data_train.DayssinceInception).describe()
sns.distplot(np.log(TFI_data_test.DayssinceInception),label='Test')
sns.distplot(np.log(TFI_data_train.DayssinceInception),label='Train')
TFI_data_train["DayssinceInception"]=np.log(TFI_data_train.DayssinceInception)
TFI_data_test["DayssinceInception"]=np.log(TFI_data_test.DayssinceInception)
a=(TFI_data_train==0).astype(int).sum(axis=0)
a
b=(TFI_data_test==0).astype(int).sum(axis=0)
b
df1 = pd.DataFrame(data=a.index, columns=['cols'])
df2 = pd.DataFrame(data=a.values/len(TFI_data_train), columns=['cnt_trn'])
df_trn = pd.merge(df1, df2, left_index=True, right_index=True)
df11 = pd.DataFrame(data=b.index, columns=['cols'])
df21 = pd.DataFrame(data=b.values/len(TFI_data_test), columns=['cnt_tst'])
df_tst = pd.merge(df11, df21, left_index=True, right_index=True)
df_zeros = pd.merge(df_trn, df_tst, left_index=True, right_index=True)
df_zeros.drop("cols_y",axis=1)
c = np.cumsum(TFI_data_train.P36.values/len(TFI_data_train))
sns.set_style("whitegrid")
plt.plot(c,label='Cumulative distribution of P36 in train')
plt.grid()
plt.legend()
plt.show()

c = np.cumsum(TFI_data_test.P36.values/len(TFI_data_test))
sns.set_style("whitegrid")
plt.plot(c,label='Cumulative distribution of P36 in test')
plt.grid()
plt.legend()
plt.show()
TFI_data_train=TFI_data_train[TFI_data_train.Id!=16]
# Removal of the only outlier
TFI_data_train["revenue"]=np.log(TFI_data_train.revenue)
#Since revenue is the approximate lognormal distribution and can be checked from the below plot
plt.figure(figsize=(10,6))
f, (ax1, ax2) = plt.subplots(2)
sns.distplot(TFI_data_train["revenue"],ax=ax1)
ax1.set_title("revenue")
sns.distplot(np.log(TFI_data_train["revenue"]),ax=ax2)
ax2.set_title("log of revenue")
f.tight_layout()
TFI_data_train.revenue[0]
import math
math.e**TFI_data_train.revenue[0]
TFI_data_train.columns
TFI_data_train_fin = TFI_data_train[['Citygroup', 'Type','DayssinceInception','P1', 'P2','P4', 'P5', 'P6',
       'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16',
       'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26',
       'P27', 'P28', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36',
       'P37', 'revenue']]
sns.barplot(y=math.e ** TFI_data_train["revenue"],x=TFI_data_train_fin["Citygroup"])
TFI_data_test["Citygroup"]=TFI_data_test.Citygroup.replace(to_replace="Big Cities",value="1")
TFI_data_test["Citygroup"]=TFI_data_test.Citygroup.replace(to_replace="Other",value="0")
TFI_data_test["Citygroup"]=pd.to_numeric(TFI_data_test["Citygroup"])
TFI_data_train_fin.head(2)
sns.barplot(y=math.e ** TFI_data_train["revenue"],x=TFI_data_train_fin["Type"])
sns.countplot(TFI_data_train.Type)
sns.countplot(TFI_data_test.Type)
TFI_data_train_fin = pd.get_dummies(TFI_data_train_fin,columns=['Type'])
TFI_data_train_fin.head(3)
TFI_data_test = pd.get_dummies(TFI_data_test,columns=['Type'])
TFI_data_test.head(2)
TFI_data_test1=TFI_data_test.drop(["City","Type_DT","Type_MB"],axis=1)
TFI_data_test1=TFI_data_test1.drop(["Type_MB"],axis=1)
TFI_data_train_fin=TFI_data_train_fin.drop(["Type_DT"],axis=1)
TFI_data_train_fin.columns
TFI_data_test1.columns
TFI_data_train_fin = TFI_data_train_fin[['Citygroup', 'DayssinceInception','Type_FC', 'Type_IL','P1', 'P2', 'P4', 'P5', 'P6', 'P7',
       'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17',
       'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27',
       'P28', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37']]
train_rev = TFI_data_train.revenue
print(len(train_rev))
print(len(TFI_data_train_fin))
TFI_data_test1=TFI_data_test1[['Citygroup', 'DayssinceInception','Type_FC', 'Type_IL','P1', 'P2', 'P4', 'P5', 'P6', 'P7',
       'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17',
       'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27',
       'P28', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37']]
y=train_rev.values
x=TFI_data_train_fin.values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=4)
#with all the features
import statsmodels.api as sm

# Note the difference in argument order
model = sm.OLS(y_train, x_train).fit()
y_trn_pred = math.e ** model.predict(x_train) 
y_test_pred = math.e ** model.predict(x_test) # make the predictions by the model

# Print out the statistics
model.summary()
print("Root mean squared error achieved from Linear Model:",np.sqrt(mean_squared_error(math.e **y_test, y_test_pred)))
from sklearn.ensemble import RandomForestRegressor
cls = RandomForestRegressor(n_estimators=1250)
cls.fit(x_train, y_train)
y_pred_trn_rf = cls.predict(x_train)
y_pred_test_rf = math.e ** cls.predict(x_test)
cls.score(x_train, y_train)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(cls,x_train, y_train, cv=5)
scores
print("Root mean squared error achieved from RF:",np.sqrt(mean_squared_error(math.e **y_test, y_pred_test_rf)))
from sklearn.grid_search import GridSearchCV
# Ridge model
model_grid = [{'normalize': [True, False], 'alpha': np.logspace(0,10)}]
ridge_clf = Ridge()

# Use a grid search and leave-one-out CV on the train set to find the best regularization parameter to use.
grid = GridSearchCV(ridge_clf, model_grid, cv=10, scoring='mean_squared_error')
grid.fit(x_train,y_train)
print("Root mean squared error achieved from Ridge:",np.sqrt(mean_squared_error(math.e **y_test, y_pred_ridge)))
print("Root mean squared error achieved from Linear Model:",np.sqrt(mean_squared_error(math.e **y_test, y_test_pred)))
print("Root mean squared error achieved from RF:",np.sqrt(mean_squared_error(math.e **y_test, y_pred_test_rf)))
print("Root mean squared error achieved from Ridge:",np.sqrt(mean_squared_error(math.e **y_test, y_pred_ridge)))
x_tst = TFI_data_test1.values
type(x_train)
final_pred = math.e ** cls.predict(x_tst)
submission = pd.DataFrame({
        "Id": TFI_data_test["Id"],
        "Prediction": final_pred
    })
submission.to_csv('randomres.csv',header=True, index=False)