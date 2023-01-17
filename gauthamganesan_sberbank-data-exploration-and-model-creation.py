import pandas as pd 

import numpy as np

import math as m

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns 

from scipy import stats

import statsmodels.api as sm





from sklearn.ensemble import GradientBoostingRegressor 

from sklearn.ensemble import RandomForestRegressor 

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



from sklearn.model_selection import GridSearchCV

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

from sklearn.metrics import mean_absolute_error, r2_score , mean_squared_error



#to randomly split data into train and test

from sklearn.model_selection import train_test_split

%matplotlib inline

seed=45
traindf = pd.read_csv(path+"train.csv")

testdf = pd.read_csv(path+"test.csv")



print(traindf.shape)

print(testdf.shape)
sns.distplot(traindf["price_doc"],bins = 100)

print(traindf["price_doc"].describe().apply(lambda x : format(x,'10.0f')))

print("\n\n\nPrice range skewness",stats.skew(traindf["price_doc"]))

plt.figure(figsize=(8,6))

plt.scatter(range(traindf.shape[0]), np.sort(traindf.price_doc.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('price', fontsize=12)

plt.show()
list(np.percentile(traindf["price_doc"], np.arange(0, 100, 2)))
print("NAs in Price Doc column",traindf["price_doc"].isnull().sum(),"\n")

print("Train Data")

print("min date:",traindf["timestamp"].min())

print("max date:",traindf["timestamp"].max())

print("number of nulls",traindf["timestamp"].isna().value_counts())



print("\nTest Data")

print("min date:",testdf["timestamp"].min())

print("max date:",testdf["timestamp"].max())

print("number of nulls",testdf["timestamp"].isna().value_counts())



print("\nTime variable type before conversion: ",traindf['timestamp'].dtype)
traindf["timestamp"]=pd.to_datetime(traindf['timestamp'])

testdf["timestamp"]=pd.to_datetime(testdf['timestamp'])

print("Time variable type : ",traindf['timestamp'].dtype)



traindf["year"] = traindf["timestamp"].dt.year

traindf["month"] = traindf["timestamp"].dt.month

testdf["year"] = testdf["timestamp"].dt.year

testdf["month"] = testdf["timestamp"].dt.month
mean = pd.DataFrame(traindf.groupby(traindf["year"])["price_doc"].agg('mean').apply(lambda x : format(x,'10.0f')))

mean["price_doc"]=mean["price_doc"].astype(int)

mean.reset_index(level=0, inplace=True)

sns.barplot(x="year",y="price_doc",data=mean, color="grey")
mean = pd.DataFrame(traindf.groupby([traindf["month"]])["price_doc"].agg('mean').apply(lambda x : format(x,'10.0f')))

mean["price_doc"]=mean["price_doc"].astype(int)

mean["month"]= mean.index

#mean.reset_index(level=0, inplace=True)

plt.figure(figsize=(30,10))

sns.barplot(x="month",y="price_doc",data=mean,color="grey")

plt.show()
mean = pd.DataFrame(traindf.groupby([traindf["year"],traindf["month"]])["price_doc"].agg('mean').apply(lambda x : format(x,'10.0f')))

mean["price_doc"]=mean["price_doc"].astype(int)

mean["year"]= mean.index

#mean.reset_index(level=0, inplace=True)

plt.figure(figsize=(40,20))

sns.barplot(x="year",y="price_doc",data=mean, color="grey")

plt.xlabel('index', fontsize=50)

plt.ylabel('price', fontsize=50)

plt.show()



#plt.bar(range(mean.shape[0]),mean.price_doc.values)

#plt.xlabel('index', fontsize=50)

#plt.ylabel('price', fontsize=50)

#plt.show()
print("Train Shape",traindf.shape)

print("Test Shape",testdf.shape)
nulltable = pd.DataFrame(traindf.isnull().sum()/traindf.shape[0]).reset_index()

nulltable.columns = ['column_name', 'missing_count']

non_nullcolumns = nulltable[nulltable['missing_count']==0]

nulltable.sort_values(by="missing_count",ascending = 0).head(20)
null_columns_remove= list(nulltable[nulltable['missing_count']>.10]["column_name"])

print(traindf.shape)

print(testdf.shape)

traindf1 = traindf.drop(null_columns_remove,axis=1)

testdf1 = testdf.drop(null_columns_remove,axis=1)

print(traindf1.shape)

print(testdf1.shape)
datatype = traindf1.dtypes.reset_index()

datatype.columns = ['column_name', 'datatype']

datatype["datatype"].value_counts()
table = pd.merge(nulltable,datatype, how="inner",on="column_name")

table["Null"] = np.where(table["missing_count"]==0,"No","Yes")

table["type"] = np.where(table["datatype"]=="object","Category","Number")

print("There are ",non_nullcolumns.shape[0],"columns with no null values and ",

     traindf1.shape[1]-non_nullcolumns.shape[0], "columns with null values that have been filled with -1 \n")

print(pd.crosstab(table.type,table["Null"]))
traindf1.fillna(-1,inplace=True)

testdf1.fillna(-1,inplace=True)
for i in table[table["datatype"]=="object"]["column_name"] :

    print("\n",i)

    print(traindf1[i].value_counts())
traindf2 = traindf1.drop("sub_area",axis=1)

testdf2 = testdf1.drop("sub_area",axis=1)

print("After removing sub_area, shape reduced from ", traindf1.shape,"to",traindf2.shape)
traindf2 = pd.get_dummies(traindf2)

testdf2 = pd.get_dummies(testdf2)

print(traindf2.dtypes.value_counts())
X = traindf2.drop(['id','timestamp','price_doc'], axis =1)

X = add_constant(X)

viftab = pd.Series([variance_inflation_factor(X.values, i) 

               for i in range(X.shape[1])], 

              index=X.columns)
vifdf=pd.DataFrame(viftab).reset_index()

vifdf.columns = ["Name","VIF"]

vifdf.sort_values("VIF",ascending=False)

#vifdf["VIF"] = vifdf["VIF"]

#vifdf["VIF"] = vifdf["VIF"].apply(lambda x : x.strip())

#vifdf1=vifdf[vifdf["VIF"] != 'inf']

#vifdf1["VIF"] =pd.to_numeric(vifdf1["VIF"])

#vifdf1.sort_values("VIF",ascending=False)



noncollinearvar=list(vifdf[vifdf["VIF"]<10]["Name"])

noncollinearvar.remove('const')

print(noncollinearvar)

vifdf.dtypes



noncollinearvar = ['full_sq', 'floor', 'green_zone_part', 'indust_part', 'school_education_centers_top_20_raion', 'healthcare_centers_raion', 'university_top_20_raion', 'ID_metro', 'green_zone_km', 'industrial_km', 'cemetery_km', 'ID_railroad_station_walk', 'ID_railroad_station_avto', 'water_km', 'big_road1_km', 'ID_big_road1', 'ID_big_road2', 'ID_bus_terminal', 'church_synagogue_km', 'catering_km', 'green_part_500', 'prom_part_500', 'office_sqm_500', 'trc_count_500', 'trc_sqm_500', 'mosque_count_500', 'leisure_count_500', 'sport_count_500', 'market_count_500', 'trc_sqm_1000', 'mosque_count_1000', 'sport_count_1000', 'market_count_1000', 'trc_sqm_1500', 'mosque_count_1500', 'market_count_1500', 'trc_sqm_2000', 'mosque_count_2000', 'market_count_2000', 'mosque_count_3000', 'mosque_count_5000', 'year', 'month']
noncollinearvar = ['full_sq', 'floor', 'green_zone_part', 'indust_part', 'school_education_centers_top_20_raion', 'healthcare_centers_raion', 'university_top_20_raion', 'ID_metro', 'green_zone_km', 'industrial_km', 'cemetery_km', 'ID_railroad_station_walk', 'ID_railroad_station_avto', 'water_km', 'big_road1_km', 'ID_big_road1', 'ID_big_road2', 'ID_bus_terminal', 'church_synagogue_km', 'catering_km', 'green_part_500', 'prom_part_500', 'office_sqm_500', 'trc_count_500', 'trc_sqm_500', 'mosque_count_500', 'leisure_count_500', 'sport_count_500', 'market_count_500', 'trc_sqm_1000', 'mosque_count_1000', 'sport_count_1000', 'market_count_1000', 'trc_sqm_1500', 'mosque_count_1500', 'market_count_1500', 'trc_sqm_2000', 'mosque_count_2000', 'market_count_2000', 'mosque_count_3000', 'mosque_count_5000', 'year', 'month']
X = traindf2[noncollinearvar]

Y = traindf2['price_doc']

param_grid = [{"max_depth":[5,8,10,12,15], "max_features":["sqrt","log2","auto"]}]

grid = GridSearchCV(GradientBoostingRegressor(),param_grid, cv=3, n_jobs =-1)

grid.fit(X,Y)

gb = grid.best_estimator_

gb.fit(X,Y)
plt.figure(figsize=(10,8))

pd.Series(gb.feature_importances_, index=X.columns).nlargest(30).plot(kind='barh')
importantfeatures = pd.Series(gb.feature_importances_, index=X.columns).nlargest(20).reset_index()

importantfeatures.columns = ["columnname","importance"]

#columnlist= importantfeatures["columnname"]

columnlist = list(importantfeatures["columnname"])

columnlist
for i in noncollinearvar:

    print(stats.pearsonr(traindf2[i], traindf2["price_doc"]))

  #  print(stats.spearmanr(traindf1["full_sq"], traindf1["price_doc"]))
plotdf=traindf2[(traindf2["full_sq"]<175) & (traindf2["price_doc"]<20000000)]

for i in noncollinearvar:

    sns.lmplot(i,"price_doc",data=plotdf,fit_reg=False)
traindf3 = traindf2[(traindf2["full_sq"]<175) & (traindf2["price_doc"]<20000000)]
featuredata=traindf3[noncollinearvar]

testdf3=testdf2[noncollinearvar]

Y=traindf3["price_doc"]

#featuredata.drop("water_km",axis=1,inplace=True)

featuredata.shape
xtrain, xtest, ytrain, ytest = train_test_split(featuredata,Y, test_size=0.3,random_state=seed)

print (xtrain.shape, ytrain.shape)

print (xtest.shape, ytest.shape)
def RMSLE(y, y0):

    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))

    

def RMSE(y, y0):

    return np.sqrt(np.mean(np.square(y- y0)))

    

def r2(y,y0):

    return r2_score(y, y0)

    
model = sm.OLS(ytrain,xtrain)

results = model.fit()

print(results.summary())
print(stats.pearsonr(traindf3["big_road1_km"], traindf3["price_doc"]))
predictions = results.predict(xtest)

predictions.head()
LR_RMSLE = RMSLE(ytest,predictions)

LR_RMSE = RMSE(ytest,predictions)

LR_R2 = r2(ytest,predictions)



print("RMSLE :",LR_RMSLE,"\nRMSE:",LR_RMSE,"\nR2:",LR_R2)
ridgereg = Ridge(alpha=.1,normalize=True, max_iter=200)

ridgereg.fit(xtrain,ytrain)

ridgereg_pred = ridgereg.predict(xtest)
RR_RMSLE = RMSLE(ytest,ridgereg_pred)

RR_RMSE = RMSE(ytest,ridgereg_pred)

RR_R2 = r2(ytest,ridgereg_pred)



print("RMSLE :",RR_RMSLE,"\nRMSE:",RR_RMSE,"\nR2:",RR_R2)
lassoreg = Lasso(alpha=.1,normalize=True, max_iter=200)

lassoreg.fit(xtrain,ytrain)

lassoreg_pred = lassoreg.predict(xtest)
LAR_RMSLE = RMSLE(ytest,lassoreg_pred)

LAR_RMSE = RMSE(ytest,lassoreg_pred)

LAR_R2 = r2(ytest,lassoreg_pred)



print("RMSLE :",LAR_RMSLE,"\nRMSE:",LAR_RMSE,"\nR2:",LAR_R2)
param_grid = [{"max_depth":[5,8,10], "max_features":["sqrt","log2","auto"]}]

grid = GridSearchCV(RandomForestRegressor(),param_grid, cv=3, n_jobs =-1)

grid.fit(xtrain,ytrain)

rf = grid.best_estimator_

rf.fit(xtrain,ytrain)
print("The Best hyperparamters of the model are",grid.best_params_)
rf_pred = rf.predict(xtest)

rf_RMSLE = RMSLE(ytest,rf_pred)

rf_RMSE = RMSE(ytest,rf_pred)

rf_R2 = r2(ytest,rf_pred)



print("RMSLE :",rf_RMSLE,"\nRMSE:",rf_RMSE,"\nR2:",rf_R2)
plt.figure(figsize=(10,8))

pd.Series(rf.feature_importances_, index=xtrain.columns).nlargest(30).plot(kind='barh')
param_grid = [{"max_depth":[5,8,10], "max_features":["sqrt","log2","auto"],"n_estimators":[100,200,300]}]

grid = GridSearchCV(GradientBoostingRegressor(),param_grid, cv=3, n_jobs =-1)

grid.fit(xtrain,ytrain)

gb = grid.best_estimator_

gb.fit(xtrain,ytrain)
print("The Best hyperparamters of the model are",grid.best_params_)
gb_pred = gb.predict(xtest)

gb_RMSLE = RMSLE(ytest,gb_pred)

gb_RMSE = RMSE(ytest,gb_pred)

gb_R2 = r2(ytest,gb_pred)



print("RMSLE :",gb_RMSLE,"\nRMSE:",gb_RMSE,"\nR2:",gb_R2)

#testgb_pred = gb.predict(testdf3)
plt.figure(figsize=(10,8))

pd.Series(gb.feature_importances_, index=xtrain.columns).nlargest(30).plot(kind='barh')
model_compare=[{"Name":'Linear Reg',"RMSLE":LR_RMSLE,"RMSE":LR_RMSE,"R2":LR_R2},

               {"Name":'Ridge Reg',"RMSLE":RR_RMSLE,"RMSE":RR_RMSE,"R2":RR_R2},

               {"Name":'Lasso Reg',"RMSLE":LAR_RMSLE,"RMSE":LAR_RMSE,"R2":LAR_R2},

               {"Name":'Random Forest',"RMSLE":rf_RMSLE,"RMSE":rf_RMSE,"R2":rf_R2},

               {"Name":'Gradient Boosting',"RMSLE":gb_RMSLE,"RMSE":gb_RMSE,"R2":gb_R2}]



model_comparedf = pd.DataFrame(model_compare)

model_comparedf =model_comparedf.set_index('Name')

print(model_comparedf)
model_comparedf[["R2"]].plot(figsize=(7,3), xticks=range(0, 5)).legend(title='Name', bbox_to_anchor=(1, 1))
model_comparedf[["RMSLE"]].plot(figsize=(7,3), xticks=range(0, 5)).legend(title='Name', bbox_to_anchor=(1, 1))
model_comparedf[["RMSE"]].plot(figsize=(7,3), xticks=range(0, 5)).legend(title='Name', bbox_to_anchor=(1, 1))