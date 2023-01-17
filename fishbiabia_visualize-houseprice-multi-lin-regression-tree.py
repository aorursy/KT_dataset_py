import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sn
from pandas.plotting import scatter_matrix

%matplotlib inline

df=pd.read_csv('../input/HousePrice.csv')
print(df.shape)   #(1460, 81)
print(df.describe())
#fig1:scatterplot
fig,ax=plt.subplots()
df.plot(kind='scatter',x='GrLivArea',y='SalePrice',ax=ax)
#alternatively,we can:
#plt.scatter(df['GrLivArea'],df['SalePrice'])
#plt.show()

HouseAge=df['YrSold']-df['YearBuilt']
#sn.distplot(HouseAge,color='seagreen', kde=False)
plt.barh(HouseAge,width=df["SalePrice"],color="green")
plt.title("Sale Price vs Age of house")
plt.ylabel("Age of house")
plt.xlabel("Sale Price");
#We find that 
plt.xticks(rotation=45) 
PriceperSF=df['SalePrice']/df['GrLivArea']
sn.barplot(df["Neighborhood"],PriceperSF)
plt.title("Sale Price per square feet vs Neighborhood");
#fig3: zoning class vs saleprice
fig,ax=plt.subplots()
sn.violinplot(data=df[['MSZoning','SalePrice']],x='MSZoning',y='SalePrice',ax=ax)
sn.stripplot(x="HeatingQC", y="SalePrice",data=df,hue='CentralAir',jitter=True,dodge=True)
plt.title("Sale Price vs Heating Quality");
#fig4
fig,ax=plt.subplots()
fig=sn.boxplot(data=pd.concat([df['ExterQual'],df['SalePrice']],axis=1),x='ExterQual',y='SalePrice',ax=ax)
#fig5: barchart
fig,ax=plt.subplots()
sn.barplot(data=df[['CentralAir','SalePrice']],x='CentralAir',y='SalePrice',ax=ax)
#fig6: scatterplot matrix
scatter_matrix(df[['YrSold','SalePrice','TotalBsmtSF','OverallQual', 'GrLivArea']], diagonal='kde', alpha=0.3,figsize=(50,50))
#correlation matirx
corr=df[['OverallQual','OverallCond','GarageCars','YearBuilt','SalePrice','LowQualFinSF','Fireplaces']].corr()
mask=np.array(corr)
mask[np.tril_indices_from(mask)] = False
sn.heatmap(corr, mask=mask,vmax=.8, square=True,annot=True)
#bubble plot
plt.scatter(df['YearBuilt'],df['SalePrice'],c=df['OverallQual'],s=df['GrLivArea']*0.01,alpha=0.5)
plt.show()
#missingvalue stastistics
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
print(len(df[df['GarageArea']==0])) #81
print(len(df[df['TotalBsmtSF']==0])) #37
print(len(df[df['PoolArea']==0])) #1453  
print(len(df[df['Fireplaces']==0]))#690

for i in ['GarageType','GarageFinish','GarageCond','GarageQual']:
    df[i]=df[i].fillna("None")
df['GarageYrBlt']=df['GarageYrBlt'].fillna(0)
for i in ['BsmtExposure', 'BsmtFinType2','BsmtFinType1','BsmtCond', 'BsmtQual']:
    df[i] = df[i].fillna("None")
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
df['FireplaceQu']=df['FireplaceQu'].fillna("None")
df['Fence']=df['Fence'].fillna("None")
df['MiscFeature'] = df['MiscFeature'].fillna("None")
df['Alley'] = df['Alley'].fillna("None")
df['PoolQC'] = df['PoolQC'].fillna("None")
df['LotFrontage'] =df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
print(df.isnull().sum().sort_values(ascending=False).head())
fig,ax=plt.subplots()
df.plot(kind='scatter',x='GrLivArea',y='SalePrice',ax=ax)
#remove abnormal point that has large area but extremely low price
df=df[(df['GrLivArea']<4000) | (df['SalePrice']>200000)]
fig,ax=plt.subplots()
df.plot(kind='scatter',x='GrLivArea',y='SalePrice',ax=ax)
fig,ax=plt.subplots()
fig=sn.boxplot(x='OverallQual',y='SalePrice',data=pd.concat([df['OverallQual'],df['SalePrice']],axis=1),ax=ax)
#we can leave the above the range's outliers here since we are not sure whether there are any significant factors that affect the sale price

#countSeries=df['MSZoning'].value_counts()
#second=countSeries[1]
#print(countSeries[countSeries == second].index[0])
columnlist=list(df)
columnlist.remove('Id')
droplist=[]
for i in columnlist:
    count=df[i].value_counts()
    first=count.values[0]
    second=count.values[1]
    r=float(second/first)
    if r<0.05:
        droplist.append(i)
print(droplist)
    
df=df.drop(droplist,1)
print(df.shape) #55 left
print(df.columns)
data=df[['SalePrice','GrLivArea','GarageCars','RoofStyle','MSZoning','KitchenQual','CentralAir','TotalBsmtSF']]
data=data.dropna()
missing = data.isnull().sum()
print(missing)
from scipy.stats import norm
sn.distplot(data['SalePrice'],fit=norm_hist)
data['SalePrice']=np.log1p(data['SalePrice']) #Sale price is skewd to the right
sn.distplot(data['GarageCars'],fit=norm)
#import scipy
#print(data.skew())
#sn.distplot(data['GrLivArea'],fit=norm)
data['LogGrArea']=np.log1p(data['GrLivArea'])
sn.distplot(data['LogGrArea'],fit=norm)
data['LogBsmtSF']=np.log1p(data['TotalBsmtSF'])
sn.distplot(data['LogBsmtSF'],fit=norm)
#data=data.dropna()
#sn.distplot(data['SalePrice'],fit=norm) #log transform of saleprice
print(data.dtypes)
data1=pd.get_dummies(data,drop_first=True)
print(data1.dtypes)
from sklearn.model_selection import train_test_split
columnlist=list(data1.columns)
columnlist.remove('SalePrice')
columnlist.remove('GrLivArea')
columnlist.remove('TotalBsmtSF')
X_train, X_test, y_train, y_test = train_test_split(data1[columnlist],data1['SalePrice'],test_size=0.3, random_state=42)
print("Training set::{}{}".format(X_train.shape,y_train.shape))
print("Testing set::{}".format(X_test.shape))
#FULL model
from sklearn import  linear_model
X_train=X_train.astype(float)
y_train=pd.DataFrame(y_train)
y_train = y_train.values.reshape(-1,1)
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_train,y_train)
print(lin_reg.coef_)
import statsmodels.api as sm
X_train2 = sm.add_constant(X_train, prepend=False)
reg = sm.OLS(y_train,X_train2).fit()
print(reg.summary())
#check normality 
import scipy.stats as stats
stats.probplot(reg.resid, dist="norm", plot=plt)
plt.show()
plt.hist(reg.resid,100)
plt.scatter(reg.fittedvalues,reg.resid)
plt.title("Residual vs. fit plot")
plt.xlabel("fitted value")
plt.ylabel("residual")
## Assumption checking - constant variance assumption and mean-zero assumption
plt.plot(reg.resid, '-')  # solid line
plt.title("Residual vs. Observation number plot")
plt.xlabel("observation number")
plt.ylabel("residual")
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
# X_train.shape[1] refers to the column number
vif["features"] = X_train.columns
vif.round(1)
import time
def processSubset(feature_set):
    # Fit model on feature_set and calculate AIC
    X_select = X_train[list(feature_set)]
    X = sm.add_constant(X_select, prepend=False)
    model = sm.OLS(y_train,X)
    regr = model.fit()
    aic = regr.aic
    return {"model":regr, "AIC":aic, "predict":feature_set}
def getBest(k):
    
    tic = time.time()
    
    results = []
    
    for combo in itertools.combinations(X_train.columns, k):
        results.append(processSubset(combo))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the lowest AIC
    best_model = models.loc[models['AIC'].argmin()]
    
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model
def forward(predictors):
    
    if 'const' in predictors:
        predictors.remove('const')
    
    
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X_train.columns if p not in predictors]
    
    tic = time.time()
    
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p]))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    aic=[]
    name=[]
    predict=[]
    for i in range(len(results)):
        aic.append(results[i]['AIC'])
        name.append(results[i]['model'])
        predict.append(results[i]['predict'])
        
    
    models['AIC']=aic
    models['model']=name
    models['predict']=predict
    
    # Choose the model with the lowest AIC
    best_model = models.loc[models['AIC'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model
models_fwd = pd.DataFrame(columns=["AIC", "model","predict"])

tic = time.time()
predictors = []

for i in range(1,len(X_train.columns)+1):    
    models_fwd.loc[i] = forward(predictors)
    predictors = models_fwd.loc[i]["model"].model.exog_names

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")
models_fwd
index = models_fwd['AIC'].argmin()
predictor = models_fwd.loc[index, 'predict']
print(predictor)
## see the model after selection
X_train3 = X_train[predictor]
X_train3 = sm.add_constant(X_train3, prepend=False)
reg_selection = sm.OLS(y_train,X_train3).fit()
print(reg_selection.summary())
## For the model selected by stepwise selection, 
## using the k-fold cross validation (specifically 10-fold) to reduce overfitting affects
# cross_val_predict function returns cross validated prediction values as fitted by the model object.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
lin_reg_s = linear_model.LinearRegression()
lin_reg_s = lin_reg_s.fit(X_train3, y_train)
predicted = cross_val_predict(lin_reg_s, X_train3, y_train, cv=10)
## Assumption checking - check of homoscedasticity
## residual plot
fig, ax = plt.subplots()
ax.scatter(y_train, y_train-predicted)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()
r2_scores = cross_val_score(lin_reg_s, X_train3, y_train, cv=10)
mse_scores = cross_val_score(lin_reg_s, X_train3, y_train, cv=10,scoring='neg_mean_squared_error')
print("R-squared::{}".format(r2_scores))
print("MSE::{}".format(mse_scores))
fig, ax = plt.subplots()
ax.plot([i for i in range(len(r2_scores))],r2_scores,lw=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('R-Squared')
ax.title.set_text("Cross Validation Scores, Avg:{}".format(np.average(r2_scores)))
plt.show()
## test dataset
X_test=X_test.astype(float)
X_test2 = X_test[predictor]
lin_reg_s_test = linear_model.LinearRegression()
lin_reg_s_test = lin_reg_s.fit(X_test2, y_test)
y_pred = lin_reg_s_test.predict(X_test2)
#y_pred = pd.DataFrame(y_pred)
residuals = pd.DataFrame(y_test-y_pred)
from sklearn import metrics
r2_score = lin_reg_s_test.score(X_test2,y_test)
print("R-squared::{}".format(r2_score))
print("MSE: %.2f" % metrics.mean_squared_error(y_test, y_pred))
fig, ax = plt.subplots()
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot with R-Squared={}".format(np.average(r2_score)))
plt.show()
print("MSE: {}".format(metrics.mean_squared_error(y_test, y_pred)))
X_test_vif = X_test2.copy()
vif2 = pd.DataFrame()
vif2["VIF Factor"] = [variance_inflation_factor(X_test_vif.values, i) for i in range(X_test_vif.shape[1])]
vif2["features"] = X_test_vif.columns
vif2.round(1)
selected =list(X_test_vif.columns)
selected.remove('LogBsmtSF')
selected.append('B_A_ratio')
selected.remove('CentralAir_Y')
print(selected)
X_train_1=X_train.copy()
X_train_1['B_A_ratio']=X_train_1['LogBsmtSF']/X_train_1['LogGrArea']
X_train4 = X_train_1[selected]
#X_train4['Bsmt_Gr_ratio']=X_train4['TotalBsmtSF']/X_train4['GrLivArea']
X_train4 = sm.add_constant(X_train4, prepend=False)
reg_selection = sm.OLS(y_train,X_train4).fit()
print(reg_selection.summary())
lin_reg_s = linear_model.LinearRegression()
X_train4=X_train4.drop(['const'],axis=1)
lin_reg_s = lin_reg_s.fit(X_train4, y_train)
predicted = cross_val_predict(lin_reg_s, X_train4, y_train, cv=10)
fig, ax = plt.subplots()
ax.scatter(y_train, y_train-predicted)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()
fig, ax = plt.subplots()
ax.plot([i for i in range(len(r2_scores))],r2_scores,lw=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('R-Squared')
ax.title.set_text("Cross Validation Scores, Avg:{}".format(np.average(r2_scores)))
plt.show()
print(X_train4.shape)
## test dataset
#X_test=X_test.astype(float)
#X_test['TotalArea']=X_test['TotalBsmtSF']+X_test['GrLivArea']
X_test_1=X_test.copy()
X_test_1['B_A_ratio']=X_test_1['LogBsmtSF']/X_test_1['LogGrArea']
X_test3 = X_test_1[selected]
#X_test3 = sm.add_constant(X_test3, prepend=False)
print(X_test3.shape)
#lin_reg_s_test1= linear_model.LinearRegression()
#lin_reg_s_test1= lin_reg_s.fit(X_test3, y_test)
y_pred1 = lin_reg_s.predict(X_test3)
#y_pred = pd.DataFrame(y_pred)
y_test=pd.DataFrame(y_test)
y_test=y_test.values.reshape(-1,1)
residuals = pd.DataFrame(y_test-y_pred1)
#r2_score=metrics.r2_score(y_test,y_pred1)
r2_score = lin_reg_s.score(X_test3,y_test)
print("R-squared::{}".format(r2_score))
print("MSE: %.2f" % metrics.mean_squared_error(y_test, y_pred1))
fig, ax = plt.subplots()
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot with R-Squared={}".format(np.average(r2_score)))
plt.show()
print("MSE: {}".format(metrics.mean_squared_error(y_test, y_pred1)))
X_test_vif_1 = X_test3.copy()
vif3 = pd.DataFrame()
vif3["VIF Factor"] = [variance_inflation_factor(X_test_vif_1.values, i) for i in range(X_test_vif_1.shape[1])]
vif3["features"] = X_test_vif_1.columns
vif3.round(1)
#coefficient=pd.concat(pd.DataFrame(selected),pd.DataFrame(lin_reg_s.coef_))
coef_name=pd.DataFrame(selected)
coef_value=pd.DataFrame(lin_reg_s.coef_).transpose()
coefficient=pd.concat([coef_name,coef_value],axis=1)
print(coefficient)
print(X_test3[['LogGrArea','B_A_ratio','GarageCars','KitchenQual_TA','MSZoning_RM', 'KitchenQual_Fa', 'KitchenQual_Gd', 'RoofStyle_Hip', 'MSZoning_FV', 'MSZoning_RL', 'MSZoning_RH']].corr())
#use test set for checking model assumption
stats.probplot(reg_selection.resid, dist="norm", plot=plt)
plt.show()
plt.hist(reg_selection.resid,100)
plt.scatter(reg_selection.fittedvalues,reg_selection.resid)
plt.title("Residual vs. fit plot")
plt.xlabel("fitted value")
plt.ylabel("residual")
## Assumption checking - constant variance assumption and mean-zero assumption
plt.plot(reg_selection.resid, '-')  # solid line
plt.title("Residual vs. Observation number plot")
plt.xlabel("observation number")
plt.ylabel("residual")
import graphviz
from sklearn import preprocessing
cat_attributes=pd.get_dummies(data[['RoofStyle','MSZoning','KitchenQual','CentralAir']])
data2=pd.concat([cat_attributes, data[['GrLivArea','GarageCars','TotalBsmtSF']]], axis=1)
#data['SalePrice']=np.expm1(data['SalePrice'])
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data2,data['SalePrice'],test_size=0.3, random_state=42)
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(X_train_reg,y_train_reg)
dtr.score(X_train_reg,y_train_reg)
y_pred_reg = dtr.predict(X_test_reg)
from sklearn.metrics import mean_squared_error
import itertools
print("MSE::{}".format(mean_squared_error(y_test_reg, y_pred_reg)))
## display the tree
import pydotplus
from IPython.display import Image 
from sklearn.externals.six import StringIO 
from sklearn.tree import export_graphviz
y_train_reg=y_train_reg.values.reshape((1020,1))
y_train_reg=pd.DataFrame(y_train_reg)
y_train_reg.columns =['SalePrice']
from IPython.display import Image
dot_data = StringIO()
export_graphviz(dtr, out_file=dot_data, feature_names=X_train_reg.columns,
                         class_names=y_train_reg.columns,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values 
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.xticks(rotation=45) 
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()  
plot_feature_importances(dtr.feature_importances_,'Decision Tree regressor', X_train_reg.columns)
im=dtr.feature_importances_.reshape(-1,1)
importance=pd.DataFrame(data=im,index=X_train_reg.columns)
importance.columns=['importance']
importance.sort_values(by='importance',inplace=True,ascending=False)
print(importance)
