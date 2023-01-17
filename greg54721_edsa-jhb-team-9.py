import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns

import time

from scipy import stats

from scipy.stats import norm, skew 

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn 



class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'
test = pd.read_csv('../input/test.csv')

train= pd.read_csv('../input/train.csv')
train.head()
test.head()
train.info()
test.info()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
print (color.BOLD + "Train data:" + color.END)

print ("Number of columns: " + str (train.shape[1]))

print ("Number of rows: " + str (train.shape[0]))



print(color.BOLD + '\nTest data: ' + color.END)

print ("Number of columns:" + str (test.shape[1]))

print ("Number of columns:" +  str (test.shape[0]))
train.SalePrice.describe()
y_skewness=round(train.SalePrice.skew(),3)



plt.figure(figsize=(17, 8))

sns.distplot(train.SalePrice,bins=20, color='green')

plt.title('SalePrice')

plt.xlabel(" ")

plt.ylabel(" ")

print(color.BOLD + 'Skew is :',y_skewness , color.END)

plt.show()
y = np.log(train.SalePrice)



plt.figure(figsize=(17, 8))

sns.distplot(y, bins=20, color='orange')

plt.title('SalePrice')

plt.xlabel(" ")

plt.ylabel(" ")

print(color.BOLD + 'Skew is :', round(y.skew(),3), color.END)

plt.show()
corr= train.corr()
corr.head()
fig = plt.figure(figsize = (30,30))

sns.heatmap(corr, annot = True, cmap= 'summer')

plt.show()
top_features= corr.index[corr['SalePrice'] > 0.5]

top_corr= train[top_features].corr().sort_values(by='SalePrice',ascending=False)



plt.subplots(figsize=(17,8))

sns.heatmap(top_corr,annot=True,cmap='summer')

plt.show()
multi_coll = train.corr().abs()



multi_coll_features = (multi_coll.where(np.triu(np.ones(multi_coll.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))



multi_coll_features=pd.DataFrame(multi_coll_features[multi_coll_features >0.50],columns= ['Correlation'])



multi_coll_features
train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'],axis = 1, inplace = True)

test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'],axis = 1, inplace = True)
plt.subplots(figsize=(17,8))

fig = sns.boxplot(x='OverallQual', y='SalePrice', data=train)

fig.axis(ymin=0, ymax=800000)

plt.show()
plt.figure(figsize=(17, 8))

sns.scatterplot(x=train['GarageCars'], y=train['SalePrice'])

plt.show()
plt.figure(figsize=(17, 8))

sns.scatterplot(x=train['TotalBsmtSF'], y=train['SalePrice'])

plt.show()
plt.subplots(figsize=(17, 8))

fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=train)

fig.axis(ymin=0, ymax=800000)

plt.xticks(rotation=90);

plt.show();
plt.figure(figsize=(17, 8))

sns.scatterplot(x=train['LotArea'], y=train['SalePrice'])

plt.show()
plt.figure(figsize=(17, 8))

sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])

plt.show()
missing_train = train.isnull().sum() / len(train.index) * 100

missing_train = round(missing_train[missing_train > 0], 2)

missing_train.sort_values(ascending = False, inplace = True)

pd.DataFrame(missing_train, columns = ['Percentage'])
plt.subplots(figsize=(17, 8))

plt.xticks(rotation='90')

sns.barplot(x = missing_train.index, y = missing_train, palette = "Blues_r", ec = 'Black')

plt.xlabel("Column Name")

plt.ylabel("Percentage Missing")

plt.title("Percentage Missing in Train Data")

plt.show()
len(missing_train)
missing_test = test.isnull().sum() / len(test.index) * 100

missing_test = round(missing_test[missing_test > 0], 2)

missing_test.sort_values(ascending = False, inplace = True)

pd.DataFrame(missing_test, columns = ['Percentage'])
plt.subplots(figsize=(17, 8))

plt.xticks(rotation='90')

sns.barplot(x = missing_test.index, y = missing_test, palette = 'Blues_d', ec = 'Black')

plt.xlabel("Column Name")

plt.ylabel("Percentage Missing")

plt.title("Percentage Missing in Test Data")

plt.show()
len(missing_test)
missing_values = pd.concat([train.isnull().sum(), train.isnull().sum() / train.shape[0], test.isnull().sum(), test.isnull().sum() / test.shape[0]], axis=1, keys=['Train', 'Percentage', 'Test', 'Percentage'])

missing_values = missing_values[missing_values.sum(axis=1) > 0].sort_values(by = 'Train', ascending = False)

missing_values['Percentage'] = round(missing_values['Percentage'] * 100, 2)

missing_values['Test'] = missing_values['Test'].astype(int)

missing_values
train_ID = train['Id']

test_ID = test['Id']
train.drop("Id",axis = 1, inplace = True)

test.drop("Id",axis = 1, inplace = True)
train = train.drop('SalePrice', axis=1)
train_num = len(train)

train_num
test_num = len(test)

test_num
joined_datasets = pd.concat(objs = [train, test], axis=0)
joined_datasets.head()
joined_datasets.shape
joined_datasets_nan = (joined_datasets.isnull().sum() / len(joined_datasets)) * 100
joined_datasets_nan = joined_datasets_nan.drop(joined_datasets_nan[joined_datasets_nan == 0].index).sort_values(ascending = False)

missing_data_perc = pd.DataFrame({'Percentage':joined_datasets_nan})

missing_data_perc
len(missing_data_perc)
num_features = joined_datasets.select_dtypes(include=[np.number])
num = num_features.isnull().sum()

num = num[num > 0]

num
len(num)
cat_features = joined_datasets.select_dtypes(exclude = [np.number])
cat = cat_features.isnull().sum()

cat = cat[cat > 0]

cat
len(cat)
for cols in ("Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Fence","FireplaceQu",

            "GarageType","GarageFinish","GarageQual","GarageCond","Functional","MasVnrType","MiscFeature","PoolQC",'MSSubClass'):

    

    joined_datasets[cols] = joined_datasets[cols].fillna("None")
for cols in ("MiscVal","PoolArea",'TotalBsmtSF',"MiscVal","BsmtFinSF1","BsmtFinSF2", "BsmtHalfBath",

            "BsmtFullBath","BsmtUnfSF","MasVnrArea",'TotalBsmtSF',"GarageYrBlt","GarageCars"):

    

    joined_datasets[cols] = joined_datasets[cols].fillna(0)
for cols in ('Electrical','KitchenQual','MSZoning', 'Exterior1st','Exterior2nd','SaleType'):

    

    joined_datasets[cols] = joined_datasets[cols].fillna(joined_datasets[cols].mode()[0])

joined_datasets["LotFrontage"] = joined_datasets.groupby("Neighborhood")["LotFrontage"].transform(lambda LotF: LotF.fillna(LotF.median()))
joined_datasets = joined_datasets.drop(['Utilities'], axis=1)
datasets_na = (joined_datasets.isnull().sum() / len(joined_datasets)) * 100

sum(datasets_na)
joined_datasets.head()
joined_datasets['MSSubClass'] = joined_datasets['MSSubClass'].apply(str)

joined_datasets['OverallCond'] = joined_datasets['OverallCond'].astype(str)

joined_datasets['YrSold'] = joined_datasets['YrSold'].astype(str)

joined_datasets['MoSold'] = joined_datasets['MoSold'].astype(str)
cate_feats = joined_datasets.dtypes[joined_datasets.dtypes == "object"].index
cate_feats.shape
from sklearn.preprocessing import LabelEncoder
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for col in columns:

    label = LabelEncoder() 

    label.fit(list(joined_datasets[col].values)) 

    joined_datasets[col] = label.transform(list(joined_datasets[col].values))
joined_datasets.shape
numeric_feats = joined_datasets.dtypes[joined_datasets.dtypes != "object"].index
skewed_var = joined_datasets[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_var
len(skewed_var)
top_skewed_var = skewed_var[abs(skewed_var) > 0.75]

top_skewed_var.sort_values(ascending=False)
len(top_skewed_var)
top_skewed_var = top_skewed_var.index

joined_datasets[top_skewed_var] = np.log1p(joined_datasets[top_skewed_var])
joined_datasets = pd.get_dummies(joined_datasets)

print(joined_datasets.shape)
train = joined_datasets[:train_num]

test = joined_datasets[train_num:]
print(train.shape)

print(test.shape)
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso , Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics
reg_line= LinearRegression()

tree = DecisionTreeRegressor()

forest = RandomForestRegressor()

lasso = Lasso()

ENet = ElasticNet()

ridge = Ridge()
print('lm: ', reg_line.get_params().keys())

print('lasso: ', lasso.get_params().keys())

print('ridge: ', ridge.get_params().keys())

print('en: ', ENet.get_params().keys())

print('regre_tree: ', tree.get_params().keys())

print('forest: ', forest.get_params().keys())
param_grid_tree = dict(max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],

                       min_samples_split = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 

                       min_samples_leaf = [0.1, 0.2, 0.3, 0.4, 0.5], 

                       max_features = list(range(1,train.shape[1])))

param_grid_lasso = dict(alpha = [0.0001, 0.00015, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.01, 0.1])

param_grid_ridge = dict(alpha = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000])#, max_iter=[100,110,120,130,140])

param_grid_en = dict(alpha = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000])

param_grid_forest = dict(n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200], 

                        max_depth = np.linspace(1, 32, 32, endpoint=True), 

                        min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True), 

                        min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True), 

                        max_features = list(range(1, train.shape[1])))
"""

Kfold cross validation is a resampling method that splits data into groups specified by the value associated with n_folds.

Multiple runs of a model are performed by randomly shuffling the groups and assigning 1 group as the test set and the remaining groups as the training sets (performed using kFold). 

The model is fit to the training sets and tested on the test set, this is then performed again by reassigning the test and train groups.

"""



n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
X_train,X_test,y_train,y_test= train_test_split(train,y, test_size=0.2, shuffle=False)
reg_line.fit(X_train,y_train)

print (reg_line.score(train, y))
a=reg_line.intercept_
a
b= pd.DataFrame(reg_line.coef_,train.columns,columns=['Coeffient'] )
b.sort_values(by='Coeffient',ascending=False).head(15)
train_lm = reg_line.predict(X_train)

test_lm = reg_line.predict(X_test)
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

sns.regplot(train_lm, (train_lm - y_train))

sns.regplot(test_lm, (test_lm - y_test))

plt.show()
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

sns.regplot(train_lm, y_train)

sns.regplot(test_lm, y_test)

plt.show()
train_plot = y_train.append(pd.Series(y_test))
plt.plot(np.arange(len(y)), reg_line.predict(train), label='Predicted')

plt.plot(np.arange(len(train_plot)), train_plot, label='Training')

plt.plot(np.arange(len(y_test))+len(y_train), y_test, label='Testing')

plt.legend()

plt.show()
import statsmodels.api as sm

def backwardElimination(x,t, Y, sl, columns):

    print(Y.isnull().sum().sum())

    print(x.shape)

    #x= np.append(arr = np.ones((len(x),1)),values = x,axis = 1)

    x_c = pd.DataFrame(data = np.ones((len(x),1)),index = x.index,columns = ['constant'])

    print(x_c.shape)

    columns.append('constant')

    x = x.join(x_c)

    #x = pd.merge(x_c,x,how = 'inner',left_index = True,right_index = True)

    #print(x.tail())

    ini = len(columns)

    numVars = x.shape[1]

    for i in range(0, numVars):

        regressor = sm.OLS(Y,x).fit()

        #print( sm.OLS(Y,x).fit())

        maxVar = max(regressor.pvalues) #.astype(float)

        if maxVar > sl:

            for j in range(0, numVars - i):

                if (regressor.pvalues[j].astype(float) == maxVar):

                    del columns[j]

                    x = x.loc[:, columns]

                    #t = t.loc[:,columns]

    return x
SL = 0.05

col_r = backwardElimination(train.copy(),test.copy() ,y, SL, train.columns.tolist()).columns
len(col_r)
trainL = train[col_r]

testL = test[col_r]
X_trainlm, X_testlm, y_trainlm, y_testlm = train_test_split(trainL, y, test_size=0.2, shuffle=False)
reg_line.fit(X_trainlm, y_trainlm)

print (reg_line.score(X_trainlm, y_trainlm))
bw_a=reg_line.intercept_
bw_a
bw_b= pd.DataFrame(reg_line.coef_,trainL.columns,columns=['Coeffient'] )
bw_b.sort_values(by='Coeffient',ascending=False).head(15)
train_bw = reg_line.predict(X_trainlm)

test_bw = reg_line.predict(X_testlm)
train_plot = y_train.append(pd.Series(y_testlm))
plt.plot(np.arange(len(y)), reg_line.predict(trainL), label='Predicted')

plt.plot(np.arange(len(train_plot)), train_plot, label='Training')

plt.plot(np.arange(len(y_testlm))+len(y_trainlm), y_testlm, label='Testing')

plt.legend()

plt.show()
random_tree = RandomizedSearchCV(estimator = tree, param_distributions = param_grid_tree, n_iter = 100, random_state = 0)



start_time = time.time()

random_result = random_tree.fit(train, y)

# Summarize results

print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

print("Execution time: " + str((time.time() - start_time)) + ' ms')
tree = DecisionTreeRegressor(min_samples_split = 0.1, min_samples_leaf = 0.1, max_features = 117, max_depth = 5.0)

tree.fit(X_train, y_train)

print (tree.score(train, y))

score = rmsle_cv(tree)

print("Tree score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
train_dt = tree.predict(X_train)

test_dt = tree.predict(X_test)
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

sns.regplot(train_dt, (train_dt - y_train))

sns.regplot(test_dt, (test_dt - y_test))

plt.show()
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

sns.regplot(train_dt, y_train)

sns.regplot(test_dt, y_test)

plt.show()
train_plot1 = y_train.append(pd.Series(y_test))

plt.plot(np.arange(len(y)), tree.predict(train), label='Predicted')

plt.plot(np.arange(len(train_plot1)), train_plot1, label='Training')

plt.plot(np.arange(len(y_test))+len(y_train), y_test, label='Testing')

plt.legend()

plt.show()
result=train

result1= pd.DataFrame(tree.feature_importances_, result.columns)

result1.columns = ['Feature']

result1.sort_values(by='Feature',ascending=False).head()
features_coeff = pd.Series(tree.feature_importances_, index = train.columns)
used_features_tree= features_coeff[features_coeff != 0]

used_features_tree.sort_values(ascending=False)
len(used_features_tree)
not_used_features_tree= features_coeff[features_coeff==0]
len(not_used_features_tree)
used_features_tree.sort_values(ascending=False).plot(kind = "bar",figsize =(12,8), color = "purple", ec='black')

plt.title("Coefficients for the selected features in the Decision Tree Model")

plt.show()
random_forest = RandomizedSearchCV(estimator = forest, param_distributions = param_grid_forest, n_iter = 100, random_state = 0)



start_time = time.time()

random_result = random_forest.fit(train, y)

# Summarize results

print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

print("Execution time: " + str((time.time() - start_time)) + ' ms')
forest = RandomForestRegressor(n_estimators = 64, min_samples_split = 0.2, min_samples_leaf = 0.1, max_features = 48, max_depth = 30.0)

forest.fit(X_train, y_train)

print (forest.score(train, y))

score = rmsle_cv(forest)

print("Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
train_fo = forest.predict(X_train)

test_fo = forest.predict(X_test)
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

sns.regplot(train_fo, (train_fo - y_train))

sns.regplot(test_fo, (test_fo - y_test))

plt.show()
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

sns.regplot(train_fo, y_train)

sns.regplot(test_fo, y_test)

plt.show()
train_plot2 = y_train.append(pd.Series(y_test))

plt.plot(np.arange(len(y)), forest.predict(train), label='Predicted')

plt.plot(np.arange(len(train_plot2)), train_plot2, label='Training')

plt.plot(np.arange(len(y_test))+len(y_train), y_test, label='Testing')

plt.legend()

plt.show()
result1=train

result2= pd.DataFrame(forest.feature_importances_, result1.columns)

result2.columns = ['Feature']

result2.sort_values(by='Feature',ascending=False).head()
features_coeff1 = pd.Series(forest.feature_importances_, index = train.columns)

used_features_forest= features_coeff1[features_coeff1 != 0]

used_features_forest.sort_values(ascending=False)
len(used_features_forest)
not_used_features_forest= features_coeff1[features_coeff1==0]

len(not_used_features_forest)
used_features_forest.sort_values(ascending=False).plot(kind = "bar",figsize =(12,8), color = "blue", ec='black')

plt.title("Coefficients for the selected features in the Random Forest Model")

plt.show()
random1 = np.round(np.exp(forest.predict(test)))

#pd.DataFrame({'Id': test_ID, 'SalePrice': random}).to_csv('RandomForest.csv', index = False)
random_lasso = RandomizedSearchCV(estimator = lasso, param_distributions = param_grid_lasso, n_iter = 100, random_state = 0)



start_time = time.time()

random_result = random_lasso.fit(train, y)

# Summarize results

print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

print("Execution time: " + str((time.time() - start_time)) + ' ms')
lasso = Lasso(alpha=0.0004)

lasso.fit(train, y)

print (lasso.score(train, y))

score = rmsle_cv(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
train_la = lasso.predict(X_train)

test_la = lasso.predict(X_test)
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

sns.regplot(train_la, (train_la - y_train))

sns.regplot(test_la, (test_la - y_test))

plt.show()


plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

sns.regplot(train_la, y_train)

sns.regplot(test_la, y_test)

plt.show()
result3=train

result4= pd.DataFrame(lasso.coef_, result3.columns)

result4.columns = ['Feature']

result4.sort_values(by='Feature',ascending=False).head()
features_coeff2 = pd.Series(lasso.coef_, index = train.columns)

used_features_lasso= features_coeff2[features_coeff2 != 0]

used_features_lasso.sort_values(ascending=False)

len(used_features_lasso)
not_used_features_lasso= features_coeff2[features_coeff2==0]

len(not_used_features_lasso)
used_features_lasso.sort_values(ascending=False).plot(kind = "bar",figsize =(20,10), color = "orange", ec='black')

plt.title("Coefficients for the selected features in the Lasso Model")

plt.show()
Lasso1 = np.round(np.exp(lasso.predict(test)))

#pd.DataFrame({'Id': test_ID, 'SalePrice': Lasso1}).to_csv('Lasso2.csv', index = False)
random_en = RandomizedSearchCV(estimator = ENet, param_distributions = param_grid_en, n_iter = 100, random_state = 0)



start_time = time.time()

random_result = random_en.fit(train, y)

# Summarize results

print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

print("Execution time: " + str((time.time() - start_time)) + ' ms')
ENet = ElasticNet(alpha = 0.001)

ENet.fit(train, y)

print(ENet.score(train, y))

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
train_en = ENet.predict(X_train)

test_en = ENet.predict(X_test)
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

sns.regplot(train_en, (train_en - y_train))

sns.regplot(test_en, (test_en - y_test))

plt.show()
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

sns.regplot(train_en, y_train)

sns.regplot(test_en, y_test)

plt.show()
result_enet=train

result_EN= pd.DataFrame(ENet.coef_, result_enet.columns)

result_EN.columns = ['Feature']

result_EN.sort_values(by='Feature',ascending=False).head()
features_coeff_en = pd.Series(ENet.coef_, index = train.columns)

used_features_Enet= features_coeff_en[features_coeff_en != 0]

used_features_Enet.sort_values(ascending=False)

len(used_features_Enet)
not_used_features_Enet= features_coeff_en[features_coeff_en==0]

len(not_used_features_Enet)
used_features_Enet.sort_values(ascending=False).plot(kind = "bar",figsize =(12,8), color = "red", ec='black')

plt.title("Coefficients for the selected features in the ENet Model")

plt.show()
ENet1 = np.round(np.exp(ENet.predict(test)))

#pd.DataFrame({'Id': test_ID, 'SalePrice': ENet1}).to_csv('ENet.csv', index = False)
random_ridge = RandomizedSearchCV(estimator = ridge, param_distributions = param_grid_ridge, cv = 3, n_iter = 100, random_state = 0)

start_time = time.time()

random_result = random_ridge.fit(train, y)

# Summarize results

print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))

print("Execution time: " + str((time.time() - start_time)) + ' ms')
ridge = Ridge(alpha=10)

ridge.fit(train, y)

print (ridge.score(train, y))

score = rmsle_cv(ridge)

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
train_ridge = ridge.predict(X_train)

test_ridge = ridge.predict(X_test)
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

sns.regplot(train_ridge, (train_ridge - y_train))

sns.regplot(test_ridge, (test_ridge - y_test))

plt.show()
plt.subplots(figsize=(17, 8))

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

sns.regplot(train_ridge, y_train)

sns.regplot(test_ridge, y_test)

plt.show()
rid=train

rid1= pd.DataFrame(ridge.coef_, rid.columns)

rid1.columns = ['Feature']

rid1.sort_values(by='Feature',ascending=False).head()
features_coeff_rid= pd.Series(ridge.coef_, index = train.columns)

used_features_ridge= features_coeff_rid[features_coeff_rid != 0]

used_features_ridge.sort_values(ascending=False)

len(used_features_ridge)
not_used_features_ridge=features_coeff_rid[features_coeff_rid==0]

len(not_used_features_ridge)
ridge1 = np.round(np.exp(ridge.predict(test)))

#pd.DataFrame({'Id': test_ID, 'SalePrice': ridge1}).to_csv('ridge1.csv', index = False)
simple_avg5 = np.round((ridge1 + Lasso1 + ENet1)/3)

pd.DataFrame({'Id': test_ID, 'SalePrice': simple_avg5}).to_csv('simple_avg.csv', index = False)
Result_train_dict = {'Linear' : {'Train Variance' : metrics.explained_variance_score(y_trainlm, train_lm), 

                               'Train r2' : metrics.r2_score(y_trainlm, train_lm),

                               'Train MAE': metrics.mean_absolute_error(y_trainlm, train_lm),

                               'Train MSE': metrics.mean_squared_error(y_trainlm, train_lm),

                               'Train RMSE' : np.sqrt(metrics.mean_squared_error(y_trainlm, train_lm)),

                               'Train RMSLE' : rmsle_cv(reg_line).mean()}, 

                    'Decision Tree' : {'Train Variance' : metrics.explained_variance_score(y_train, train_dt), 

                               'Train r2' : metrics.r2_score(y_train, train_dt),

                               'Train MAE': metrics.mean_absolute_error(y_train, train_dt),

                               'Train MSE': metrics.mean_squared_error(y_train, train_dt),

                               'Train RMSE' : np.sqrt(metrics.mean_squared_error(y_train, train_dt)),

                               'Train RMSLE' : rmsle_cv(tree).mean()}, 

                    'Forest' : {'Train Variance' : metrics.explained_variance_score(y_train, train_fo), 

                               'Train r2' : metrics.r2_score(y_train, train_fo),

                               'Train MAE': metrics.mean_absolute_error(y_train, train_fo),

                               'Train MSE': metrics.mean_squared_error(y_train, train_fo),

                               'Train RMSE' : np.sqrt(metrics.mean_squared_error(y_train, train_fo)),

                               'Train RMSLE' : rmsle_cv(forest).mean()}, 

                    'Lasso' : {'Train Variance' : metrics.explained_variance_score(y_train, train_la), 

                               'Train r2' : metrics.r2_score(y_train, train_la),

                               'Train MAE': metrics.mean_absolute_error(y_train, train_la),

                               'Train MSE': metrics.mean_squared_error(y_train, train_la),

                               'Train RMSE' : np.sqrt(metrics.mean_squared_error(y_train, train_la)),

                               'Train RMSLE' : rmsle_cv(lasso).mean()}, 

                    'Elastic Net' : {'Train Variance' : metrics.explained_variance_score(y_train, train_en), 

                               'Train r2' : metrics.r2_score(y_train, train_en),

                               'Train MAE': metrics.mean_absolute_error(y_train, train_en),

                               'Train MSE': metrics.mean_squared_error(y_train, train_en),

                               'Train RMSE' : np.sqrt(metrics.mean_squared_error(y_train, train_en)),

                               'Train RMSLE' : rmsle_cv(ENet).mean()}, 

                    'Ridge' : {'Train Variance' : metrics.explained_variance_score(y_train, train_ridge), 

                               'Train r2' : metrics.r2_score(y_train, train_ridge),

                               'Train MAE': metrics.mean_absolute_error(y_train, train_ridge),

                               'Train MSE': metrics.mean_squared_error(y_train, train_ridge),

                               'Train RMSE' : np.sqrt(metrics.mean_squared_error(y_train, train_ridge)),

                               'Train RMSLE' : rmsle_cv(ridge).mean()}}
Result_test_dict = {'Linear' : {'Test Variance' : metrics.explained_variance_score(y_testlm, test_lm), 

                               'Test r2' : metrics.r2_score(y_testlm, test_lm),

                               'Test MAE': metrics.mean_absolute_error(y_testlm, test_lm),

                               'Test MSE': metrics.mean_squared_error(y_testlm, test_lm),

                               'Test RMSE' : np.sqrt(metrics.mean_squared_error(y_testlm, test_lm)),

                               'Test RMSLE' : rmsle_cv(reg_line).mean(),

                               'Test Kaggle' : 0.14935}, 

                    'Decision Tree' : {'Test Variance' : metrics.explained_variance_score(y_test, test_dt), 

                               'Test r2' : metrics.r2_score(y_test, test_dt),

                               'Test MAE': metrics.mean_absolute_error(y_test, test_dt),

                               'Test MSE': metrics.mean_squared_error(y_test, test_dt),

                               'Test RMSE' : np.sqrt(metrics.mean_squared_error(y_test, test_dt)),

                               'Test RMSLE' : rmsle_cv(tree).mean(),

                               'Test Kaggle' : 0.26043}, 

                    'Forest' : {'Test Variance' : metrics.explained_variance_score(y_test, test_fo), 

                               'Test r2' : metrics.r2_score(y_test, test_fo),

                               'Test MAE': metrics.mean_absolute_error(y_test, test_fo),

                               'Test MSE': metrics.mean_squared_error(y_test, test_fo),

                               'Test RMSE' : np.sqrt(metrics.mean_squared_error(y_test, test_fo)),

                               'Test RMSLE' : rmsle_cv(forest).mean(),

                               'Test Kaggle' : 0.23237}, 

                    'Lasso' : {'Test Variance' : metrics.explained_variance_score(y_test, test_la), 

                               'Test r2' : metrics.r2_score(y_test, test_la),

                               'Test MAE': metrics.mean_absolute_error(y_test, test_la),

                               'Test MSE': metrics.mean_squared_error(y_test, test_la),

                               'Test RMSE' : np.sqrt(metrics.mean_squared_error(y_test, test_la)),

                               'Test RMSLE' : rmsle_cv(lasso).mean(),

                               'Test Kaggle' : 0.11976}, 

                    'Elastic Net' : {'Test Variance' : metrics.explained_variance_score(y_test, test_en), 

                               'Test r2' : metrics.r2_score(y_test, test_en),

                               'Test MAE': metrics.mean_absolute_error(y_test, test_en),

                               'Test MSE': metrics.mean_squared_error(y_test, test_en),

                               'Test RMSE' : np.sqrt(metrics.mean_squared_error(y_test, test_en)),

                               'Test RMSLE' : rmsle_cv(ENet).mean(),

                                'Test Kaggle' : 0.11955}, 

                    'Ridge' : {'Test Variance' : metrics.explained_variance_score(y_test, test_ridge), 

                               'Test r2' : metrics.r2_score(y_test, test_ridge),

                               'Test MAE': metrics.mean_absolute_error(y_test, test_ridge),

                               'Test MSE': metrics.mean_squared_error(y_test, test_ridge),

                               'Test RMSE' : np.sqrt(metrics.mean_squared_error(y_test, test_ridge)),

                               'Test RMSLE' : rmsle_cv(ridge).mean(),

                               'Test Kaggle' : 0.12031}}
Final_Table = pd.concat([pd.DataFrame(Result_train_dict), pd.DataFrame(Result_test_dict)], axis=0)
Kaggle_scores = {'Linear' : { 'Linear' : 0.14935}, 

                    'Decision Tree' : {'Tree' : 0.26043}, 

                    'Forest' : {'Forest' : 0.23237}, 

                    'Lasso' : {'Lasso' : 0.11976}, 

                    'Elastic Net' : {'ENet' : 0.11955}, 

                    'Ridge' : {'Ridge' : 0.12031},

                    'Lasso + ridge + ENet + Linear' :{'4 Way Average' : 0.12702},

                    'Lasso + ridge + ENet' : {'3 Way Average' : 0.11936}}
pd.DataFrame(Kaggle_scores).plot(kind='bar', figsize = (18, 8))

plt.show()
Final_Table.plot(kind='bar', figsize = (18, 8))

plt.show()