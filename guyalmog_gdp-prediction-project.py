import os
print(os.listdir("../input/countries-of-the-world"))
print(os.listdir("../input/additional-data"))
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
dfc = pd.read_csv("../input/countries-of-the-world/countries_of_the_world.csv", decimal=',')
df_WB = pd.read_csv("../input/additional-data/additional_data_WB.csv")
df_rel = pd.read_csv("../input/additional-data/additional_data_religion.csv")
df_freedom = pd.read_csv("../input/additional-data/additional_data_freedom.csv")

### changing the column name to fit the other data
dfc.rename(columns = {'Country':'Country Name'}, inplace=True)

### removing the space after each country name
dfc['Country Name'].values
dfc['Country Name'] = dfc['Country Name'].apply(lambda x: x.split(" ")[0] if len(x.split())==1 else x)
### merging the data with the WorldBank data
df_n_wb = dfc.merge(df_WB, on = 'Country Name', how='left')
df_n_wb.head()
### removing the space in the country name column of the religion data
df_rel['Country Name'] = df_rel['Country Name'].apply(lambda x: x.strip())

#merging the data with the religion data
df_n_wb_rel = df_n_wb.merge(df_rel, on = 'Country Name', how = 'left' )
df_n_wb_rel.drop(['Unnamed: 1'],axis=1, inplace=True)
df_n_wb_rel['Country Name'].values
### merging the data with the freedom index data
df = df_n_wb_rel.merge(df_freedom, on='Country Name', how='inner')

### rename some columns to be more comfortable to work with:

df.rename(columns = {'GDP ($ per capita)':'GDP', 'Country Name':'Country','Imports of goods and services  2013':'Imports', 'Exports of goods and services  2013':'Exports'
                    ,'Foreign direct investment, net inflows 2013': 'Net foreign invest'}, inplace=True)
df.Region = [x.strip() for x in df.Region]
df = df.rename(columns = lambda x: x.split('(')[0].strip())
df.shape
df.Country = df.Country.astype('category')
df.Region = df.Region.astype('category')
df['Main Religion'] = df['Main Religion'].astype('str')
df.info()

df['Main Religion'].value_counts()
df['Main Religion'] = df['Main Religion'].replace('Christian ', 'Christian').replace('Christian (Free Wesleyan Church claims over 30','Christian')
df['Main Religion'] = df['Main Religion'].replace('Muslim*', 'Muslim')
df['Main Religion'] = df['Main Religion'].replace('Buddist', 'Buddhist').replace('Buddhist', 'Buddhist').replace('BuddhismAnd','Buddhist')
df['Main Religion'] = df['Main Religion'].replace('Buddhist ', 'Buddhist')
df['Main Religion'] = df['Main Religion'].replace('syncretic (part Christian', 'Christian')
df['Main Religion'] = df['Main Religion'].replace('Zionist 40% (a blend of Christianity and indigenous ancestral worship)', 'Zionist')

### presenting the number of null values by total and percentage:

total = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
percent = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df))
data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data
### Let's have a look at the variable Climate for it is a categorial one.
## let's find out what can fit the most to our null value for climate in each country

print(pd.pivot_table(df,index=['Climate'],values = 'Region', aggfunc='sum'))
print(df[df.Climate.isnull()]['Country'])
df.loc[3,['Climate']] = 3
df.loc[25,['Climate']] = 3
df.loc[31,['Climate']] =3
df.loc[64,['Climate']] =3
df.loc[74,['Climate']] =1
df.loc[77,['Climate']] =2.5
df.loc[79,['Climate']] =3
df.loc[80,['Climate']] =3
df.loc[88,['Climate']] =3
df.loc[92,['Climate']] =3
df.loc[94,['Climate']] =1
df.loc[97,['Climate']] =2
df.loc[113,['Climate']] =1
df.loc[117,['Climate']] =3
df.loc[121,['Climate']] =3
df.loc[132,['Climate']] =3
df.Climate.isnull().sum()

df.drop(['Government expenditure on education % of GDP 2014'], axis=1, inplace=True)
total = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
percent = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df))
data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data
total = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
percent = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df))
data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data
%matplotlib inline
plt.figure(figsize = (10,3))
plt.subplot(1,3,1)
sns.distplot(df['ATMs per 100,000 adults 2013'])
plt.subplot(1,3,2)
sns.kdeplot(df['Exports'])
plt.subplot(1,3,3)
sns.kdeplot(df['Net foreign invest'])
plt.tight_layout()
columns = data.index[:16]
for column in columns:
    df[column] = df[column].fillna(df.groupby('Region')[column].transform('mean'))

total = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
percent = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df))
data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data
corrs = df.corr()
fig, ax = plt.subplots(figsize = (12,12))
heatcor = sns.heatmap(corrs, cbar=True ,ax=ax).set(title = 'Correlation Map', xlabel = 'Columns', ylabel = 'Columns' )
top_corrs = df.corr().nlargest(10, 'GDP').index
cm = np.corrcoef(df[top_corrs].values.T)
heatcor = sns.heatmap(cm, cbar = True, annot = True, cmap='BrBG', yticklabels = top_corrs.values, xticklabels=top_corrs.values)
df = df.drop(['Imports'], axis=1)

top_corrs = df.corr().nlargest(8, 'GDP').index
sns.pairplot(df[top_corrs])
fig = plt.figure(figsize = (15,10))
plt.subplot(2,3,1)
sns.barplot(df['GDP'], df['Region'],palette='BrBG',ci = None )
plt.subplot(2,3,2)
sns.barplot(df['Exports'], df['Region'],palette='BrBG', ci = None)
plt.subplot(2, 3, 3)
sns.barplot(df['Net migration'], df['Region'], palette='BrBG', ci = None)
plt.subplot(2, 3, 4)
sns.barplot(df['Pop. Density'], df['Region'],palette='BrBG', ci = None)
plt.subplot(2, 3, 5)
sns.barplot(df['Deathrate'], df['Region'],palette='BrBG', ci = None)
plt.subplot(2, 3, 6)
sns.countplot(y = df['Region'],palette='BrBG')
plt.tight_layout()
fig = plt.figure(figsize = (18,10))
plt.subplot(2,4,1)
sns.barplot(df['GDP'], df['Main Religion'],palette='BrBG',ci = None)
plt.subplot(2,4,2)
sns.barplot(df['Exports'], df['Main Religion'],palette='BrBG', ci = None)
plt.subplot(2, 4, 3)
sns.barplot(df['Net migration'], df['Main Religion'], palette='BrBG', ci = None)
plt.subplot(2, 4, 5)
sns.barplot(df['Pop. Density'], df['Main Religion'],palette='BrBG', ci = None)
plt.subplot(2, 4, 6)
sns.barplot(df['Deathrate'], df['Main Religion'],palette='BrBG', ci = None)
plt.subplot(2, 4, 7)
sns.countplot(y = df['Main Religion'],palette='BrBG')
plt.tight_layout()

GDP_scaled = StandardScaler().fit_transform(df['GDP'][:,np.newaxis])
low_values = GDP_scaled[GDP_scaled[:,0].argsort()][:10]
high_values = GDP_scaled[GDP_scaled[:,0].argsort()][-10:]
print('the lower values of the distribution are:')
print(low_values)
print('the higher values of the distribution are:')
print(high_values)
df[df['GDP'].values == df.GDP.max()]
df_final = pd.get_dummies(df, columns=['Region', 'Main Religion', 'Climate'])

df_final.GDP.describe()
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(df_final['GDP'], fit=norm)
plt.subplot(1,2,2)
res = stats.probplot(df_final['GDP'], plot=plt)
plt.tight_layout()
y = df_final.GDP.values
y = np.log(y)
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(y, fit=norm)
plt.subplot(1,2,2)
res = stats.probplot(y, plot=plt)
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV , RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
df_final1 = df_final.copy()
df_final1.columns
x_all = df_final1.drop(['Country','GDP'], axis=1)
x_features = df_final1[['Population', 'Pop. Density','Literacy','Infant mortality','Birthrate',
                      'ATMs per 100,000 adults 2013', 'Financial Freedom', 'Business Freedom',
                      'Phones','Net foreign invest', 'Service', 'Industry','Exports', 'Investment Freedom' ]]
x_all_scaled = StandardScaler().fit_transform(x_all)
x_features_scaled = StandardScaler().fit_transform(x_features)
x_all_train, x_all_test, y_train, y_test = train_test_split(x_all.values, y, train_size = 0.8)
x_features_train, x_features_test, y_train, y_test = train_test_split(x_features.values, y, train_size = 0.8)
x_all_scaled_train, x_all_scaled_test, y_train, y_test = train_test_split(x_all_scaled, y, train_size = 0.8)
x_features_scaled_train, x_features_scaled_test, y_train, y_test = train_test_split(x_features_scaled, y, train_size = 0.8)
models = []
models.append(('Lasso', Lasso()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('XGB', xgb.XGBRegressor(objective = 'reg:squarederror')))
models.append(('LR', LinearRegression()))
models.append(('SVR', SVR()))
models.append(('Enet',ElasticNet(tol=0.5)))
models.append(('LightGBM',lgb.LGBMRegressor()))
names = []
results = []
n_fold = 6
for name, model in models:
    kfold = KFold(n_fold, random_state = None)
    cv_results = cross_val_score(model,x_all_train,y_train, scoring = 'neg_mean_squared_error' )
    results.append(cv_results)
    names.append(name)
    model.fit(x_all_train, y_train)
    predictions = model.predict(x_all_test)
    r2_msg = 'And the R2 score is: %f' %(r2_score(predictions, y_test))
    rmse_msg = 'For ' '%s' ' , the mean squared error is: ' '%f(%f)' % (name, cv_results.mean(), cv_results.std())
    print(rmse_msg)
    print(r2_msg)
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
names = []
results = []
n_fold = 6
for name, model in models:
    kfold = KFold(n_fold, random_state = None)
    cv_results = cross_val_score(model,x_all_scaled_train,y_train, scoring = 'neg_mean_squared_error' )
    results.append(cv_results)
    names.append(name)
    model.fit(x_all_scaled_train, y_train)
    predictions = model.predict(x_all_scaled_test)
    r2_msg = 'And the R2 score is: %f' %(r2_score(predictions, y_test))
    rmse_msg = 'For ' '%s' ' , the mean squared error is: ' '%f(%f)' % (name, cv_results.mean(), cv_results.std())
    print(rmse_msg)
    print(r2_msg)
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
names = []
results = []
n_fold = 6
for name, model in models:
    kfold = KFold(n_fold, random_state = None)
    cv_results = cross_val_score(model,x_features_train,y_train, scoring = 'neg_mean_squared_error' )
    results.append(cv_results)
    names.append(name)
    model.fit(x_features_train, y_train)
    predictions = model.predict(x_features_test)
    r2_msg = 'And the R2 score is: %f' %(r2_score(predictions, y_test))
    rmse_msg = 'For ' '%s' ' , the mean squared error is: ' '%f(%f)' % (name, cv_results.mean(), cv_results.std())
    print(rmse_msg)
    print(r2_msg)
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

names = []
results = []
n_fold = 6
for name, model in models:
    kfold = KFold(n_fold, random_state = None)
    cv_results = (cross_val_score(model,x_features_scaled_train,y_train, scoring = 'neg_mean_squared_error' ))
    results.append(cv_results)
    names.append(name)
    model.fit(x_features_scaled_train, y_train)
    predictions = model.predict(x_features_scaled_test)
    r2_msg = 'And the R2 score is: %f' %(r2_score(predictions, y_test))
    rmse_msg = 'For ' '%s' ' , the mean squared error is: ' '%f(%f)' % (name, cv_results.mean(), cv_results.std())
    print(rmse_msg)
    print(r2_msg)

fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
model_xgb = xgb.XGBRegressor(objective = 'reg:squarederror')
model_randomforest = RandomForestRegressor()

model_randomforest.fit(x_features_scaled_train, y_train)
RF_predictions = model_randomforest.predict(x_features_scaled_test)
print(r2_score(RF_predictions, y_test))
print(mean_squared_error(RF_predictions, y_test))
graph = sns.regplot(RF_predictions, y_test).set(title='Random Forest model predictions', xlabel='Predicted GDP', ylabel='Actual GDP')
###performing Search grid search
parameters = { 
                      'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
                                                 }
gridRF = GridSearchCV(model_randomforest,param_grid= parameters, n_jobs = 4, cv=5)
gridRF.fit(x_features_scaled_train, y_train)
gridRF.best_estimator_
gridRF_preds = gridRF.predict(x_features_scaled_test)
print(mean_squared_error(gridRF_preds, y_test))
print(r2_score(gridRF_preds, y_test))
graph = sns.regplot(gridRF_preds, y_test).set(title='Random Forest model predictions', xlabel='Predicted GDP', ylabel='Actual GDP')
model_xgb.fit(x_features_scaled_train, y_train)
xgb_predictions = model_xgb.predict(x_features_scaled_test)
print(mean_squared_error(xgb_predictions, y_test))
print(r2_score(xgb_predictions, y_test))
graph = sns.regplot(xgb_predictions, y_test).set(title='XGB model predictions', xlabel='Predicted GDP', ylabel='Actual GDP')
###performing Search grid search
n_estimators = [100,300, 400, 500]
learning_rate = [0.03,0.09, 0.1, 0.13, 0.2]
max_depth = [3,4,5,10]
min_child_weight = [1,2,3,4]
parameters = { 
                      'objective':['reg:linear'],
                      'learning_rate': learning_rate, 
                      'max_depth': max_depth,
                      'min_child_weight': min_child_weight,
                      'silent': [1],
                      'subsample': [0.5, 0.6, 0.8],
                      'n_estimators': n_estimators,
                      'booster': ['gbtree']
                                                 }
gridXGB = GridSearchCV(model_xgb,param_grid= parameters, n_jobs = 4, cv=5)
gridXGB.fit(x_features_scaled_train, y_train)

gridXGB_preds = gridXGB.predict(x_features_scaled_test)
print(mean_squared_error(gridXGB_preds, y_test))
print(r2_score(gridXGB_preds, y_test))
sns.regplot(gridXGB_preds, y_test)
model_xgb_2 = xgb.XGBRegressor(booster = 'gbtree',
 learning_rate = 0.19,
 max_depth= 5,
 min_child_weight= 2,
 n_estimators= 90,
 objective= 'reg:linear',
 silent= 1,
 subsample= 0.6)
model_xgb_2.fit(x_features_scaled_train, y_train)
xgb_pred_2 = model_xgb_2.predict(x_features_scaled_test)
print(mean_squared_error(xgb_pred_2, y_test))
print(r2_score(xgb_pred_2, y_test))
graph = sns.regplot(xgb_pred_2, y_test).set(title='XGB model predictions', xlabel='Predicted GDP', ylabel='Actual GDP')
featuers_coefficients = model_xgb_2.feature_importances_.tolist()
feature_names = x_features.columns
for i in range(len(feature_names)):
    coefs = 'The coefficient for the feature %s is: ' '%f' %(feature_names[i], featuers_coefficients[i])
    print(coefs)


feats = pd.DataFrame(pd.Series(featuers_coefficients, feature_names).sort_values(ascending=False),columns=['Coefficient'])
feats