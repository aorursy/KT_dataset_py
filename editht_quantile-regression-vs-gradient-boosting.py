import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy.stats import spearmanr

from statsmodels.graphics.gofplots import qqplot

plt.style.use('bmh')

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.decomposition import PCA

import statsmodels.formula.api as sm

from statsmodels.regression.quantile_regression import QuantReg

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import KFold, train_test_split

from sklearn.ensemble import GradientBoostingRegressor

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_test = pd.read_csv('../input/test.csv')

set(df_train.columns).difference(df_test.columns)
df_train['IsTrain'] = 1

df_test['IsTrain']=0

df = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)
df.groupby('IsTrain')['IsTrain'].count()
df_IDs = df[['Id','IsTrain']]

del df['Id']
VarCatCnt = df[:].nunique()

df_uniq = pd.DataFrame({'VarName': VarCatCnt.index,'cnt': VarCatCnt.values})

df_uniq.head()
df_contVars = df_uniq[(df_uniq.cnt > 30) | (df_uniq.VarName == 'SalePrice') | (df_uniq.VarName == 'IsTrain')]

df_num = df[df_contVars.VarName]

df_num.shape
df_categVars = df_uniq[(df_uniq.cnt <= 30) | (df_uniq.VarName == 'SalePrice') | (df_uniq.VarName == 'IsTrain')]

df_categ = df[df_categVars.VarName]

df_categ.shape
for i in range(0, len(df_num.columns)-1):

    if (df_num[df_num.columns[i]].isnull().sum() != 0) & (df_num.columns[i] != 'SalePrice'):

        print(df_num.columns[i] + " Null Count:" + str(df_num[df_num.columns[i]].isnull().sum()))

        df_num[df_num.columns[i]] = df_num[df_num.columns[i]].fillna(0)
for i in range(0, len(df_categ.columns)-1):

    if (df_categ[df_categ.columns[i]].dtype == object) & (df_categ[df_categ.columns[i]].isnull().sum() != 0):

            #print(df_categ.columns[i] + " Null Count:" + str(df_categ[df_categ.columns[i]].isnull().sum()))

            df_categ[df_categ.columns[i]].replace(np.nan, 'NA', inplace= True)
df_categ_N = df_categ.select_dtypes(include = ['float64', 'int64'])

df_categ_N.info()
df_categ['BsmtFullBath'].replace(np.nan, 0, inplace= True)

df_categ['BsmtHalfBath'].replace(np.nan, 0, inplace= True)
c = 0

len_c = 3 # (len(df_categ.columns)-2)

fig, axes = plt.subplots(len_c, 2, figsize=(10, 13))     # fig height = 70 -> in figsize(width,height)

for i, ax in enumerate(fig.axes):

    if (c < len_c) & (i % 2 == 0):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

        sns.countplot(x=df_categ.columns[c], alpha=0.7, data=df_categ, ax=ax)



    if (c < len_c) & (i % 2 != 0):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

        sns.boxplot(data = df_categ, x=df_categ.columns[c], y='SalePrice', ax=ax)

        c = c + 1

fig.tight_layout()
df_num.insert(loc=20, column='OverallQual', value=df_categ[['OverallQual']])

df_num.insert(loc=20, column='OverallCond', value=df_categ[['OverallCond']])

df_num.insert(loc=20, column='YrSold', value=df_categ[['YrSold']])

df_num.insert(loc=20, column='TotRmsAbvGrd', value=df_categ[['TotRmsAbvGrd']])

df_num.insert(loc=20, column='Fireplaces', value=df_categ[['Fireplaces']])

df_num.insert(loc=20, column='GarageCars', value=df_categ[['GarageCars']])
df_num['GarageCars'].fillna(0, inplace=True)
df_categ.drop(['GarageCars'

                ,'Fireplaces'

                ,'TotRmsAbvGrd'

                ,'YrSold'

                ,'OverallQual'

                ,'OverallCond']

            , axis=1, inplace = True)
CatVarQual = ['ExterQual','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual']

map_dict = {'Ex': 5,

            'Gd': 4,

            'TA': 3,

            'Fa': 2,

            'Po': 1,

            'NA': 3}



df_categQ = pd.DataFrame()

for i in range(0, len(CatVarQual)):

    df_categQ[CatVarQual[i]+'_N'] = df_categ[CatVarQual[i]].map(map_dict)
CatVarList = [column for column in df_categ if (column not in set(CatVarQual))]
df_categN = df_categ

for i in range(0, len(CatVarList)-2):

    catVar = CatVarList[i]  #catVar = 'MSSubClass'

    cl = df_categ.groupby(catVar)['SalePrice'].median().sort_values()

    df_cl = pd.DataFrame({'Category': cl.index,'SortVal': cl.values})

    df_cl.replace(np.nan, df_categ['SalePrice'].median(), inplace= True)

    df_cl[catVar+'_N']=df_cl['SortVal']/10000

    #df_cl[catVar+'_N']=df_cl['SortVal'].rank()

    #print(df_cl) #if want to see how the categories got ranked

    df_categN = pd.merge(df_categN,

                        df_cl[['Category', catVar+'_N','SortVal']],

                        left_on=catVar,

                        right_on='Category',

                        how = 'left')

    df_categN.drop(['Category','SortVal',catVar], axis=1, inplace = True)

df_categN.drop(CatVarQual, axis=1, inplace = True)

#df_categN.columns
sns.pairplot(data=df_categ,

            x_vars='Neighborhood',

            y_vars=['SalePrice'],

            size = 6)

plt.xticks(rotation=45);
sns.pairplot(data=df_categN,

            x_vars='Neighborhood_N',

            y_vars=['SalePrice'],

            size = 6);
print("Before category encoding:")

df_n = df_categ[df_categ['IsTrain'] == 1]

print(spearmanr(df_n['SalePrice'],df_n['Neighborhood']))



print("After category encoding:")

df_n = df_categN[df_categN['IsTrain'] == 1]

print(spearmanr(df_n['SalePrice'],df_n['Neighborhood_N']))

print("Pearson corr = "+ str(df_n.corr(method='pearson')['SalePrice']['Neighborhood_N']))
df_categN = pd.merge(df_categN, df_categQ, left_index=True, right_index=True, sort=False)

#df_categN.columns
df = pd.merge(df_categN[df_categN.columns[2:]], df_num, left_index=True, right_index=True, sort=False)
df_num['SalePrice'].describe()
#skewness and kurtosis

print("Skewness: %f" % df['SalePrice'].skew())

print("Kurtosis: %f" % df['SalePrice'].kurt())
sns.distplot(df[df['SalePrice'].isnull() == False]['SalePrice'],fit=norm);
qqplot(df[df['SalePrice'].isnull() == False]['SalePrice'], line='s');
len_c = 4   #(len(df_num.columns)-2)

fig, axes = plt.subplots(round(len_c / 2), 2, figsize=(12, 10))     # fig height = 70 -> in figsize(width,height)

for i, ax in enumerate(fig.axes):

    if (i < len_c):

        sns.distplot(df_num[df_num['IsTrain']==1][df_num.columns[i]], label="IsTrain = 1", fit=norm, ax=ax)



fig.tight_layout()
# Scatterplots SalePrice vs. numeric Vars

sns.pairplot(data=df_num,

            x_vars=df_num.columns[:4],

            y_vars=['SalePrice']);
df_corr = df.corr(method='pearson')['SalePrice'][:-2]   

golden_features_list = df_corr[abs(df_corr) > 0.5].sort_values(ascending=False)

print("There is {} correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))

#df[golden_features_list.index].head()
df.insert(loc=0, column='TotArea', value=(df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']) )
df.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea'], axis=1, inplace = True)
df_corr = df.corr(method='pearson')['SalePrice'][:-2]  

golden_features_list = df_corr[abs(df_corr) > 0.3].sort_values(ascending=False)

Top_features_list = df_corr[abs(df_corr) > 0.5].sort_values(ascending=False)

#print("There is {} correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
#correlation matrix heatmap

corrmat = df[Top_features_list.index].corr()

f, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(corrmat, cmap="RdBu", vmin=-1, vmax=1, square=True, annot=True, fmt=".1f");
# Separating out the features (and Training sample from Testing)

X = df.loc[:1459, golden_features_list.index].values



# Separating out the target

y = df.iloc[:1460,-2].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()

y_trainS = sc_Y.fit_transform(y_train.reshape(-1,1))

y_testS = sc_Y.transform(y_test.reshape(-1,1))
print("mean = " + str(np.mean(X_train[:,4])))

print("std = " + str(np.std(X_train[:,4])))
pca = PCA(n_components = 5)

principalComponents = pca.fit_transform(X_train)

principalComponentsTest = pca.transform(X_test)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['PrincComp_1', 'PrincComp_2','PrincComp_3','PrincComp_4','PrincComp_5'])

principalDftest = pd.DataFrame(data = principalComponentsTest

             , columns = ['PrincComp_1', 'PrincComp_2','PrincComp_3','PrincComp_4','PrincComp_5'])
print('Variance explained by all components: ' + str(pca.explained_variance_ratio_.sum()))

pca.explained_variance_ratio_
compareDf = pd.concat((principalDf, pd.DataFrame(X_train, columns = golden_features_list.index)), axis=1)
corrmat = compareDf.corr()['PrincComp_1':'PrincComp_5']

f, ax = plt.subplots(figsize=(20, 5))

sns.heatmap(corrmat, cmap="RdBu", vmin=-1, vmax=1, square=False, annot=True, fmt=".1f");
principalDf['SalePrice'] = y_trainS
mod = sm.quantreg('SalePrice ~ PrincComp_1 + PrincComp_2 + PrincComp_3 + PrincComp_4 + PrincComp_5', principalDf)

res = mod.fit(q=.5)

print(res.summary())
pred = res.predict(principalDftest) # make the predictions by the model

y_pred = sc_Y.inverse_transform(pred)
# Plot the y_test and the prediction (y_pred)

fig = plt.figure(figsize=(15, 5))

plt.plot(np.arange(0,len(y_test),1), y_test, 'b.', markersize=10, label='Actual')

plt.plot(np.arange(0,len(y_test),1), y_pred, 'r-', label='Prediction', alpha =0.5)

plt.xlabel('Obs')

plt.ylabel('SalePrice')

#plt.ylim(-10, 20)

plt.legend(loc='upper right');
DFyy = pd.DataFrame({'y_test':y_test,'y_pred': y_pred})

DFyy.sort_values(by=['y_test'],inplace=True)

plt.plot(np.arange(0,len(DFyy),1), DFyy['y_pred'])

plt.plot(np.arange(0,len(DFyy),1), DFyy['y_test'], alpha=0.5)

#plt.ylim(0,500000)

plt.ylabel('Red= y_test,  Blue = y_pred')

plt.xlabel('Index ')

plt.title('Predicted vs. Real');

print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');
plt.plot(np.arange(0,len(DFyy),1), DFyy['y_pred']/DFyy['y_test'])

plt.ylabel('Ratio = pred/real')

plt.xlabel('Index')

plt.title('Ratio of Predicted vs. Real (1=excellent Prediction)');

print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');
plt.scatter(y_test, y_pred)

plt.ylim(-1, 500000)

plt.xlim(-1, 500000)

plt.plot(y_test, y_test, "r")

plt.xlabel('y_actual')

plt.ylabel('y_predicted');
plt.scatter(np.arange(0,len(DFyy),1), (DFyy['y_test'] - DFyy['y_pred'])/DFyy['y_test'] )

plt.ylim(-0.75,0.75)

plt.ylabel('Relative Error = (real - pred)/real')

plt.xlabel('Index')

plt.title('Relative Error in Testing sample');

print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');
# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

def rmsle(real, predicted):

    sum=0.0

    for x in range(len(predicted)):

        if predicted[x]<0 or real[x]<0: #check for negative

            print('Warning:1 negative value skipped')

            continue

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted))**0.5
print('Prediction accuracy on Testing Sample:')

print('RMSLE       = %f' % (rmsle(y_test, y_pred)))

print('r-squared   = %f' % (r2_score(y_test, y_pred)))
pred_train = res.predict(principalDf)

y_pred_train = sc_Y.inverse_transform(pred_train)
print('Model accuracy on Training Sample:')

print('RMSLE       = %f' % (rmsle(y_train, y_pred_train)))

print('r-squared   = %f' % (r2_score(y_train, y_pred_train)))
skf = KFold(n_splits=3, shuffle=True, random_state=123)



i = 1

Intercept_CV   = list()

PrincComp_1_CV = list()

PrincComp_2_CV = list()

PrincComp_3_CV = list()

PrincComp_4_CV = list()

PrincComp_5_CV = list()



for train_index, test_index in skf.split(X,y):

    #------ Standardize -------

    sc_X = StandardScaler()

    X_train = sc_X.fit_transform(X[train_index])

    X_test = sc_X.transform(X[test_index])

    sc_Y = StandardScaler()

    y_trainS = sc_Y.fit_transform(y[train_index].reshape(-1,1))

    y_test = y[test_index]

    #----- PCA ----------------

    pca = PCA(n_components = 5)

    principalComponents = pca.fit_transform(X_train)

    principalComponentsTest = pca.transform(X_test)

    

    #---- Applying the Standard scaler so the parameters are easier to compare

    ScPCA = StandardScaler()

    principalComponents = ScPCA.fit_transform(principalComponents)

    principalComponentsTest = ScPCA.transform(principalComponentsTest)

    principalDf = pd.DataFrame(data = principalComponents

             , columns = ['PrincComp_1', 'PrincComp_2','PrincComp_3','PrincComp_4','PrincComp_5'])

    principalDftest = pd.DataFrame(data = principalComponentsTest

             , columns = ['PrincComp_1', 'PrincComp_2','PrincComp_3','PrincComp_4','PrincComp_5'])

    principalDf['SalePrice'] = y_trainS

    

    #principalDf = StandardScaler().fit_transform(principalDf)

    

    #----- QUANTILE REGRESSION ----------

    mod = sm.quantreg('SalePrice ~ PrincComp_1 + PrincComp_2 + PrincComp_3 + PrincComp_4 + PrincComp_5', principalDf)

    res = mod.fit(q=.5)

    pred = res.predict(principalDftest) # make the predictions by the model

    y_pred = sc_Y.inverse_transform(pred)

    print('Split No.: ' + str(i))

    print('RMSLE       = %f' % (rmsle(y_test, y_pred)))

    print('r-squared   = %f' % (r2_score(y_test, y_pred)))

    

    Intercept_CV.append(res.params['Intercept'])

    PrincComp_1_CV.append(res.params['PrincComp_1'])

    PrincComp_2_CV.append(res.params['PrincComp_2'])

    PrincComp_3_CV.append(res.params['PrincComp_3'])

    PrincComp_4_CV.append(res.params['PrincComp_4'])

    PrincComp_5_CV.append(res.params['PrincComp_5'])

    i=i+1



#--- Model Stability Check: Mean and standard deviation of each regression parameter

print('Intercept mean = ' + str(np.mean(Intercept_CV)) + ' std = ' + str(np.std(Intercept_CV)))

print('PrincComp_1 mean = ' + str(np.mean(PrincComp_1_CV)) + ' std = ' + str(np.std(PrincComp_1_CV)))

print('PrincComp_2 mean = ' + str(np.mean(PrincComp_2_CV)) + ' std = ' + str(np.std(PrincComp_2_CV)))

print('PrincComp_3 mean = ' + str(np.mean(PrincComp_3_CV)) + ' std = ' + str(np.std(PrincComp_3_CV)))

print('PrincComp_4 mean = ' + str(np.mean(PrincComp_4_CV)) + ' std = ' + str(np.std(PrincComp_4_CV)))

print('PrincComp_5 mean = ' + str(np.mean(PrincComp_5_CV)) + ' std = ' + str(np.std(PrincComp_5_CV)))
# Separating out the features (and Training sample from Testing)

X = df.iloc[:1460, :-2].values



# Separating out the target

y = df.iloc[:1460,-2].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
GBRmedian = GradientBoostingRegressor(loss='quantile', alpha=0.5,

                                n_estimators=250, max_depth=5,

                                learning_rate=.1, min_samples_leaf=10,

                                min_samples_split=20)

GBRmedian.fit(X_train, y_train);
# Make the prediction on the Testing sample

y_pred = GBRmedian.predict(X_test)

y_pred_train = GBRmedian.predict(X_train)
print('Model Accuracy on Training sample:')

print('RMSLE       = %f' % (rmsle(y_train, y_pred_train)))

print('r-squared   = %f' % (r2_score(y_train, y_pred_train)))
print('Accuracy of prediction on Testing sample:')

print('RMSLE       = %f' % (rmsle(y_test, y_pred)))

print('r-squared   = %f' % (r2_score(y_test, y_pred)))
# Plot feature importance

feature_importance = GBRmedian.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

# Let's plot to top10 most important variables

sorted_idx = sorted_idx[-10:]

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, df.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance');
# Plot the y_test and the prediction (y_pred)

fig = plt.figure(figsize=(15, 5))

plt.plot(np.arange(0,len(y_test),1), y_test, 'b.', markersize=10, label='Actual')

plt.plot(np.arange(0,len(y_test),1), y_pred, 'r-', label='Prediction', alpha = 0.5)

plt.xlabel('Obs')

plt.ylabel('SalePrice')

#plt.ylim(-10, 20)

plt.legend(loc='upper right');
DFyy = pd.DataFrame({'y_test':y_test,'y_pred': y_pred})

DFyy.sort_values(by=['y_test'],inplace=True)

plt.plot(np.arange(0,len(DFyy),1), DFyy['y_pred'])

plt.plot(np.arange(0,len(DFyy),1), DFyy['y_test'], alpha=0.5)

#plt.ylim(0,500000)

plt.ylabel('Red= y_test,  Blue = y_pred')

plt.xlabel('Index ')

plt.title('Predicted vs. Real');

print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');
plt.plot(np.arange(0,len(DFyy),1), DFyy['y_pred']/DFyy['y_test'])

plt.ylabel('Ratio = pred/real')

plt.xlabel('Index')

plt.ylim(0,2)

plt.title('Ratio Predicted vs. Real (1=excellent Prediction)');

print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');
plt.scatter(y_test, y_pred)

plt.ylim(-1, 500000)

plt.xlim(-1, 500000)

plt.plot(y_test, y_test, "r")

plt.xlabel('y_actual')

plt.ylabel('y_predicted');
plt.scatter(np.arange(0,len(DFyy),1), (DFyy['y_test'] - DFyy['y_pred'])/DFyy['y_test'] )

plt.ylim(-0.75,0.75)

plt.ylabel('Relative Error = (real - pred)/real')

plt.xlabel('Index')

plt.title('Ratio Predicted vs. Real (1=perfectPrediction)');

print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');
skf = KFold(n_splits=3, shuffle=True, random_state=123)



for train_index, test_index in skf.split(X,y):

    #------ Standardize -------

    sc_X = StandardScaler()

    X_train = sc_X.fit_transform(X[train_index])

    X_test = sc_X.transform(X[test_index])

    y_train = y[train_index]

    y_test = y[test_index]

    

    #----- Gradient Boosted Regression ----------

    GBRmedian = GradientBoostingRegressor(loss='quantile', alpha=0.5,

                                    n_estimators=250, max_depth=5,

                                    learning_rate=.1, min_samples_leaf=10,

                                    min_samples_split=20)

    GBRmedian.fit(X_train, y_train)

    # Make the prediction on the Testing sample

    y_pred = GBRmedian.predict(X_test)

    y_pred_train = GBRmedian.predict(X_train)



    print('Accuracy of prediction on Testing sample:')

    print('RMSLE       = %f' % (rmsle(y_test, y_pred)))

    print('r-squared   = %f' % (r2_score(y_test, y_pred)))

    

    #---------------------------

    # Plot feature importance

    feature_importance = GBRmedian.feature_importances_

    # make importances relative to max importance

    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)

    # Let's plot to top10 most important variables

    sorted_idx = sorted_idx[-10:]

    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, df.columns[sorted_idx])

    plt.xlabel('Relative Importance')

    plt.title('Variable Importance')

    plt.show();
