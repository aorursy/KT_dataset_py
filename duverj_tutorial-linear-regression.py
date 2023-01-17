import warnings

warnings.filterwarnings('ignore')



import pandas as pd

pd.options.display.max_columns = None

pd.set_option('display.float_format', lambda x: '%.6f' % x)

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

plt.style.use('ggplot')

from sklearn import model_selection

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from scipy.stats import skew
#"kaggle" prefix to avoid confusion with train-test splits we'll generate later for cross-validation

kaggle_train = pd.read_csv('../input/train.csv') 

kaggle_test = pd.read_csv('../input/test.csv')



#merging the two dataframes in one

df = pd.concat([kaggle_train, kaggle_test]).reset_index()
noSalePrice = [x for x in df.columns if x!='SalePrice']

df[noSalePrice] = df[noSalePrice].fillna(df[noSalePrice].mean())
fig, ax = plt.subplots(1, 3, figsize=(17,5))

sns.scatterplot('GrLivArea', 'SalePrice', data=df[:1460], ax=ax[0]); #after 1460 no more SalePrice values anymore

sns.scatterplot('OverallQual', 'SalePrice', data=df[:1460], ax=ax[1]);

sns.scatterplot('TotalBsmtSF', 'SalePrice', data=df[:1460], ax=ax[2]);

ax[1].set_title('Before Removing Outliers');



for a in ax:

    a.yaxis.label.set_visible(False)

    a.get_yaxis().set_visible(False)
df = df[~df['Id'].isin([524, 1299])] #getting rid of outliers, situated at Ids 524 and 1299



fig, ax = plt.subplots(1, 3, figsize=(17,5))

sns.scatterplot('GrLivArea', 'SalePrice', data=df[:1458], ax=ax[0]);

sns.scatterplot('OverallQual', 'SalePrice', data=df[:1458], ax=ax[1]);

sns.scatterplot('TotalBsmtSF', 'SalePrice', data=df[:1458], ax=ax[2]);

ax[1].set_title('After Removing Outliers');



for a in ax:

    a.yaxis.label.set_visible(False)

    a.get_yaxis().set_visible(False)
fig, ax = plt.subplots(1, 2, figsize=(17,5))

sns.distplot(df[:1458]['SalePrice'], ax=ax[0]);

ax[0].set_title('Before Log Transformation');



#log transformation

df.loc[df.SalePrice.notnull(), 'SalePrice_LOG'] = np.log1p(df.loc[df.SalePrice.notnull(), 'SalePrice']) 



sns.distplot(df[:1458]['SalePrice_LOG'], ax=ax[1])

ax[1].set_title('After Log Transformation');
#TRANSFORMING SALEPRICE AND GETTING RID OF SALEPRICE_LOG, JUST NEEDED FOR THE GRAPHS

df['SalePrice'] = np.log1p(df['SalePrice'])

if('SalePrice_LOG' in df.columns): df.drop('SalePrice_LOG', axis=1, inplace=True)
fig, ax = plt.subplots(1, 3, figsize=(17,5))

sns.scatterplot('GrLivArea', 'SalePrice', data=df, ax=ax[0]);

sns.scatterplot('OverallQual', 'SalePrice', data=df, ax=ax[1]);

sns.scatterplot('TotalBsmtSF', 'SalePrice', data=df, ax=ax[2]);

ax[1].yaxis.label.set_visible(False)

ax[2].yaxis.label.set_visible(False)
dfc = df[['SalePrice', 'GrLivArea', 'OverallQual', 'TotalBsmtSF']].corr()

dfc[['SalePrice']][1:]
from patsy import dmatrices

Y, X = dmatrices('SalePrice ~ OverallQual+GrLivArea+TotalBsmtSF', df, return_type='dataframe')

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif
#training dataset

X = [[1, 2], [3, 4]]

Y = [5, 6]



#defining and training the machine in two lines of code

model = linear_model.LinearRegression()

model.fit(X, Y);



#from this point, the machine can compute a Y value every time it is given some X values

#here, with 5 and 6 as inputs, the machine predicts the output will be 7 

model.predict([[5,6]])[0]
X = df[:1458][['GrLivArea']] #independant variables

Y = df[:1458]['SalePrice'] #dependant variable

model = linear_model.LinearRegression() #LinearRegression() = Least Square Algorithm

model.fit(X, Y); #fitting, or "training" the model
plt.rcParams['figure.figsize'] = (12.0, 6.0)

sns.scatterplot('GrLivArea', 'SalePrice', data=df); 



Ypredicted = model.predict(X)

sns.lineplot(X.iloc[:, 0].tolist(), Ypredicted, color='black', label='Least Squares Line'); 



Yinvented = X.iloc[:, 0].as_matrix() * 0.00145 + 10

sns.lineplot(X.iloc[:, 0].tolist(), Yinvented, color='blue', label='My Custom Line');
ind_vars = ['GrLivArea', 'OverallQual', 'TotalBsmtSF']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(df[:1458][ind_vars], df[:1458]['SalePrice'], test_size=0.33)
model = LinearRegression()

model.fit(Xtrain, Ytrain);

Ypredicted = model.predict(Xtest)

dff = pd.DataFrame({"The values our model predicted":np.expm1(Ypredicted), 

              "The values it should have predicted, assuming it was perfect":np.expm1(Ytest.tolist())})

dff.astype(int).head()
def adjusted_r2(Xtest, r2):



    p = len(Xtest.columns) #number of independant values

    n = len(Xtest) #length of test dataset

    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    return adj_r2
r2 = r2_score(Ytest, Ypredicted)

adj_r2 = adjusted_r2(Xtest, r2)

mse = mean_squared_error(Ytest, Ypredicted)

pd.DataFrame(data=[r2, adj_r2, mse], index=['r2', 'r2_adj', 'mse'])
X = df[:1458][ind_vars]

Y = df[:1458]['SalePrice']

mse = -model_selection.cross_val_score(model, X, Y, cv=10, scoring='neg_mean_squared_error').mean()

round(mse, 4)
#extracting numeric independant variables

dfc = df[:1458].corr()[['SalePrice']].sort_values('SalePrice', ascending=False) 

num_cols = dfc.drop(['SalePrice', 'Id']).index 



# df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols].astype(float)), columns=num_cols) #scaling the numeric columns

Y = df[:1458]['SalePrice']

X = df[:1458][num_cols]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=42)
#TRAINING MODELS

linear_model = LinearRegression().fit(Xtrain, Ytrain)

ridge_model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 200], normalize=True).fit(Xtrain, Ytrain)

lasso_model = LassoCV(alphas=[0.00001, 0.0001, 0.001, 0.01], normalize=True).fit(Xtrain, Ytrain)

elastic_model = ElasticNetCV(alphas= [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02], normalize=True).fit(Xtrain, Ytrain)



#TESTING PERFORMANCES

print('Linear  | mse :', round(mean_squared_error(Ytest, linear_model.predict(Xtest)), 5))

print('Ridge   | mse :', round(mean_squared_error(Ytest, ridge_model.predict(Xtest)), 5), '| alpha :', ridge_model.alpha_)

print('Lasso   | mse :', round(mean_squared_error(Ytest, lasso_model.predict(Xtest)), 5), '| alpha :', lasso_model.alpha_)

print('Elastic | mse :', round(mean_squared_error(Ytest, elastic_model.predict(Xtest)), 5), '| alpha :', elastic_model.alpha_)
# FUNCTION TO RETURN THE ACCURACY OF THE REGRESSION, THROUGH MEAN SQUARED ERROR

def test_regression(model, X, Y): 

    return round(np.sqrt(-model_selection.cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')).mean(), 5)
# FUNCTION TO CREATE ALL COMBINATIONS OF CORRELATION, SKEWNESS AND MODELS

 # AND STOCKING RESULTS IN LIST OF DICTS

def tests(X, Y, test_dicts, include_dummies):



    #LOOPING THROUGH CORRELATIONS THRESHOLDS

    for i in corr_list:



        #getting only ind. variables that have over the requested correlation with dep. variable

        not_correlated_vars = correlations[correlations.iloc[:,0]<i].index

        cols_to_keep = [x for x in X.columns if x not in not_correlated_vars]

        X2 = X[cols_to_keep]



        #LOOPING THROUGH SKEWNESS THRESHOLDS

        for i2 in skew_list:



            X3 = X2.copy()

            #getting only ind. variables that have over the requested skewness

            skewed_feats = skewness[skewness.iloc[:,0]>i2].index

            skewed_feats = [x for x in skewed_feats if x in X3.columns]

            X3[skewed_feats] = np.log1p(X3[skewed_feats])

            

            #stocking relevant information into dict

            param_dict = {'correlation_threshold':str(i), 'skewness_thresold':str(i2), 'number_of_variables':X3.shape[1], 'include_dummies':include_dummies}



            #TESTING THE FOUR MODELS FOR EACH COMBINATION OF CORRELATION/SKEWNESS

                #AND APPENDING THEIR SCORE TO LIST OF DICTS, WITH INFORMATION ASSOCIATED

            model = LinearRegression()

            #z in zscore only to have column at end of dataframe

            test_dicts.append({**{'zscore':test_regression(model, X3, Y), 'model':'linear'}, **param_dict})



            model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 200], normalize=True)

            test_dicts.append({**{'zscore':test_regression(model, X3, Y), 'model':'ridge'}, **param_dict})





            model = LassoCV(alphas=[0.00001, 0.0001, 0.001, 0.01], tol=0.1, normalize=True)

            test_dicts.append({**{'zscore':test_regression(model, X3, Y), 'model':'lasso'}, **param_dict})





            model = ElasticNetCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1], tol=0.1, normalize=True)

            test_dicts.append({**{'zscore':test_regression(model, X3, Y), 'model':'elastic'}, **param_dict})
test_dicts = []



#defining correlation and skewness threshold

corr_list = [0, 0.25, 0.5]

skew_list = [0, 0.5, 0.75, 0.9, 999] #999 threshold is same as skewing no variable



#computing skewness and correlation of ind.variables

skewness = df[num_cols].apply(lambda x: skew(x)).abs().to_frame() #skewness can be evaluated on both train and test

correlations = df[:1458].corr().abs().sort_values('SalePrice', ascending=False)['SalePrice'].to_frame()[1:].drop('Id')



#RUNNING THE FIRST TESTS, WITHOUT CATEGORICAL VARIABLES

tests(X, Y, test_dicts, include_dummies='no')





#RUNNING THE SECOND SET OF TESTS, INCLUDING CATEGORICAL VARIABLES

df_dummies = pd.get_dummies(df)

X = df_dummies[:1458].drop(['SalePrice', 'Id'], axis=1)

tests(X, Y, test_dicts, include_dummies='yes')



#computing the results dataframe

results = pd.DataFrame(test_dicts)

results.sort_values('zscore').head()
#RECREATING BEST SCENARIO

Xtrain = df_dummies[:1458].drop(['SalePrice'], axis=1) 

Ytrain = df[:1458]['SalePrice']

Xtest = df_dummies[1458:].drop(['SalePrice'], axis=1) 



skewed_feats = skewness[skewness.iloc[:,0]>0.9].index

skewed_feats = [x for x in skewed_feats if x in Xtest.columns]

Xtest[skewed_feats] = np.log1p(Xtest[skewed_feats])

Xtrain[skewed_feats] = np.log1p(Xtrain[skewed_feats])



model = LassoCV(alphas=[0.00001, 0.0001, 0.001, 0.01], tol=0.1, normalize=True)



#FITTING AND PREDICTING

model.fit(Xtrain, Ytrain)

predicted_prices = np.expm1(model.predict(Xtest))



#SUBMITTING

my_submission = pd.DataFrame({'Id': Xtest.Id, 'SalePrice': predicted_prices}) 

my_submission.to_csv('submission.csv', index=False)