# !pip install pandas

# !pip install numpy

# !pip install sklearn

# !pip install matplotlib

# !pip install scipy

# !pip install statsmodels

# !pip install keras

# !pip install mplleaflet

# !pip install fancyimputation
# Math, stat and data

import pandas as pd

import numpy as np

from scipy import stats

import statsmodels.formula.api as smf

import fancyimpute as fi



# sklearn for regressions

from sklearn import ensemble, linear_model, clone

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import mean_squared_error, r2_score, make_scorer

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



# keras for deep learning

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



# packages for geopatial regressions

# import pysal as ps

# import geopandas as gpd



# packages for viz

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import Image

import mplleaflet as mpll





import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
path_train = r'../input/train.csv'

path_test = r'../input/test.csv'
df_train = pd.read_csv(path_train)

df_test = pd.read_csv(path_test)
df_train.head(10)
df_test.head(10)
df_train.describe().transpose()
# very slow because the pairplot create matrix 81*81 with all correlations

# g = sns.pairplot(df_train, hue="MSZoning")

# g.savefig('first-pairplot')
corr = df_train.corr()

k = 50

f, ax = plt.subplots(figsize=(12, 9))

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.15)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index

print("Number of Categorical features: ", len(numerical_feats))



categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index

print("Number of Numerical features: ", len(categorical_feats))
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index



skewed_feats = df_train[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
kurtosis_feats = df_train[numeric_feats].apply(lambda x: stats.kurtosistest(x.dropna())).sort_values(ascending=False)

kurtosis_feats
function1 = ''' SalePrice ~ 

 + C(MSSubClass)

 + C(MSZoning)

 + LotFrontage

 + LotArea

 + C(Street)

 + C(Alley)

 + C(LotShape)

 + C(LandContour)

 + C(Utilities)

 + C(LotConfig)

 + C(LandSlope)

 + C(Neighborhood)

 + C(Condition1)

 + C(Condition2)

 + C(BldgType)

 + C(HouseStyle)

 + OverallQual

 + OverallCond

 + YearBuilt

 + YearRemodAdd

 + C(RoofStyle)

 + C(RoofMatl)

 + C(Exterior1st)

 + C(Exterior2nd)

'''



model1 = smf.ols(function1, df_train).fit()

print(model1.summary())
def separe_numeric_categoric(df):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df_n = df.select_dtypes(include=numerics)

    df_c = df.select_dtypes(exclude=numerics)

    print(f'The DF have {len(list(df_n))} numerical features and {len(list(df_c))} categorical fets')

    return df_n, df_c

    

    

def find_missing(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    filter(lambda x: x>=minimum, percent)

    return percent





def count_missing(df):

    missing = find_missing(df)

    total_columns_with_missing = 0

    for i in (missing):

        if i>0:

            total_columns_with_missing += 1

    return total_columns_with_missing

    



def remove_missing_data(df,minimum=.1):

    percent = find_missing(df)

    number = len(list(filter(lambda x: x>=(1.0-minimum), percent)))

    names = list(percent.keys()[:number])

    df = df.drop(names, 1, errors='ignore')

    print(f'{number} columns exclude because haven`t minimium data.')

    return df





def one_hot(df, cols):

    for each in cols:

        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)

        df = pd.concat([df, dummies], axis=1)

    df = df.drop(cols, axis=1)

    return df


def impute_missing_data(df,minimium_data=.1):

    columns_missing = count_missing(df)

    print(f'Total columns with missing values: {count_missing(df)} of a {len(list(df))} columns in df')

        

    # remove features without minimium size of information

    df = remove_missing_data(df,minimium_data)

    

    numerical_df, categorical_df = separe_numeric_categoric(df)

    

    # Autocomplete using MICE for numerical features.

    try:

        df_numerical_complete = fi.MICE(verbose=False).complete(numerical_df.values)

        n_missing = count_missing(df)

        print(f'{columns_missing-n_missing} numerical features imputated')

        

        # Complete the columns name.

        temp = pd.DataFrame(columns=numerical_df.columns, data=df_numerical_complete)



        # df temp com os dados numericos completados e os categóricos.

        df = pd.concat([temp, categorical_df], axis=1)

        

    except Exception as e:

        print(e)

        print('Without Missing data in numerical features')

    

    missing = find_missing(df)

    names = missing.keys()

    n = 0

    for i, c in enumerate(missing):

        if c > 0:

            col = names[i]

            print(f'Start the prediction of {col}')

            clf = RandomForestClassifier()

            le = LabelEncoder()

            ## inverter a ordem da predição das categóricas pode melhorar a precisao.

            categorical_train = list(categorical_df.loc[:,categorical_df.columns != col])



            temp = one_hot(df,categorical_train)

            df1 = temp[temp[col].notnull()]

            df2 = temp[temp[col].isnull()]

            df1_x = df1.loc[:, df1.columns != col]

            df2_x = df2.loc[:, df1.columns != col]

            

            df1_y = df1[col]

            le.fit(df1_y)

            df1_y = le.transform(df1_y)

            clf.fit(df1_x, df1_y)

            df2_yHat = clf.predict(df2_x)

            df2_yHat = le.inverse_transform(df2_yHat)

            df2_yHat = pd.DataFrame(data=df2_yHat, columns=[col])

            df1_y = le.inverse_transform(df1_y)

            df1_y = pd.DataFrame(data=df1_y,columns=[col])

            

            df2_x.reset_index(inplace=True)   

            result2 = pd.concat([df2_yHat, df2_x], axis=1)

            try:

                del result2['index']

            except:

                pass



            df1_x.reset_index(inplace=True)

            result1 = pd.concat([df1_y, df1_x], axis=1)

            try:

                del result1['index']

            except:

                pass

            

            result = pd.concat([result1, result2])

            result = result.set_index(['Id'])

            df.reset_index()            

            try:

                df.set_index(['Id'],inplace=True)

            except:

                pass

            df[col] = result[col]

            

            n += 1



    print(f'Number of columns categorical with missing data solved: {n}')

    df = df.reset_index()

    return df





df_train = impute_missing_data(df_train)
print(f'Let\'s count misssing columns again: {count_missing(df_train)}, excelent!')
sns.distplot(df_train['SalePrice'] , fit=stats.norm);



(mu, sigma) = stats.norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = np.log(df_train['GrLivArea']), y = np.log(df_train['SalePrice']))

plt.ylabel('Price', fontsize=13)

plt.xlabel('Size', fontsize=13)

plt.show()





sns.distplot(np.log(df_train['SalePrice']) , fit=stats.norm);



(mu, sigma) = stats.norm.fit(np.log(df_train['SalePrice']))

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



fig = plt.figure()

res = stats.probplot(np.log(df_train['SalePrice']), plot=plt)

plt.show()





df_train['SalePrice'] = np.log(df_train['SalePrice'])
# YearBuilt, YearRemodAdd need special atention, because have a great difference in values ..
# we will not apply boxcox on saleprice because the kaggle competition expects 

# a simple log application that we did in the previous step.

def normalizing(df,exception=['Id','SalePrice','YearBuilt','YearRemodAdd','GarageYrBlt']):

    

    numerical_feats = df.dtypes[df_train.dtypes != "object"].index

    print("Number of Categorical features: ", len(numerical_feats))



    categorical_feats = df.dtypes[df_train.dtypes == "object"].index

    print("Number of Numerical features: ", len(categorical_feats))

    

    # skew problem

    skewed_feats = df_train[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)

    print("\nSkew in numerical features: \n")

    skewness = pd.DataFrame({'top 5 positives Skew' :skewed_feats})

    print(skewness.head())

    

    # kurtosis problem

    kurtosis_feats = df_train[numeric_feats].apply(lambda x: stats.kurtosistest(x.dropna())).sort_values(ascending=False)

    print("\nKurtosis in numerical features (stat and pvalue): \n")

    kurtosis = pd.DataFrame({'Top 5 leptokurtic feats' :kurtosis_feats})

    print(kurtosis.head())

    

    # apply boxcox transformation without rules.

    sucess, fail = 0, 0

    for feat in numerical_feats:

        if feat not in exception:

            

            try:

                df[feat] = stats.boxcox(df[feat])[0]

                print(f'Appling BoxCox in {feat}')

                sucess += 1

            except Exception as e:

                print(str(e))

                fail += 1

                pass

            

    print(f'\n{sucess} feats with boxcox transformation and {fail} feats with error in transf.')



    return df



df_train = normalizing(df_train)
_, df_c = separe_numeric_categoric(df_train)

df_c.head()
def plotBupu(df,col):

    g = sns.factorplot(x=col, data=df, kind="count",

                       palette="BuPu", size=6, aspect=1.5)

    g.set_xticklabels(step=2)
for i in list(df_c):

    plotBupu(df_train,i)
df = df_train.drop(['Utilities', 'Condition2', 'RoofMatl', 'Heating', 'GarageQual', 'GarageCond'], axis=1)

df_n, df_c = separe_numeric_categoric(df)

df = one_hot(df, list(df_c))
def imput_quadratic(df,k=10,exclude='SalePrice'):

    f, ax = plt.subplots(figsize=(12, 9))

    cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

    cm = np.corrcoef(df[cols].values.T)

    sns.set(font_scale=1.15)

    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

    for i in cols:

        if i != exclude:

            df['quad_'+i] = df[i]*df[i]

    print(f'{k} Quadratics variables imputed')

    return df



df = imput_quadratic(df,15)
df = df.set_index('Id')
def dfSplit(df,ratio,y='SalePrice'):

    train, test = train_test_split(df, test_size = ratio)

    y_train = train[y]

    y_test = test[y]

    x_train = train.ix[:, train.columns != y]

    x_test = test.ix[:, test.columns != y]

    return x_train, y_train, x_test, y_test
def tPoly(df, degree=1):

    polynomial = PolynomialFeatures(degree=degree)

    return polynomial.fit_transform(df)
scorer = make_scorer(mean_squared_error, greater_is_better = False)





def testRegs(df, clf, degree=1, ratio=.2, y='SalePrice', metrics=[]):



    x_train,y_train,x_test,y_test = dfSplit(df,ratio,y='SalePrice')



    poly_x_train = tPoly(x_train,degree)

    poly_x_test = tPoly(x_test,degree)



    clf.fit(poly_x_train,y_train)

    

    y_hat = clf.predict(poly_x_test)

    

    rmse = np.sqrt(-cross_val_score(clf, poly_x_train, y_train, scoring = scorer, cv = 10))

    

    print("RMSE:", rmse.mean())

    

    y_train_pred = clf.predict(poly_x_train)

    y_test_pred = clf.predict(poly_x_test)



    print('R2: %.2f, Score: %.2f, Parameters: %i' % (r2_score(y_test, y_hat), 

                                                     clf.score(poly_x_train,y_train), 

                                                     clf.coef_.shape[0]))



    plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")

    plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")

    plt.xlabel("Predicted values")

    plt.ylabel("Residuals")

    plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

    plt.show()



    plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")

    plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")

    plt.xlabel("Predicted values")

    plt.ylabel("Real values")

    plt.legend(loc = "upper left")

    plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

    plt.show()
ols_simple = linear_model.LinearRegression()



testRegs(df,ols_simple,1)
ridge = linear_model.RidgeCV(alphas = [0.001, 0.01, 0.1, 0.5, 0.75, 1, 1.2, 1.5, 2.5, 5])

testRegs(df,ridge)
lasso = linear_model.LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003,

                                       0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1],

                                       max_iter = 50000, cv = 10)

testRegs(df,lasso)
ensemble = ensemble.GradientBoostingRegressor(n_estimators = 500, 

                                              max_depth = 5, 

                                              min_samples_split = 2)



x_train, y_train, x_test, y_test = dfSplit(df,.1)

gbr = ensemble.fit(x_train, y_train)

print(ensemble.score(x_test, y_test))

print(np.sqrt(-cross_val_score(gbr, x_train, y_train, scoring = scorer, cv = 5)))
def rmsle_cv(model,n_folds=5):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)

    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        for model in self.models_:

            model.fit(X, y)



        return self

    

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   

    

averaged_models = AveragingModels(models = (ols_simple, ridge, gbr, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
def simple_model():

    model = Sequential()

    model.add(Dense(259, input_dim=259, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
estimator = KerasRegressor(build_fn=simple_model, nb_epoch=10000000, batch_size=1000, verbose=False)



x_train, y_train, x_test, y_test = dfSplit(df,.1)



kfold = KFold(n_splits=10)

results = cross_val_score(estimator, x_train.values, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
def simple_model_with_hidden():

    model = Sequential()

    model.add(Dense(300, input_dim=259, kernel_initializer='normal', activation='relu'))

    model.add(Dense(20, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



estimator = KerasRegressor(build_fn=simple_model_with_hidden, nb_epoch=10e15, batch_size=100, verbose=False)



x_train, y_train, x_test, y_test = dfSplit(df,.3)



kfold = KFold(n_splits=5)

results = cross_val_score(estimator, x_train.values, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))