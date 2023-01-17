import pandas as pd

import numpy as np

from pandas import DataFrame,Series

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

%matplotlib inline
# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

# prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

# prices.hist()



#data = pd.read_csv('‪D:\Data\UCI Datasets\winequality-red.csv',sep = ';')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
train.shape,test.shape
train.head(1)
import warnings

warnings.filterwarnings('ignore')
#check for dupes for Id

idsUnique = len(set(train.Id))

idsTotal = train.shape[0]

idsdupe = idsTotal - idsUnique

print(idsdupe)

#drop id col

train.drop(['Id'],axis =1,inplace=True)
# most correlated features

# Colormap YIOrRd is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r



corrmat = train.corr()

plt.figure(figsize = (15,7))

# or fig, ax = plt.subplots(figsize=(20, 10))

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

#sns.set_palette("husl")

g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap = 'RdYlBu')
train_nas = train.isnull().sum()

train_nas = train_nas[train_nas>0]

train_nas.sort_values(ascending=False)[:7]
test_nas = test.isnull().sum()

test_nas = test_nas[test_nas>0]

test_nas.sort_values(ascending = False)[:7]
corr = train.corr()

corr
corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
categorical_features = train.select_dtypes(include=['object']).columns

categorical_features
numerical_features = train.select_dtypes(exclude=['object']).columns

numerical_features
categorical_features.size,numerical_features.size
numerical_features = numerical_features.drop("SalePrice")

train_num = train[numerical_features]

train_cat = train[categorical_features]
all_num = all_data[numerical_features]

all_cat = all_data[categorical_features]


print("NAs for numerical features in whole data : " + str(all_num.isnull().values.sum()))

all_num = all_num.fillna(all_num.mean())

print("Remaining NAs for numerical features in train : " + str(all_num.isnull().values.sum()))


print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

train_num = train_num.fillna(train_num.mean())

print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
from scipy.stats import skew 

skewness = all_num.apply(lambda x: skew(x.dropna()))

skewness = skewness[abs(skewness) > 0.5]

skewness.index

skew_features = all_num[skewness.index]

skew_features  = np.log1p(skew_features)

all_num[skewness.index] = skew_features
all_cat = pd.get_dummies(all_cat)

all_cat.shape
all_data = pd.concat([all_cat,all_num],axis=1)

all_data.shape
train.shape
test.shape
skewness = train_num.apply(lambda x: skew(x.dropna()))

skewness[abs(skewness) > 0.5].sort_values(ascending=False)

skewness = skewness[abs(skewness) > 0.5]

skewness.index

skew_features = train[skewness.index]

skew_features.columns
skew_features  = np.log1p(skew_features)
skew_features.isnull().values.sum()
#train_num[skewness.index] = skew_features
print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

train_num = train_num.fillna(train_num.mean())

print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
train_num.head()
train_cat = pd.get_dummies(train_cat)

train_cat.shape
train_cat.head()
train_cat.isnull().values.sum()
train1 = pd.concat([train_cat,train_num],axis=1)

train1.shape
target = train['SalePrice']


# plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")

# plt.scatter(test_pre,test_pre - y_test, c = "black",  label = "Validation data")

# plt.title("Linear regression")

# plt.xlabel("Predicted values")

# plt.ylabel("Residuals")

# plt.legend(loc = "upper left")

# plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

# plt.show()
train.head()
from sklearn.model_selection import train_test_split # to split the data into two parts

X_train,X_test,y_train,y_test = train_test_split(train,target, random_state = 0)

#Fill the training and test data with require information

# X_train = train[features] 

# y_train = train[target]

# X_test = test[features]

# y_test = test[target]

# from sklearn.preprocessing import MinMaxScaler

# sc = MinMaxScaler()

# X_train = sc.fit_transform(X_train)

# X_test = sc.transform(X_test)
#For All Data   

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice

train["SalePrice"] = np.log1p(train["SalePrice"])

y_train = train['SalePrice']

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train,y_train)

alpha = ridge.alpha_

print('best alpha',alpha)

print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)

# print("Ridge RMSE on Training set :", rmse_CV_train(ridge).mean())

# print("Ridge RMSE on Test set :", rmse_CV_test(ridge).mean())

y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)

#ridge.score(X_test,y_test)
y_test_rdg = ridge.predict(X_test)

df = pd.DataFrame(data = y_test_rdg,columns = ['SalePrice'])

df['Id'] = test['Id']

df = df[['Id','SalePrice']]

df['SalePrice'] = df['SalePrice'].map('{:.2f}'.format)
df.to_csv('output_HP.csv', sep=',',index = False) 
plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "green",  label = "Training data")

#plt.scatter(y_test_rdg,y_test_rdg - y_test, c = "green",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()


from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
from sklearn.linear_model import Ridge

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
linridge = Ridge(alpha=12.797).fit(X_train, y_train)
linridge.score(X_train, y_train)



print('Housing Prices dataset')



print('ridge regression linear model intercept: {}'

     .format(linridge.intercept_))

# print('ridge regression features: {}'

#      .format(features))

print('ridge regression linear model coeff:\n{}'

     .format(linridge.coef_))

print('R-squared score (training): {:.3f}'

     .format(linridge.score(X_train, y_train)))

# print('R-squared score (test): {:.3f}'

#      .format(linridge.score(X_test, y_test)))

print('Number of non-zero features: {}'

     .format(np.sum(linridge.coef_ != 0)))

print('Number of zero features: {}'

     .format(np.sum(linridge.coef_ == 0)))
predictions = np.expm1(linridge.predict(X_test))

df = pd.DataFrame(data = predictions,columns = ['SalePrice'])

df['Id'] = test['Id']

df = df[['Id','SalePrice']]

df['SalePrice'] = df['SalePrice']#.map('{:.2f}'.format)

df.to_csv('output_HP.csv', sep=',',index = False)

df.head()

from sklearn.linear_model import Lasso

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





#lasso = Lasso(random_state=0)

alphas = np.logspace(-4, -0.5, 30)



tuned_parameters = [{'alpha': alphas}]

n_folds = 3

#lasso_cv = LassoCV(alphas=alphas, random_state=0)

lasso_cv = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)



lasso_cv.fit(X_train, y_train)

lasso_cv.score(X_train, y_train)

#lasso_cv.predict(X_test)
cv_lasso = rmse_cv(lasso_cv)
cv_lasso.mean()
coef = pd.Series(lasso_cv.coef_, index = X_train.columns)

coef.head()
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
imp_coef
plt.rcParams['figure.figsize'] = (10.0, 7.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
from sklearn.neighbors import KNeighborsRegressor



knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)



#print(knnreg.predict(X_test))

print('R-squared train score: {:.3f}'

     .format(knnreg.score(X_train, y_train)))
alphas = [1,4,8,16,32,64]

cv_knn = [rmse_cv(KNeighborsRegressor(n_neighbors = alpha)).mean() 

            for alpha in alphas]
np.array(cv_knn).mean()