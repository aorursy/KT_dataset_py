target = 'area'
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')



import statsmodels.api as sm

from statsmodels.compat import lzip

import statsmodels.stats.api as sms

from statsmodels.formula.api import ols

from scipy.stats import zscore

from statsmodels.stats.stattools import durbin_watson

from sklearn.model_selection import train_test_split,KFold

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import RFECV

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
# path = 'forestfires.csv'

path = "../input/forest-fires-data-set/forestfires.csv"

df = pd.read_csv(path)



df.shape
df.dtypes
df.describe().T
df.isna().sum().sum()
plt.rcParams["figure.figsize"] = 9,5
plt.figure(figsize=(16,5))

print("Skew: {}".format(df[target].skew()))

print("Kurtosis: {}".format(df[target].kurtosis()))

ax = sns.kdeplot(df[target],shade=True,color='g')

plt.xticks([i for i in range(0,1200,50)])

plt.show()
ax = sns.boxplot(df[target])
# Outlier points

y_outliers = df[abs(zscore(df[target])) >= 3 ]

y_outliers
dfa = df.drop(columns=target)

cat_columns = dfa.select_dtypes(include='object').columns.tolist()

num_columns = dfa.select_dtypes(exclude='object').columns.tolist()



cat_columns,num_columns
# analyzing categorical columns

plt.figure(figsize=(16,10))

for i,col in enumerate(cat_columns,1):

    plt.subplot(2,2,i)

    sns.countplot(data=dfa,y=col)

    plt.subplot(2,2,i+2)

    df[col].value_counts(normalize=True).plot.bar()

    plt.ylabel(col)

    plt.xlabel('% distribution per category')

plt.tight_layout()

plt.show()    
plt.figure(figsize=(18,40))

for i,col in enumerate(num_columns,1):

    plt.subplot(8,4,i)

    sns.kdeplot(df[col],color='g',shade=True)

    plt.subplot(8,4,i+10)

    df[col].plot.box()

plt.tight_layout() 

plt.show()

num_data = df[num_columns]

pd.DataFrame(data=[num_data.skew(),num_data.kurtosis()],index=['skewness','kurtosis'])
print(df['area'].describe(),'\n')

print(y_outliers)
# a categorical variable based on forest fire area damage

# No damage, low, moderate, high, very high

def area_cat(area):

    if area == 0.0:

        return "No damage"

    elif area <= 1:

        return "low"

    elif area <= 25:

        return "moderate"

    elif area <= 100:

        return "high"

    else:

        return "very high"



df['damage_category'] = df['area'].apply(area_cat)

df.head()
cat_columns
for col in cat_columns:

    cross = pd.crosstab(index=df['damage_category'],columns=df[col],normalize='index')

    cross.plot.barh(stacked=True,rot=40,cmap='hot')

    plt.xlabel('% distribution per category')

    plt.xticks(np.arange(0,1.1,0.1))

    plt.title("Forestfire damage each {}".format(col))

plt.show()
plt.figure(figsize=(20,40))

for i,col in enumerate(num_columns,1):

    plt.subplot(10,1,i)

    if col in ['X','Y']:

        sns.swarmplot(data=df,x=col,y=target,hue='damage_category')

    else:

        sns.scatterplot(data=df,x=col,y=target,hue='damage_category')

plt.show()
selected_features = df.drop(columns=['damage_category','day','month']).columns

selected_features
sns.pairplot(df,hue='damage_category',vars=selected_features)

plt.show()
out_columns = ['area','FFMC','ISI','rain']
df = pd.get_dummies(df,columns=['day','month'],drop_first=True)
print(df[out_columns].describe())

np.log1p(df[out_columns]).skew(), np.log1p(df[out_columns]).kurtosis()
# FFMC and rain are still having high skew and kurtosis values, 

# since we will be using Linear regression model we cannot operate with such high values

# so for FFMC we can remove the outliers in them using z-score method

mask = df.loc[:,['FFMC']].apply(zscore).abs() < 3



# Since most of the values in rain are 0.0, we can convert it as a categorical column

df['rain'] = df['rain'].apply(lambda x: int(x > 0.0))



df = df[mask.values]

df.shape
out_columns.remove('rain')

df[out_columns] = np.log1p(df[out_columns])
df[out_columns].skew()
# we will use this dataframe for building our ML model

df_ml = df.drop(columns=['damage_category']).copy()
X = df.drop(columns=['area','damage_category'])

y = df['area']
X_constant = sm.add_constant(X)



# Build OLS model

lin_reg = sm.OLS(y,X_constant).fit()

lin_reg.summary()
import scipy.stats as stats

import pylab



# get an instance of Influence with influence and outlier measures 

st_resid = lin_reg.get_influence().resid_studentized_internal

stats.probplot(st_resid,dist="norm",plot=pylab)

plt.show()
# return fstat and p-value

sm.stats.diagnostic.linear_rainbow(lin_reg)
# The mean expected value around 0, it implies linearity is preserved

lin_reg.resid.mean()
def linearity_test(model, y):

    '''

    Function for visually inspecting the assumption of linearity in a linear regression model.

    It plots observed vs. predicted values and residuals vs. predicted values.

    

    Args:

    * model - fitted OLS model from statsmodels

    * y - observed values

    '''

    fitted_vals = model.predict()

    resids = model.resid



    fig, ax = plt.subplots(1,2,figsize=(15,5))



    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})

    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)

    ax[0].set(xlabel='Predicted', ylabel='Observed')



    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})

    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)

    ax[1].set(xlabel='Predicted', ylabel='Residuals')

    

linearity_test(lin_reg, y) 

plt.tight_layout()
sns.distplot(lin_reg.resid,fit=stats.norm)

plt.text(4,0.5,f"Skewness: {round(lin_reg.resid.skew(),2)}",fontsize=15)

plt.show()
sm.qqplot(lin_reg.resid,line ='r')

jb = [round(n,2) for n in stats.jarque_bera(lin_reg.resid)]

plt.text(-2,4,f"Jarque bera: {jb}",fontsize=15)

plt.show()
sms.het_goldfeldquandt(lin_reg.resid, lin_reg.model.exog)
model = lin_reg

fitted_vals = model.predict()

resids = model.resid

resids_standardized = model.get_influence().resid_studentized_internal



fig, ax = plt.subplots(1,2)



sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})

ax[0].set_title('Predicted vs Residuals', fontsize=16)

ax[0].set(xlabel='Predicted Values', ylabel='Residuals')



sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})

ax[1].set_title('Scale-Location', fontsize=16)

ax[1].set(xlabel='Predicted Values', ylabel='sqrt(abs(Residuals))')



name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt(model.resid, model.model.exog)

lzip(name, test)

plt.tight_layout()
import statsmodels.tsa.api as smt

# Confidence intervals are drawn as a cone. 

# By default, this is set to a 95% confidence interval, 

# suggesting that correlation values outside of this code are very likely a correlation 

# and not a statistical fluke

acf = smt.graphics.plot_acf(lin_reg.resid, lags=50 , alpha=0.05)

acf.show()
plt.figure(figsize =(16,10))



sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',fmt=".2f",cbar=False)

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=X.columns).sort_values(by="vif",ascending=False)
lr = LinearRegression()

lr.fit(X, y)



print(f'Intercept: {lr.intercept_}')

print(f'R^2 score: {lr.score(X, y)}')

pd.DataFrame({"Coefficients": lr.coef_}, index=X.columns)
X = df.drop(columns=['area','damage_category'])

y = df['area']
def check_stats(X,y):

    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(pd.DataFrame({'vif': vif}, index=X.columns).sort_values(by="vif",ascending=False)[:10])

    lin_reg = sm.OLS(y,X).fit()

    print(lin_reg.summary())

check_stats(X,y)
X.drop(columns=['FFMC'],inplace=True)

# check_stats(X,y)
X.drop(columns=['Y'],inplace=True)

# check_stats(X,y)
X.drop(columns=['month_jul'],inplace=True)

# check_stats(X,y)
X.drop(columns=['day_thu'],inplace=True)

# check_stats(X,y)
X.drop(columns=['day_mon'],inplace=True)

# check_stats(X,y)
X.drop(columns=['month_aug'],inplace=True)

check_stats(X,y)
X_m, y_m = df_ml.drop(columns=[target]), df_ml[target]
# RFECV is a variant with inbuilt Cross validation

model = LinearRegression()

selector = RFECV(model,cv=5)

selector = selector.fit(X_m, y_m)

print(f"Out of {len(X_m.columns)} features, best number of features {selector.n_features_}")

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score")

plt.plot(range(1, len(X_m.columns) + 1), selector.grid_scores_)

plt.show()
# In our stats method we found that the intercept was not relevant 

# Let's try that feature out in our ML model

model = LinearRegression(fit_intercept=False)

selector = RFECV(model,cv=5)

selector = selector.fit(X_m, y_m)

print(f"Out of {len(X_m.columns)} features, best number of features {selector.n_features_}")



plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score")

plt.plot(range(1, len(X_m.columns) + 1), selector.grid_scores_)

print(X_m.columns[selector.support_].values)

plt.show()
mask = selector.support_

print(f"Best features according to RFE {X_m.columns[mask].values}")



X_m1 = X_m.iloc[:,mask]

# We could have used train test split or cross validation strategies

# for scoring the model but in order to compare with the stats model 

# we will use the whole data

model1 = LinearRegression().fit(X_m1,y_m)

print(f"R2 Score: {model1.score(X_m1,y_m)}")
model = LinearRegression(fit_intercept=False)

sfs1 = sfs(model,k_features=20,forward=True,scoring='r2',cv=5)

sfs1.fit(X_m,y_m)

fig = plot_sfs(sfs1.get_metric_dict())

plt.title('Forward Selection')

plt.grid()

plt.show()
print(sfs1.k_features, sfs1.k_feature_names_,sep="\n")
index = list(sfs1.k_feature_idx_)

X_m1 = X_m.iloc[:,index]

model1 = LinearRegression().fit(X_m1,y_m)

print(f"R2 Score: {model1.score(X_m1,y_m)}")
model = LinearRegression(fit_intercept=False)

sfs1 = sfs(model,k_features=6,forward=False,scoring='r2',cv=5)

sfs1.fit(X_m,y_m)

fig = plot_sfs(sfs1.get_metric_dict())

plt.title('Backward Selection')

plt.grid(True)

plt.show()
index = list(sfs1.k_feature_idx_)

print(f"Best features according to RFE: {X_m.columns[index]}")



X_m1 = X_m.iloc[:,index]

model1 = LinearRegression().fit(X_m1,y_m)

print(f"R2 Score: {model1.score(X_m1,y_m)}")
# higher the alpha value, more restriction on the coefficients; 

# lower the alpha > more generalization, coefficients are barely

rr = RidgeCV(cv=5,fit_intercept=False) 

rr.fit(X_m, y_m)

rr.score(X_m,y_m)
rr.alpha_
plt.plot(rr.coef_,alpha=0.7,marker='*',markersize=10,color='red',label=r'Ridge; $\alpha =10$') 

plt.grid(True)

plt.xticks(range(0,28,1))

plt.legend()

plt.show()