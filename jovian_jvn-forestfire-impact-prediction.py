target = 'area'
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')



import statsmodels.api as sm

from scipy.stats import zscore

from statsmodels.stats.stattools import durbin_watson

from sklearn.model_selection import train_test_split,KFold

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
import jovian
#path = 'forestfires.csv'

path = "../input/forest-fires-data-set/forestfires.csv"

df = pd.read_csv(path)



df.shape
df.dtypes
df.describe().T
df.isna().sum()
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

    sns.boxplot(df[col])

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

#     print(cross)



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


plt.figure(figsize =(16,8))



sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')

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
df[out_columns].skew(), df[out_columns].kurtosis()
X = df.drop(columns=['area','damage_category'])

y = df['area']
X_constant = sm.add_constant(X)
# Build OLS model

lin_reg = sm.OLS(y,X_constant).fit()

lin_reg.summary()
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
sm.qqplot(lin_reg.resid,line ='r')

plt.show()
lin_reg.resid.mean()
from statsmodels.compat import lzip

import numpy as np

from statsmodels.compat import lzip

%matplotlib inline

%config InlineBackend.figure_format ='retina'

import seaborn as sns 

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

sns.set_style('darkgrid')

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)



model = lin_reg

fitted_vals = model.predict()

resids = model.resid

resids_standardized = model.get_influence().resid_studentized_internal

fig, ax = plt.subplots(1,2)



sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})

ax[0].set_title('Residuals vs Fitted', fontsize=16)

ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})

ax[1].set_title('Scale-Location', fontsize=16)

ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')



name = ['F statistic', 'p-value']

test = sms.het_goldfeldquandt(model.resid, model.model.exog)

lzip(name, test)
import statsmodels.tsa.api as smt



acf = smt.graphics.plot_acf(lin_reg.resid, lags=40 , alpha=0.05)

acf.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=X.columns)
# Basic model
lin_reg = LinearRegression()

lin_reg.fit(X, y)

print(f'Coefficients: {lin_reg.coef_}')

print(f'Intercept: {lin_reg.intercept_}')

print(f'R^2 score: {lin_reg.score(X, y)}')
# Using p-value and variable inflation factor
df.drop(['DC','FFMC','ISI','month_aug','month_sep','month_jul','month_oct'],axis=1,inplace =True)

X = df.drop(columns=['area','damage_category'])

y = df['area']
X_constant =sm.add_constant(X)

lin_reg = sm.OLS(y,X_constant).fit()

lin_reg.summary()
# Feature Selection techniques - RFE, forward or backward selection
lin_reg = LinearRegression()

lin_reg.fit(X, y)
print(f'Coefficients: {lin_reg.coef_}')

print(f'Intercept: {lin_reg.intercept_}')

print(f'R^2 score: {lin_reg.score(X, y)}')
jovian.commit()