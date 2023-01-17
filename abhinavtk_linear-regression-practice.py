import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
boston = load_boston()
print(boston.DESCR)
boston.data.shape
boston.target.shape
cols = boston.feature_names
boston_df = pd.DataFrame(boston.data, columns=cols)
boston_df.head()
boston_target = pd.DataFrame(boston.target, columns = ['Target'])
boston_target.head()
df = pd.concat([boston_df,boston_target], axis=1)
df.head()
df.describe(include='all')
df.isna().sum()
# No null values
# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in df:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        if column!='CHAS':
            sns.distplot(df[column])
            plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.tight_layout()
# Dropping columns with skewness
boston_df = boston_df.drop(columns=['CRIM','ZN','CHAS','B','RAD'])
boston_df.head()
plt.figure(figsize=(20,30), facecolor='white')
plotnumber = 1

for column in boston_df:
    if plotnumber<=15 :
        ax = plt.subplot(5,3,plotnumber)
        plt.scatter(boston_df[column],boston_target)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Target',fontsize=20)
    plotnumber+=1
plt.tight_layout()
# Removoing columns that don't show a linear relationship with target
boston_df = boston_df.drop(columns=['INDUS','AGE','DIS','TAX'])
scaler =StandardScaler()

X_scaled = scaler.fit_transform(boston_df)
X_scaled.shape
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = X_scaled

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
# we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = boston_df.columns
vif
# We don't find any VIF greater than 5
x_train,x_test,y_train,y_test = train_test_split(X_scaled,boston_target,test_size = 0.25,random_state=355)
x_train
x_train.shape
x_test.shape
y_train
regression = LinearRegression()
regression.fit(x_train,y_train)
regression.score(x_train,y_train)
# For adjusted R-Squared let's create a function
def adj_r2(x,y):
    r2 = regression.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(x_train,y_train)
# We have a r2 value of 67% and adjusted r2 value of 66%
regression.score(x_test,y_test)
adj_r2(x_test,y_test)
# For test data we have a r2 value of 69% and adjusted r2 value of 68%. So the model is not overfitting.
# Lasso Regularization
# LassoCV will return best alpha and coefficients after performing 10 cross validations
lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
lasscv.fit(x_train, y_train)
# best alpha parameter
alpha = lasscv.alpha_
alpha
#now that we have best parameter, let's use Lasso regression and see how well our data has fitted before

lasso_reg = Lasso(alpha)
lasso_reg.fit(x_train, y_train)
lasso_reg.score(x_test, y_test)
import pickle
# saving the model to the local file system
filename = 'finalized_model.pickle'
pickle.dump(regression, open(filename, 'wb'))
