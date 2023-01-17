# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Importing the Dataset
mtcars = house = pd.read_csv("../input/mtcars.csv")
# Converting the continuous variables in categorical ones
mtcars
mtcars['vs']=mtcars['vs'].astype('category')
mtcars['am']=mtcars['am'].astype('category')
mtcars['gear']=mtcars['gear'].astype('category')
mtcars['carb']=mtcars['carb'].astype('category')
mtcars['cyl']=mtcars['cyl'].astype('category')
mtcars.info()


g = sns.PairGrid(mtcars[[ 'mpg',  'disp', 'hp', 'drat', 'wt','qsec']])
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)

sns.pairplot(mtcars, x_vars=['disp', 'hp', 'drat', 'wt'], y_vars=["mpg"],
             height=5, aspect=.8, kind="reg");

fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(1,5,figsize=(20,7))

fig.suptitle("Boxplot for Mtcars", fontsize=35)

sns.boxplot(x="gear", y="mpg", data=mtcars,ax=ax1)
sns.boxplot(x="carb", y="mpg", data=mtcars,ax=ax2)
sns.boxplot(x="vs", y="mpg", data=mtcars,ax=ax3)
sns.boxplot(x="am", y="mpg", data=mtcars,ax=ax4)
sns.boxplot(x="cyl", y="mpg", data=mtcars,ax=ax5)



## Countplot for categorical variables
fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(1,5,figsize=(25,7))

fig.suptitle("Countplot for Mtcars", fontsize=35)

sns.countplot(x="gear", data=mtcars,ax=ax1)
sns.countplot(x="cyl", data=mtcars,ax=ax2)
sns.countplot(x="carb", data=mtcars,ax=ax3)
sns.countplot(x="vs", data=mtcars,ax=ax4)
sns.countplot(x="am", data=mtcars,ax=ax5)

## Correlation Matrix
corr = mtcars.corr()
corr.style.background_gradient()
corr.style.background_gradient().set_precision(2)


## Declaring the variables
iv=mtcars[['cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am','gear', 'carb']]
dv=mtcars['mpg']

## Creating the dummy variables 
iv=pd.get_dummies(iv)
iv=iv.drop(['cyl_4', 'vs_0','am_0','gear_3','carb_1','carb_2','carb_3','carb_4','carb_6',
           'carb_8','hp','wt'], axis=1)

## Dividing the Dataset into Test and Train
from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=1/4,random_state=0)

## Implementing Linear Regression
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(iv_train,dv_train)


## Seeing the result of Linear Regression
import statsmodels.api as sm
X2 = sm.add_constant(iv_train)
est = sm.OLS(dv_train,X2)
est2 = est.fit()
print(est2.summary())

## Predictiions for TEST
y_pred=regressor.predict(iv_test)

## Seeing the Predictions
y_pred