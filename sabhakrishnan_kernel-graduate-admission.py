# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv',index_col=0)

df.head()
df.info()
df.describe()
import seaborn as sns

sns.set(style="ticks")

sns.pairplot(df)
df.corr()
# Check for Normality for GRE and Chance of Admit.

from scipy import stats

sns.distplot(df['GRE Score'])

stats.shapiro(df['GRE Score'])
sns.distplot(df['Chance of Admit '])

stats.shapiro(df['Chance of Admit '])
from sklearn.preprocessing import StandardScaler

sc_df = StandardScaler()

X = sc_df.fit_transform(df)

X1=pd.DataFrame(X,columns=df.columns)

X1.head()
stats.mannwhitneyu(X1['GRE Score'],X1['Chance of Admit '])
#EDA.

sns.scatterplot(df['GRE Score'],df['Chance of Admit '])
sns.scatterplot(df['TOEFL Score'],df['Chance of Admit '])
sns.barplot(df['University Rating'],df['Chance of Admit '])
sns.barplot(df['SOP'],df['Chance of Admit '])
sns.barplot(df['University Rating'],df['Chance of Admit '])
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

x=X1.drop('Chance of Admit ',axis=1)

y=X1['Chance of Admit ']

xc=sm.add_constant(x)

model=sm.OLS(y,xc).fit()
model.summary()
# Backward Elimination:

cols = list(x.columns)

pmax = 1

while (len(cols)>0):

    p = []

    X = x[cols]

    xc = sm.add_constant(X)

    model = sm.OLS(y,xc).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols)

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols

print(selected_features_BE)
x_update=X1[['GRE Score', 'TOEFL Score', 'LOR ', 'CGPA', 'Research']]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x_update,y,test_size=0.3,random_state=0)
from sklearn import metrics

LR = LinearRegression()

LR.fit(xtrain,ytrain)

medv_pred=LR.predict(xtest)

mse=metrics.mean_squared_error(ytest,medv_pred)

rmse=np.sqrt(mse)

print(rmse)
r_sq=metrics.r2_score(ytest,medv_pred)

print(r_sq)
from sklearn.linear_model import Ridge,Lasso,ElasticNet

m1=LinearRegression()

m2=Ridge(alpha=0.1,normalize=True)

m3=Lasso(alpha=0.1,normalize=True)

m4=ElasticNet(alpha=0.01,l1_ratio=0.989,normalize=True)
from sklearn.model_selection import KFold

from sklearn import metrics

kf=KFold(n_splits=5,shuffle=True,random_state=0)

for model, name in zip([m1,m2,m3,m4],['Linear_Regression','Ridge','Lasso','ElasticNet']):

    rmse=[]

    for train_idx,test_idx in kf.split(x_update,y):

        Xtrain,Xtest=x_update.iloc[train_idx,:],x_update.iloc[test_idx,:]

        Ytrain,Ytest=y.iloc[train_idx],y.iloc[test_idx]

        model.fit(Xtrain,Ytrain)

        Y_predict=model.predict(Xtest)

        #cm=metrics.confusion_matrix(Ytest,Y_predict)

        mse=metrics.mean_squared_error(Ytest,Y_predict)

        rmse.append(np.sqrt(mse))

    print("RMSE score=%0.03f(+-%0.5f)[%s]"% (np.mean(rmse),np.var(rmse,ddof=1),name))

    r_sq=metrics.r2_score(Ytest,Y_predict)

    print('R^2 value:',r_sq)

    
from sklearn.model_selection import GridSearchCV

param={'alpha':np.arange(0.01,1,0.01),'l1_ratio':np.arange(0.1,1,0.01)}

GS=GridSearchCV(m4,param,cv=5)

GS.fit(xtrain,ytrain)

GS.best_params_