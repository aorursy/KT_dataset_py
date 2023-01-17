import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, ElasticNet

from statsmodels.stats.outliers_influence import variance_inflation_factor
mtcars=pd.read_csv('../input/mtcars/mtcars.csv')

mtcars.head()
mtcars1=mtcars.iloc[:,1:]

mtcars1.head()
mtcars1.info()
mtcars1.describe()
mtcar4=mtcars1.transform(lambda x:x**0.5)

X_wo1=mtcar4.drop(['mpg'],axis=1)

Y_wo1=mtcar4['mpg'].values

X_const_wo1=sm.add_constant(X_wo1)

model_wo1=sm.OLS(Y_wo1,X_const_wo1).fit()

model_wo1.summary()

X=mtcars1.drop(['mpg'],axis=1)

Y=mtcars1.mpg

vif_sqrt=[variance_inflation_factor(X_const_wo1.values,i) for i in range(X_const_wo1.shape[1])]

pd.DataFrame({'vif':vif_sqrt[1:]},index=X.columns).T
X_wo1=X_wo1[['drat','vs']]

x_train1,x_test1,y_train1,y_test1=train_test_split(X_wo1,Y_wo1,test_size=0.3,random_state=1)

lin_reg_log=LinearRegression()

lin_reg_log.fit(x_train1,y_train1)

print('R^2 for train:',lin_reg_log.score(x_train1,y_train1))

print('R^2 for test:',lin_reg_log.score(x_test1,y_test1))
mtcar5=mtcars1[['disp','hp','drat','wt','qsec','mpg']]

mt_log=mtcar5.transform(lambda x:np.log(x))

mt_log[['cyl','vs','am','gear','carb']]=mtcars1[['cyl','vs','am','gear','carb']]

mt_log.head()

X_wo1=mt_log.drop(['mpg'],axis=1)

Y_wo1=mt_log['mpg'].values



X_const_wo1=sm.add_constant(X_wo1)

model_wo1=sm.OLS(Y_wo1,X_const_wo1).fit()

model_wo1.summary()



vif_sqrt=[variance_inflation_factor(X_const_wo1.values,i) for i in range(X_const_wo1.shape[1])]

vif_pd=pd.DataFrame({'vif':vif_sqrt[1:]},index=X.columns).T

print(vif_pd)



X_wo1=X_wo1[['hp','gear','am','vs']]

x_train1,x_test1,y_train1,y_test1=train_test_split(X_wo1,Y_wo1,test_size=0.3,random_state=1)

lin_reg_log=LinearRegression()

lin_reg_log.fit(x_train1,y_train1)

print()



print('R^2 for train:',lin_reg_log.score(x_train1,y_train1))

print('R^2 for test:',lin_reg_log.score(x_test1,y_test1))
mtcar5=mtcars1[['disp','hp','drat','wt','qsec','mpg']]

mt_log=mtcar5.transform(lambda x:1/x)

mt_log[['cyl','vs','am','gear','carb']]=mtcars1[['cyl','vs','am','gear','carb']]

mt_log.head()

X_wo1=mt_log.drop(['mpg'],axis=1)

Y_wo1=mt_log['mpg'].values

X_const_wo1=sm.add_constant(X_wo1)

model_wo1=sm.OLS(Y_wo1,X_const_wo1).fit()

model_wo1.summary()

vif_sqrt=[variance_inflation_factor(X_const_wo1.values,i) for i in range(X_const_wo1.shape[1])]

vif_pd=pd.DataFrame({'vif':vif_sqrt[1:]},index=X_wo1.columns).T

print(vif_pd)



X_wo1=X_wo1[['drat','carb','vs']]

x_train1,x_test1,y_train1,y_test1=train_test_split(X_wo1,Y_wo1,test_size=0.3,random_state=1)

lin_reg_log=LinearRegression()

lin_reg_log.fit(x_train1,y_train1)

print()



print('R^2 for train:',lin_reg_log.score(x_train1,y_train1))

print('R^2 for test:',lin_reg_log.score(x_test1,y_test1))
cor=mtcars1.corr()

cor1=cor['mpg']

featu=cor1[abs(cor1)>0.5][1:]# [1:] remove mpg from list



mult_cor=mtcars1[['cyl', 'disp', 'hp', 'drat', 'wt', 'vs', 'am', 'carb','mpg']].corr()

cor_max=max(abs(featu.values))

final=featu[abs(featu.values)==cor_max]

final
X2=mtcars1[['wt']]

Y2=mtcars1.mpg

x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)

lin1=LinearRegression()

lin1.fit(x_train,y_train)

print('R2 for train:',lin1.score(x_train,y_train))

print('R2 for test:',lin1.score(x_test,y_test))
X=mtcars1.drop(['mpg'],axis=1)

Y=mtcars1.mpg

model=sm.OLS(Y,X).fit()

model.pvalues
lin=LinearRegression()

cols=list(X.columns)

select_feat=[]



while(len(cols)>0):

    p=[]

    X1=X[cols]

    model1=sm.OLS(Y,X1).fit()

    p=pd.Series(model1.pvalues,index=X1.columns)

    pmax=max(p)

    if(pmax>0.05):

        feature_with_p_max = p.idxmax()

        cols.remove(feature_with_p_max)

    else:

        break

select_feat=cols

select_feat
X2=mtcars1[['wt', 'qsec', 'am']]

Y2=mtcars1.mpg

x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)

lin1=LinearRegression()

lin1.fit(x_train,y_train)

print('R2 for train:',lin1.score(x_train,y_train))

print('R2 for test:',lin1.score(x_test,y_test))
lin=LinearRegression()

X.columns

highsc=0

nof=0

support_score=[]

noflist=np.arange(1,11)

for n in noflist:

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

    rfe=RFE(lin,n)

    X_train_rfe=rfe.fit_transform(x_train,y_train)

    X_test_rfe=rfe.transform(x_test)

    lin.fit(X_train_rfe,y_train)

    score=lin.score(X_test_rfe,y_test)

    if(score>highsc):

        highsc=score

        nof=n

        support_score=rfe.support_

        

temp=pd.Series(support_score,index=X.columns)

print('No of optimum features:',n)

print('SCore for optimum features:',highsc)

print('Features Selected:\n')

temp[temp==True].index
X2=mtcars1[['drat', 'wt', 'gear', 'carb']]

Y2=mtcars1.mpg

x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)

lin1=LinearRegression()

lin1.fit(x_train,y_train)

print('R2 for train:',lin1.score(x_train,y_train))

print('R2 for test:',lin1.score(x_test,y_test))
vif=[variance_inflation_factor(X.values, j) for j in range(X.shape[1])]

vif_pd=pd.Series(vif,index=X.columns)

vif_pd
def calculate_vif(x):

    output=pd.DataFrame()

    vif=[variance_inflation_factor(x.values, j) for j in range(x.shape[1])]

    cols = x.shape[1]

    thresh=5.0

    for i in range(cols):

        print('Iteration:',i)

        a=np.argmax(vif)

        print('Max vif found at:',a)

        if(vif[a]>thresh):

            if i==0:

                output=x.drop(x.columns[a],axis=1)

                vif=[variance_inflation_factor(output.values, j) for j in range(output.shape[1])]

            else:

                output=output.drop(output.columns[a],axis=1)

                vif=[variance_inflation_factor(output.values, j) for j in range(output.shape[1])]

        else:

            break

    return output.columns

calculate_vif(X).values
X2=mtcars1[['disp', 'vs', 'am']]

Y2=mtcars1.mpg

x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)

lin1=LinearRegression()

lin1.fit(x_train,y_train)

print('R2 for train:',lin1.score(x_train,y_train))

print('R2 for test:',lin1.score(x_test,y_test))
X=mtcars1.drop(['mpg'],axis=1)

Y=mtcars1.mpg



reg = LassoCV()

reg.fit(X, Y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,Y))

coef = pd.Series(reg.coef_, index = X.columns)



coeff=coef.sort_values()

coeff.plot(kind='bar')

plt.show()

X2=mtcars1[['disp', 'hp']]

Y2=mtcars1.mpg

x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)

lin1=LinearRegression()

lin1.fit(x_train,y_train)

print('R2 for train:',lin1.score(x_train,y_train))

print('R2 for test:',lin1.score(x_test,y_test))
reg1=ElasticNet()

reg1.fit(X, Y)

print("Best alpha using built-in ElasticNet: %f" % reg1.alpha)

print("Best score using built-in ElasticNet: %f" %reg1.score(X,Y))

coef_elastic = pd.Series(reg1.coef_, index = X.columns)
coeff=coef_elastic.sort_values()

coeff.plot(kind='bar')

plt.show()
X2=mtcars1[['wt','carb','cyl','disp', 'hp','qsec']]

Y2=mtcars1.mpg

x_train,x_test,y_train,y_test=train_test_split(X2,Y2,test_size=0.3,random_state=1)

lin1=LinearRegression()

lin1.fit(x_train,y_train)

print('R2 for train:',lin1.score(x_train,y_train))

print('R2 for test:',lin1.score(x_test,y_test))