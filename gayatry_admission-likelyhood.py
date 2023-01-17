import pandas as pd

import numpy as np

from pandas import Series,DataFrame



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv',index_col='Serial No.')



df
df.info()
df.describe()
def normalising(feature):

    nmx = 10

    nmn = 0

    mx = feature.max()

    mn = feature.min()

    return ((nmx- nmn)/(mx - mn)*(feature - mx) + nmx)
#values = df.columns

norm_df = normalising(df)

norm_df
norm_df.describe()
figs,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(16,8))

sns.scatterplot('GRE Score','Chance of Admit ',data=df,ax=ax1)

sns.scatterplot('TOEFL Score','Chance of Admit ',data=df,ax=ax2)

sns.barplot('University Rating','Chance of Admit ',data=df,ax=ax3)

sns.barplot('SOP','Chance of Admit ',data=df,ax=ax4)

sns.barplot('LOR ','Chance of Admit ',data=df,ax=ax5)

sns.scatterplot('CGPA','Chance of Admit ',data=df,ax=ax6)
research_count = df.groupby('Research')['Research'].count()

print(research_count)

labels = ['0:Without','1:With']

plt.pie(research_count,explode=[0,0.05],labels=labels)

plt.legend(loc='lower right')
df1=df.copy()

df1['Research'] = df['Research'].replace([1,0],['Yes','No'])

sns.catplot(x='Chance of Admit ',y='Research',data=df1,kind='violin')
sns.kdeplot(norm_df['Chance of Admit '],shade=True)

sns.kdeplot(norm_df['CGPA'])
sns.heatmap(norm_df.corr(),annot=True)

fig=plt.gcf()

fig.set_size_inches(10,5)
X = norm_df.drop('Chance of Admit ',axis=1)

Y = norm_df['Chance of Admit ']
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()



from sklearn.model_selection import cross_val_score



MSEs = cross_val_score(lin_reg,X,Y,cv=5,scoring='neg_root_mean_squared_error')

mean_MSE = np.mean(MSEs)

print(f' The Negative MSE which is to be maximised is {mean_MSE}')
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



ridge_reg = Ridge()



parameters = {'alpha':[1e-15,1e-8,.002,0.1,0.5,1,5,10,20]}



RMSEs = GridSearchCV(ridge_reg,parameters,scoring='neg_root_mean_squared_error',cv=5)

RMSEs.fit(X,Y)

print(RMSEs.best_params_)

print(f' The Negative MSE which is to be maximised is {RMSEs.best_score_}')
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



lasso_reg = Lasso()



parameters = {'alpha':[.002,0.1,0.5,1,5,10,20]}



LMSEs = GridSearchCV(lasso_reg,parameters,scoring='neg_root_mean_squared_error',cv=5)

LMSEs.fit(X,Y)

print(LMSEs.best_params_)

print(f' The Negative MSE which is to be maximised is {LMSEs.best_score_}')
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(X,Y)
print(f'The intercept of the Linear Model / best fit line is {lreg.intercept_}')

print(f'Number of coefficents is {len(lreg.coef_)}')
coef_df = DataFrame(X.columns,columns=['Feature'])

coef_df['Coeff'] = lreg.coef_

coef_df
sns.catplot(x='Feature',y='Coeff',data=coef_df,kind='point',height=8)
lreg1 = LinearRegression()
norm_df.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
lreg1.fit(x_train,y_train)
y_pred = lreg1.predict(x_test)
rms = np.mean((y_pred-y_test)*2)

print(f'The root-mean-square error of the predicted values from the actual values is {rms}')
pred_val = DataFrame(y_pred,columns=['Predicted'])

pred_val['Actual'] = y_test.values

pred_val
sns.lmplot(x='Predicted',y='Actual',data=pred_val)
sns.scatterplot(x=y_pred,y=(y_pred-y_test),data=pred_val)