import numpy as np  

import pandas as pd  

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

from scipy import stats

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split,KFold,learning_curve

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.head()
cols = df[['age','bmi','children','charges']]

cols.isnull()

cols = df[['sex','smoker','region']]

cols.isna()
le = LabelEncoder()

df1 = df[['sex','smoker','region']]

df[['sex','smoker','region']] = df1.apply(lambda x: le.fit_transform(x))

df
df3 = df.corr(method='pearson').round(2)

df3.loc[df3.index == 'charges']
sns.boxplot(x=df['age'],y=df['charges'])
sns.boxplot(x=df['bmi'],y=df['charges'])
sns.boxplot(x=df['children'],y=df['charges'])
sns.boxplot(x=df['sex'],y=df['charges'])
sns.boxplot(x=df['smoker'],y=df['charges'])
sns.boxplot(x=df['region'],y=df['charges'])
df4 = df[['age','bmi','children','charges','sex','smoker','region']]

z_score = stats.zscore(df4)

sns.distplot(z_score,hist=True);
np.where(z_score>3)
df4 = df4[(z_score<3).all(axis=1)]

df4.shape
sns.boxplot(df4['age'],df4['charges'])
Q1 = df4.quantile(0.25)

Q3 = df4.quantile(0.75)

IQR = Q3-Q1
df5 = df4[~((df4 < (Q1-1.5*IQR))| (df4 > (Q3+1.5*IQR))).any(axis=1)]

df5.shape
sns.boxplot(df5['age'],df5['charges'])
def plot_curve(estimator,x,y,cv= KFold(),m=np.linspace(0.5,1,5)):

                        

                            size,score_train,score_test= learning_curve(estimator,x,y,train_sizes=m)

                            mean_train = np.mean(score_train,axis=1)

                            mean_test = np.mean(score_test,axis=1)

                            std_train = np.std(score_train,axis=1)

                            std_test = np.std(score_test,axis=1)

                            plt.fill_between(size,mean_train - std_train,mean_train + std_train,alpha=0.1)

                            plt.fill_between(size,mean_test - std_test,mean_test + std_test,alpha=0.1)

                            plt.plot(size,mean_train,label='Training samples')

                            plt.plot(size,mean_test,label='Cross-Validation set')

                            plt.xlabel('Training samples')

                            plt.ylabel('Error')

                            plt.legend()

                            plt.title('Learning curve')
def prediction(model):

                        x = df5[['age','bmi','children','sex','smoker','region']]

                        y = df5['charges']

                        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

                        

                        model = model.fit(x_train,y_train)

                        yhat = model.predict(x_test)

                        df_pred = x_test

                        df_pred['charges'] = yhat

                        df_pred['Actual values'] = y_test

                        from sklearn.metrics import accuracy_score,mean_squared_error,r2_score,mean_absolute_error

                        print('MSE:',mean_squared_error(y_test,yhat))

                        print('R2:',r2_score(y_test,yhat))

                        print('MAE:',mean_absolute_error(y_test,yhat))

                        return df_pred




re = XGBRegressor(reg_alpha=0.9)

plot_curve(re,x,y)

prediction(re)


rtree = RandomForestRegressor()

plot_curve(rtree,x,y)

prediction(rtree)


l = Lasso(alpha=0.1)

plot_curve(l,x,y)

prediction(l)


lr = LinearRegression()

plot_curve(lr,x,y)

prediction(lr)


r = Ridge(alpha=0.1)

plot_curve(r,x,y)

prediction(r)