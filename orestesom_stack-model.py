import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, RobustScaler, MinMaxScaler

from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.pipeline import make_pipeline, Pipeline

from scipy import stats

from scipy.stats import norm, skew

from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso, LinearRegression, BayesianRidge 

from sklearn.ensemble import RandomForestRegressor

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



 

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/szeged-weather/weatherHistory.csv')

df = df.drop(['Loud Cover', 'Apparent Temperature (C)'], axis=1)
display(df.head())
mask = np.zeros_like(df.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(15,10))



sns.heatmap(df.corr(), annot=True, mask=mask, linewidths=0.1, square=True, annot_kws={'size':8}, cmap="BuGn" )
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)



df['day']  = df['Formatted Date'].dt.day

df['month']  = df['Formatted Date'].dt.month



df = df.drop('Formatted Date', axis=1)
f, axes =  plt.subplots(1,2, figsize=(15,5))



sns.boxplot(x='Precip Type', y='Temperature (C)', data=df, palette="Set2", ax=axes[0])



sns.boxplot(x='month', y='Temperature (C)', data=df, palette="Set2", ax=axes[1])

sns.despine(left=True, bottom=True)



f, axe =  plt.subplots(1,1, figsize=(15,5))



sns.boxplot(x='Summary', y='Temperature (C)', data=df, palette="Set2", ax=axe)

plt.setp(axe.get_xticklabels(), rotation=45, fontsize=7)

sns.despine(left=True, bottom=True)
df = df.drop('Summary', axis=1)
sns.distplot(df['Temperature (C)'], fit=norm)

(mu, sigma) = norm.fit(df['Temperature (C)'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Temperature (C)')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df['Temperature (C)'], plot=plt)

plt.show()



print(skew(df['Temperature (C)']))
f, axes =  plt.subplots(1,2, figsize=(15,5))



sns.regplot(x='Humidity', y='Temperature (C)', data=df, ax=axes[0], color='g')

sns.regplot(x='Visibility (km)', y='Temperature (C)', data=df, ax=axes[1])
df['Humidity'] = df['Humidity'].loc[(df['Humidity']>0.0) & (df['Temperature (C)']) > 0]



sns.regplot(x='Humidity', y='Temperature (C)', data=df, color='g')
df_nan = (df.isna().sum() / len(df)) * 100

df_nan = df_nan.drop(df_nan[df_nan == 0].index).sort_values(ascending=False)[:5]

df_nan
df['Humidity'] = df.groupby('Visibility (km)')['Humidity'].transform(lambda x: x.fillna(x.median()))

df['Precip Type'] = df.groupby('Visibility (km)')['Precip Type'].transform(lambda x: x.fillna(x.mode()[0]))
df_nan = (df.isna().sum() / len(df)) * 100

df_nan = df_nan.drop(df_nan[df_nan == 0].index).sort_values(ascending=False)[:5]

df_nan
display(df.dtypes)
categorical_cols = [col for col in df.columns if df[col].dtypes == 'object']

print(categorical_cols)
label_enc = LabelEncoder()



for cols in categorical_cols:

    df[cols] = label_enc.fit_transform(df[cols])





display(df.head())
y = df['Temperature (C)'].values.reshape(-1, 1)

X = df.drop('Temperature (C)', axis=1)
print(X.shape)

print(y.shape)
n_folds = 5



def rmse_cv(model):

    kf = KFold(n_folds, shuffle=True).get_n_splits(X)

    rmse= np.sqrt(-cross_val_score(model, X, y.ravel(), scoring="neg_mean_squared_error", cv = kf))

    return rmse







scores = pd.DataFrame({}, columns=[ 'Model', 'Score'])


lr = make_pipeline(MinMaxScaler(), LinearRegression())



lr_sc = rmse_cv(lr)



print('RMSE in Linear Regression:', lr_sc.mean())



r = scores.shape[0] + 1 



scores.loc[r] = ['Linear Regressor', lr_sc.mean()]



#The score is king of rare and you can thiks it's too high for these values, but no problem it works just fine.


lr_poly_3 = make_pipeline(MinMaxScaler(), PolynomialFeatures(3), LinearRegression())



lr_poly_3_scor = rmse_cv(lr_poly_3)





print('RMSE in Linear Regression Poly 3:', lr_poly_3_scor.mean())





r = scores.shape[0] + 1 



scores.loc[r] = ['Linear Regressor Poly = 3', lr_poly_3_scor.mean()]
#split the data outside the rmse_cv funtion to find the best alpha.





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



#make a Robust Scaler that fit's better for outliders

scaler = RobustScaler()





#make the change

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





#try some alphas with Ridge Cross Validation

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60], store_cv_values=True,

               cv=None).fit(X_train, y_train)





#Create a score of alpha and the alpah itself

score = ridge.score(X_train, y_train)

alpha = ridge.alpha_





print('Best alpha: ', alpha)



print('Try again for more precision centered in: ', alpha)



#Try with some changes the best alpha and check if improves

ridge_2 = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,

                          alpha * 0.9, alpha * .95, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.2,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], cv=None,

                          store_cv_values=True).fit(X_train, y_train)







score_2 = ridge_2.score(X_train, y_train)

alpha_2 = ridge_2.alpha_





best_alpha = alpha



if score_2 > score:

    best_alpha = alpha_2

    

    



print('Best alpha: ', best_alpha)





#Finally the model!

ridge_f = make_pipeline(RobustScaler(), Ridge(alpha=best_alpha))

 

ridge_scor = rmse_cv(ridge_f)





print('Ridge RMSE: ', ridge_scor.mean())





r = scores.shape[0] + 1 



scores.loc[r] = ['Ridge', ridge_scor.mean()]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.001, 0.01, 0.03,0.06, 1], 

                max_iter = 50000, cv = 10).fit(X_train, y_train.ravel())







alpha = lasso.alpha_





print('Best alpha: ', alpha)

print('Try again for more precision centered in: ', alpha)



lasso_2 = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,

                       alpha * 0.9, alpha * .95, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.2,

                        alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],  

                  max_iter = 50000, cv = 10).fit(X_train, y_train.ravel())





alpha_2 = lasso_2.alpha_



best_alpha = alpha



if alpha_2 > alpha:

    best_alpha = alpha_2

    









print('Best alpha: ', best_alpha)



lasso_f = make_pipeline(RobustScaler(), Lasso(alpha=best_alpha))



lasso_scor = rmse_cv(lasso_f)







print('Lasso RMSE: ', lasso_scor.mean())



r = scores.shape[0] + 1 



scores.loc[r] = ['Lasso', lasso_scor.mean()]
bayes = make_pipeline(RobustScaler(), BayesianRidge())



bayes_scor = rmse_cv(bayes)



r = scores.shape[0] + 1 





print('Bayes RMSE: ', bayes_scor.mean())

scores.loc[r] = ['reg', bayes_scor.mean()]
RFR = make_pipeline(MinMaxScaler(), RandomForestRegressor(n_estimators=350))



rfr_scor = rmse_cv(RFR)





print('Random Forest Regressor: ', rfr_scor.mean())



#As I expected the score is better!
r = scores.shape[0] + 1 

scores.loc[r] = ['Random forest reggresor', rfr_scor.mean()]
#Split the data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models=(RFR, lr_poly_3))



averaged_model_scor = rmse_cv(averaged_models)



r = scores.shape[0] + 1 



scores.loc[r] = ['Stacked RFR-LR', averaged_model_scor.mean()]
display(scores.sort_values(by='Score'))

averaged_models.fit(X_train, y_train.ravel())



predictions = averaged_models.predict(X_test)



mean_sq_error = np.sqrt(mean_squared_error(predictions, y_test))





print(mean_sq_error)

print(predictions[:10].round(0))

print(y_test[:10].round(0))