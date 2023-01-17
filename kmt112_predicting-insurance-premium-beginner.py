import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.preprocessing import StandardScaler

from scipy import stats

from scipy.stats import norm,skew

from matplotlib.pyplot import figure

from sklearn.linear_model import LinearRegression
df=pd.read_csv('../input/insurance-premium-prediction/insurance.csv')

df.head()

df.isna().sum()
df['expenses'].describe()

sns.distplot(df['expenses']);



#plot Skewness and Kurtosis

print("Skewness: %f" % df['expenses'].skew())

print("Kurtosis: %f" % df['expenses'].kurt())
fig =plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(df['expenses'], fit=norm);

(mu,sigma)= norm.fit(df['expenses'])

print('\n mu= {:.2f}\n'.format(mu,sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=${:.2f})'.format(mu,sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('Distribution of Charges')

plt.subplot(1,2,2)

res=stats.probplot(df['expenses'],plot=plt)

plt.suptitle('Before Transformation')



df.expenses=np.log1p(df.expenses)

y=df.expenses.values

y_orig=df.expenses



fig=plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(df['expenses'], fit=norm);

(mu,sigma)=norm.fit(df['expenses'])

print(' \n mu={:.2f} and sigma = {:.2f}\n'.format(mu,sigma))

plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)], loc='best')

plt.title('Distribution of expenses')

plt.subplot(1,2,2)

res=stats.probplot(df['expenses'], plot=plt)

plt.suptitle('After Transformation')

df
#create a dictionary to map the features

bin_dict = {'yes' :1, 'no': 0}

bin_dict_2={'female':1, 'male':0}

#map the category values in our dict

df['smoker']= df['smoker'].map(bin_dict)

df['sex']=df['sex'].map(bin_dict_2)

#check if it has been converted

df.head()
cat_dict = {'southwest':1,'southeast':2, 'northwest':3, 'northeast':4}

#map the category values in our dict



df['region']=df['region'].map(cat_dict)
df.head()
df.duplicated().sum()
df=df.drop_duplicates()
df.age.plot(kind="hist")
df.smoker.plot(kind="hist")
df.children.plot(kind="hist")
plt.figure(figsize = (12,8))

g = sns.countplot(x="age",data=df)

g.set_title("different age groups", fontsize=20)

g.set_xlabel("age", fontsize=15)

g.set_ylabel("count", fontsize=20)
df.region.plot(kind="hist")
# finding correlation

df.corr()

corrMatrix = df.corr()

sns.heatmap(corrMatrix,annot=True)
vars = df.columns

# vars = numerical_features

figures_per_time = 4

count = 0 

y = df['expenses']

for var in vars:

    x = df[var]

    # print(y.shape,x.shape)

    plt.figure(count//figures_per_time,figsize=(25,5))

    plt.subplot(1,figures_per_time,np.mod(count,4)+1)

    plt.scatter(x, y);

    plt.title('f model: T= {}'.format(var))

    count+=1
sns.scatterplot(x=df['bmi'], y=df['expenses'], hue=df['smoker'])
sns.lmplot(x="bmi", y="expenses", hue="smoker", data=df)
cat_df=['sex', 'region', 'smoker']

num_df=['age','bmi','children', 'expenses']

dum_sex = pd.get_dummies(df.sex)

dum_sex.columns = ['female', 'male']

dum_region = pd.get_dummies(df.region)

dum_region.columns = ['southwest','southeast','northwest','northeast']

dum_smoker = pd.get_dummies(df.smoker)

dum_smoker.columns = ['smokeryes','smokerno']

dummies = pd.concat([df, dum_sex,dum_region,dum_smoker],axis='columns')

df = dummies.drop(['sex','smoker','region'],axis='columns')

df.shape
from sklearn.model_selection import train_test_split
x=df.drop(columns='expenses')

y=df[['expenses']]

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=17)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from datetime import datetime

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error , make_scorer, mean_squared_error, r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

from sklearn.linear_model import LinearRegression



from xgboost.sklearn import XGBRegressor

from lightgbm import LGBMRegressor
k_folds = KFold(n_splits =18, shuffle=True, random_state=42)
lightgbm = LGBMRegressor(objective='regression',num_leaves=4,learning_rate=0.01, n_extimators=9000,max_bin=200,bagging_fraction=0.75,bagging_seed=7,feature_fraction=0.2,feature_fraction_seed=7,verbose=-1)
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt,cv=k_folds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7,alphas=alphas2, random_state=42, cv=k_folds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 

                                                        alphas=e_alphas, cv=k_folds,l1_ratio=e_l1ratio))

stack_gen = StackingCVRegressor(regressors=(ridge,lasso,elasticnet,lightgbm), meta_regressor=elasticnet,use_features_in_secondary=True)



svr= make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
print('Elasticnet')

elastic_model = elasticnet.fit(x_train, y_train)

print('Lasso')

lasso_model = lasso.fit(x_train, y_train)

print('Ridge')

ridge_model = ridge.fit(x_train, y_train)

print('lightgbm')

lgb_model_full_data = lightgbm.fit(x_train, y_train)

print('Svr')

svr_model_full_data = svr.fit(x_train, y_train)

print('Stack_gen_model')

stack_gen_model=stack_gen.fit(np.array(x_train), np.array(y_train))

def blend_models_predict(X):

    return ((0.167 * elastic_model.predict(X)) + \

            (0.167 * lasso_model.predict(X)) + \

            (0.167 * ridge_model.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.30 * stack_gen_model.predict(np.array(X))))
model = LinearRegression()

model.fit(x_train,y_train)

train_predict=model.predict(x_train)

test_predict=model.predict(x_test)





blend_train=blend_models_predict(x_train)

blend_train=np.expm1(blend_train.mean(axis=0))

blend_test=blend_models_predict(x_test)

blend_test=np.expm1(blend_test.mean(axis=0))
print("Predicting the train data")

print("Predicting the test data")

print("MAE")

print("Train : ",mean_absolute_error(np.expm1(y_train),blend_train))

print("Test  : ",mean_absolute_error(np.expm1(y_test),blend_test))

print("Train_LR : ",mean_absolute_error(np.expm1(y_train),np.expm1(train_predict)))

print("Test_LR  : ",mean_absolute_error(np.expm1(y_test),np.expm1(test_predict)))

print("====================================")

print("MSE")

print("Train : ",mean_squared_error(np.expm1(y_train),blend_train))

print("Test  : ",mean_squared_error(np.expm1(y_test),blend_test))

print("Train_LR : ",mean_squared_error(np.expm1(y_train),np.expm1(train_predict)))

print("Test_LR  : ",mean_squared_error(np.expm1(y_test),np.expm1(test_predict)))

print("====================================")

print("RMSE")

print("Train : ",np.sqrt(mean_squared_error(np.expm1(y_train),blend_train)))

print("Test  : ",np.sqrt(mean_squared_error(np.expm1(y_test),blend_test)))

print("Train_LR : ",np.sqrt(mean_squared_error(np.expm1(y_train),np.expm1(train_predict))))

print("Test_LR  : ",np.sqrt(mean_squared_error(np.expm1(y_test),np.expm1(test_predict))))

print("====================================")

print("R^2")

print("Train : ",r2_score(np.expm1(y_train),blend_train))

print("Test  : ",r2_score(np.expm1(y_test),blend_test))

print("Train_LR : ",r2_score(y_train,train_predict))

print("Test_LR  : ",r2_score(y_test,test_predict))

plt.figure(figsize=(10,7))

plt.title("Actual vs. predicted expenses",fontsize=25)

plt.xlabel("Actual expenses",fontsize=18)

plt.ylabel("Predicted expenses", fontsize=18)

plt.scatter(x=np.expm1(y_test),y=blend_test, c='b', label='Blended')

plt.scatter(x=np.expm1(y_test),y=np.expm1(test_predict), c='r', label='LinReg')

plt.plot([0,80000], [0,80000], '-g', label='perfect')

plt.legend(loc='upper left')

plt.show()




