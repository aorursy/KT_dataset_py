import numpy as np 

import pandas as pd

import time

import seaborn as sns

import matplotlib.pyplot as plt



from scipy import stats

from scipy.stats import norm, skew



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PowerTransformer, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.linear_model import Lasso, Ridge

from lightgbm import LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor
df = pd.read_csv("/kaggle/input/madrid-airbnb-data/listings.csv")
df.head()
seed = 0
train, test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(23)
def missing_data(data):

    

    missing_datatypes = [j for i in data.columns for j in ['-','?','--','@','NA','NaN','na','Na',' '] 

                         if j in data[i].unique()]



    if len(missing_datatypes) > 0 and data.isnull().values.any() == False:

        print(set(missing_datatypes))



    elif len(missing_datatypes) > 0 and data.isnull().values.any() == True:

        missing_datatypes.append('NaN')

        print(set(missing_datatypes))



    elif len(missing_datatypes) == 0 and data.isnull().values.any() == True:

        print('NaN')



    else:

        print('No missing data founded')
missing_data(train)
train = train[['neighbourhood_group', 'neighbourhood', 'room_type', 'minimum_nights', 'price']]

test = test[['neighbourhood_group', 'neighbourhood', 'room_type', 'minimum_nights', 'price']]
data_distribution = train.hist(figsize=(5,5))
sns.distplot(train['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['price'], plot=plt)
train.corr()['price'].sort_values().drop('price')
#correlation matrix

f, ax = plt.subplots(figsize = (12, 9))

sns.heatmap(train.corr(),annot = False, vmax=.8)
sns.pairplot(train, height = 1.5)

plt.show()
var = 'minimum_nights'

data = pd.concat([train['price'], train[var]], axis=1)

data.plot.scatter(x=var, y='price', alpha=0.3)
data_distribution = train.hist(figsize=(5, 3))
sns.distplot(train['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['price'], plot=plt)
X_train = train.drop(['price'], axis=1)

y_train = train['price'].values



X_test = test.drop(['price'], axis=1)

y_test= test['price'].values
#Because price have zero/negative values, I use a power transform which accept them.



num_cols = X_train._get_numeric_data().columns.tolist()



pt = PowerTransformer(method='yeo-johnson')



X_train[num_cols]= pt.fit_transform(X_train[num_cols])

X_test[num_cols]= pt.transform(X_test[num_cols])



y_train = pt.fit_transform(y_train.reshape(-1, 1))

y_test = pt.transform(y_test.reshape(-1, 1))
data_distribution = X_train.hist(figsize=(5,5))
y_train_plot = train.copy()



y_train_plot['price'] = y_train



sns.distplot(y_train_plot['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(y_train_plot['price'], plot=plt)
le = LabelEncoder()



cat_cols_train = X_train.select_dtypes(include=['string', 'object']).columns.tolist()



cat_cols_test = X_test.select_dtypes(include=['string', 'object']).columns.tolist()





for col in cat_cols_train:

    X_train[col] = le.fit_transform(X_train[col].astype('string'))



# I fit the test dataset because it contains previously unseen labels in the train dataset

for col in cat_cols_test:

    X_test[col] = le.fit_transform(X_test[col].astype('string'))
X_train.head(3)
X_test.head(3)
X_train['price'] = y_train.ravel().tolist()



X_train.drop(X_train[(X_train['price']<-4)].index, inplace=True)



y_train = X_train['price']



X_train.drop('price', axis=1, inplace=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train.values, random_state = seed)



model = Lasso(alpha=0.02)

model.fit(X_train,y_train)



plt.figure(figsize=(10,10))



model_ft_imp = pd.DataFrame(data=model.coef_,columns=['FeatureImp'], index = X_train.columns).sort_values(by='FeatureImp', ascending=False)



model_ft_imp_nonzero = model_ft_imp[model_ft_imp['FeatureImp'] != 0]



sns.barplot(x=model_ft_imp_nonzero['FeatureImp'], y=model_ft_imp_nonzero.index, palette="Reds")



plt.title('Lasso Feature importance', fontsize=20)

plt.show()
def estimator_params(X,y):



    estimator_params = []

    score = []    

    Time = []



    estimators = [GradientBoostingRegressor(),

                   LGBMRegressor(),

                   Ridge(),

                   Lasso()]





    params = [ {'max_depth':[5,10,15], 

                'min_samples_split':[10, 50, 100],

                'learning_rate':[0.01,0.1,0.5], 

                'max_features':['sqrt'],

                'random_state': [seed]},

                            

               {'num_leaves': [5,10,20], 

                'max_depth': [None, 5, 10, 20], 

                'learning_rate': [0.01,0.1,0.5], 

                'n_estimators': [10, 50, 100],

                'random_state': [seed]},

    

                {'alpha': [5, 10, 20, 50,100],

                'tol': [0.5,0.9],

                'random_state': [seed]},

             

                {'alpha' : [0.1, 1],

                 'max_iter': [1000, 2000],

                 'random_state': [seed]}]

    

    # KFold

    

    kf = KFold(n_splits = 5, shuffle=True, random_state = seed)

    

    cv_params = {'cv': kf, 'scoring': 'neg_root_mean_squared_error', 'verbose': 0}





    # GridSearchCV

    

    for estimator,param in zip(estimators, params):

        start = time.time()

        

        grid_solver = GridSearchCV(estimator, param_grid = param, **cv_params).fit(X_train, y_train)



        estimator_params.append(grid_solver.best_estimator_)

        score.append(-(grid_solver.best_score_))

        stop = time.time()

        print('{} optimization finished'.format(str(estimator)))

        print()

        Time.append(stop-start)

        

    global estimator_params_df

    

    estimator_params_df = pd.DataFrame(columns = ['Estimator_params','Score_RMSE','Time'])

    estimator_params_df['Estimator_params']= estimator_params

    estimator_params_df['Score_RMSE'] = score

    estimator_params_df['Time']= Time

    

    return estimator_params_df
estimator_params(X_train, y_train)
estimator_params_df['Estimator_params'][1]
model = LGBMRegressor(max_depth=10, num_leaves=20, random_state=0)



model.fit(X_train,y_train)



predictions = model.predict(X_test)



# Reversing the power transformation

predictions = pt.inverse_transform(predictions.reshape(-1,1))



predictions = np.around(predictions,2).ravel().tolist()
predictions