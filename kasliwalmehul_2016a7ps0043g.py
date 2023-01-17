import pandas as pd

import numpy as np

from math import sqrt

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error,make_scorer

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer

import matplotlib.pyplot as plt
dfx = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df = dfx.iloc[:,1:]

df = df.drop(['type'],axis=1)

df.describe()

df = df.dropna()

print(df.shape)

print(df['rating'].value_counts())
corr = df.iloc[:,:12].corr()

corr.style.background_gradient(cmap='coolwarm')
X = df.iloc[:,:11].values

Y = df.iloc[:,-1].values
p = np.random.permutation(len(Y))

Y = np.asarray(Y)

X,Y = X[p], Y[p]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=41)
#scaler = StandardScaler()

scaler = RobustScaler()

#scaler = Normalizer()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val) 
X_val.shape
def RMSE(y_true,y_pred):    

     rmse = sqrt(mean_squared_error(y_true, np.round(y_pred)))

     print("RMSE: %2.3f" % rmse)

     return rmse

    

scorer = make_scorer(RMSE,greater_is_better=False)



param_dist = {

    'max_depth': range(85,105),

    'max_features': ['sqrt'],

    'n_estimators': [900,950,1000,1050,100,1200,1300,1400]

}



random_search = GridSearchCV(

    estimator=ExtraTreesRegressor(),

    param_grid={

        'max_depth': range(93,98),

        'n_estimators': range(858,864),

        'max_features': ['sqrt','log2']}

    ,cv=3,scoring=scorer, verbose=2)
random_result = random_search.fit(X_train,Y_train)
best_params = random_result.best_params_

print(best_params)

print(random_result.best_score_)
ss = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

ss = ss.drop(['type'],axis=1)

X_test_submit = ss.iloc[:,1:]

X_test_submit[pd.isna(X_test_submit).any(axis=1)]

for column in X_test_submit.columns:

    X_test_submit[column].fillna(X_test_submit[column].mode()[0], inplace=True)

X_test_submit = scaler.transform(X_test_submit)
X = np.concatenate((X_train,X_val),axis=0)

Y = np.concatenate((Y_train,Y_val),axis=0)

p = np.random.permutation(len(X))

X,Y = X[p], Y[p]
reg_best = ExtraTreesRegressor(max_depth=100,n_estimators=1400,max_features='sqrt').fit(X_train,Y_train)
Y_predicted = reg_best.predict(X_val)

Y_predicted.shape

rms = sqrt(mean_squared_error(Y_val, np.round(Y_predicted)))

print(rms)
reg_best = ExtraTreesRegressor(max_depth=100,n_estimators=1400,max_features='sqrt').fit(X,Y)

Y_predicted_submit = reg_best.predict(X_test_submit)

Y_predicted_submit.shape
out = pd.DataFrame({'id':ss.iloc[:,0],'rating':np.round(Y_predicted_submit)})

out.to_csv('submission.csv',index=False)
reg_secondbest = ExtraTreesRegressor(max_depth=100,n_estimators=500,max_features='sqrt').fit(X_train,Y_train)
Y_predicted = reg_secondbest.predict(X_val)

Y_predicted.shape

rms = sqrt(mean_squared_error(Y_val, np.round(Y_predicted)))

print(rms)
reg_secondbest = ExtraTreesRegressor(max_depth=100,n_estimators=500,max_features='sqrt').fit(X,Y)

Y_predicted_submit = reg_secondbest.predict(X_test_submit)

Y_predicted_submit.shape
out = pd.DataFrame({'id':ss.iloc[:,0],'rating':np.round(Y_predicted_submit)})

out.to_csv('submission.csv',index=False)
feature_importances = pd.DataFrame(reg_best.feature_importances_,index = range(0,11),columns=['importance']).sort_values('importance',ascending=False)
feature_importances
objects = feature_importances.index

y_pos = np.arange(len(objects))

performance = feature_importances['importance']



plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.show()

plt.scatter(df.iloc[:,-1],df.iloc[:,10])
Y_predicted.mean()
Y_predicted.std()