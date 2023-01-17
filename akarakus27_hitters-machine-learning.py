import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('../input/hitters/Hitters.csv')
df.head()
df.shape
df.tail()
df.columns
df.sort_values('Salary',ascending=False)
df.isnull().sum()
df['Experience'] = pd.cut(df['Years'],4)

pd.cut(df['Years'],4).value_counts()
df['Experience'] = pd.cut(df['Years'],[0,5,10,15,25],labels=[1,2,3,4])
df.groupby(['League','Division', 'Experience']).agg({'Salary':'mean'})
df['Salary'] = df['Salary'].fillna(df.groupby(['League','Division', 'Experience'])['Salary'].transform('mean'))
df.describe([0.01, 0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
df.shape
df.head()
 

num_features = df.select_dtypes(['int64']).columns

for feature in num_features:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    
    IQR = Q3-Q1
    
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR
    
    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):
        print(feature," : " + str(df[(df[feature] > upper) | (df[feature] < lower)].shape[0]))
    else:
        print(feature, " : 0")
        
  
df.shape
from sklearn.neighbors import LocalOutlierFactor

clf=LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(df[num_features])
df_scores=clf.negative_outlier_factor_
df_scores= np.sort(df_scores)
df_scores[0:20]
sns.boxplot(df_scores);
threshold=np.sort(df_scores)[7]
print(threshold)
df = df.loc[df_scores > threshold]
df = df.reset_index(drop=True)
df.shape
df.info()
cat_features = ['League','Division','NewLeague'] 
num_features = list(df.select_dtypes(['int64']).columns)
cat_features
corr = df.corr()
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr,annot=True)
plt.show()
for col in num_features:
    #sns.scatterplot(x=col ,y='Salary',data=df,hue='League')
    sns.jointplot(x =col, y = 'Salary', data = df, kind = "reg")
    plt.show()
df.groupby('League').mean().T
df.groupby('Division').mean().T
df.groupby('NewLeague').mean().T
for col in cat_features:
    print('Exploring {} feature'.format(col.upper()))
    print(df[col].value_counts(normalize=True,ascending=False))
    sns.barplot(x=col, y="Salary", data=df)
    plt.show()
sns.scatterplot(x=df['CHits']/df['Hits'] ,y='Salary',data=df,hue='League')
plt.show()
df.head()
df['Experience'] = pd.cut(df['Years'],[0,5,10,15,25],labels=[1,2,3,4])
df['Experience'].value_counts()
df.head()
sns.lineplot(x='Experience', y="Salary", data=df, estimator=np.mean)
df['CRBI_bins'] = pd.cut(df['CRBI'],4,labels=[1,2,3,4])
cat_features.extend(['Experience','CRBI_bins'])
cat_features
df.info()
df['New_HitRate']=df["CAtBat"]/df["CHits"]
df['New_AtBat']=df["CAtBat"]/df["AtBat"]
df['New_RBI']=df["CRBI"]/df["RBI"]
df['New_Walks']=df["CWalks"]/df["Walks"]
df['New_Hits']=df["CHits"]/df["Hits"]
df['New_HmRun']=df["CHmRun"]/df["HmRun"]
df['New_Runs']=df["CRuns"]/df["Runs"]
df['New_ChmrunRate']=df["CHmRun"]/df["CHits"]
df['New_Cat']=df["CAtBat"]/df["CRuns"]
df['New_Assist']=df["Hits"]/df["Assists"]
 
num_features.extend(['New_HitRate','New_RBI','New_Walks','New_Hits','New_HmRun','New_Runs','New_ChmrunRate','New_AtBat','New_Cat','New_Assist'])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
df = pd.get_dummies(df, columns = cat_features, drop_first = True)
df.head()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
from sklearn.preprocessing import StandardScaler, MinMaxScaler

std_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

df[num_features] = std_scaler.fit_transform(df[num_features])
df.head()
y = df["Salary"]
X = df.drop('Salary', axis=1)
from sklearn.feature_selection import RFECV #Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
from sklearn.ensemble import RandomForestRegressor

def select_features(X,y):
    # numerik olmayan degiskenlerin silinmesi
    X = X.select_dtypes([np.number]).dropna(axis=1)
    
    clf = RandomForestRegressor(random_state=46)
    clf.fit(X, y)
    
    selector = RFECV(clf,cv=10)
    selector.fit(X, y)
    
    features = pd.DataFrame()
    features['Feature'] = X.columns
    features['Importance'] = clf.feature_importances_
    features.sort_values(by=['Importance'], ascending=False, inplace=True)
    features.set_index('Feature', inplace=True)
    features.plot(kind='bar', figsize=(12, 5))
    
    
    best_columns = list(X.columns[selector.support_])
    print("Best Columns \n"+"-"*12+"\n{}\n".format(best_columns))
    
    return best_columns
best_features = select_features(X,y)
best_features
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=46)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
lr_model = LinearRegression()
lr_model
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lr_rmse
lr_cv_rmse =  np.sqrt(np.mean(-cross_val_score(lr_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
lr_cv_rmse
np.sqrt(-cross_val_score(lr_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error"))
coefs = pd.DataFrame(lr_model.coef_, index = X_train.columns)
coefs
intercept = lr_model.intercept_
intercept
ridge_model = Ridge()
ridge_model
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
ridge_rmse
ridge_cv_rmse =  np.sqrt(np.mean(-cross_val_score(ridge_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
ridge_cv_rmse
np.sqrt(-cross_val_score(ridge_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error"))
pd.Series(ridge_model.coef_, index = X_train.columns)
lasso_model = Lasso()
lasso_model
lasso_model.fit(X_train,y_train)
y_pred = lasso_model.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lasso_rmse
lasso_cv_rmse =  np.sqrt(np.mean(-cross_val_score(lasso_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
lasso_cv_rmse 
np.sqrt(-cross_val_score(lasso_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error"))
pd.Series(lasso_model.coef_, index = X_train.columns)
elasticnet_model = ElasticNet()
elasticnet_model
elasticnet_model.fit(X_train, y_train)
y_pred = elasticnet_model.predict(X_test)
elasticnet_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
elasticnet_rmse
elasticnet_cv_rmse =  np.sqrt(np.mean(-cross_val_score(elasticnet_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
elasticnet_cv_rmse 
np.sqrt(-cross_val_score(elasticnet_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error"))
pd.Series(elasticnet_model.coef_, index = X_train.columns)
ridge_params = {'alpha' :10**np.linspace(10,-2,100)*0.5 ,
                'solver' : ['auto', 'svd', 'cholesky', 'lsqr']}
ridge_model = Ridge()
ridge_gridcv_model = GridSearchCV(estimator=ridge_model, param_grid=ridge_params, cv=10, n_jobs=-1, verbose=2).fit(X_train,y_train)
ridge_gridcv_model.best_params_
ridge_tuned_model = Ridge(**ridge_gridcv_model.best_params_)
ridge_tuned_model.fit(X_train, y_train)
y_pred = ridge_tuned_model.predict(X_test)
ridge_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
ridge_tuned_rmse
ridge_tuned_cv_rmse =  np.sqrt(np.mean(-cross_val_score(ridge_tuned_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
ridge_tuned_cv_rmse 
ridge_model = Ridge()
coefs = []

for i in 10**np.linspace(10,-2,100)*0.5 :
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    print(mean_squared_error(y_test, y_pred, squared=False))
lasso_params = {'alpha':np.linspace(0,1,1000)}

lasso_model = Lasso(tol = 0.001)
lasso_gridcv_model = GridSearchCV(estimator=lasso_model, param_grid = lasso_params, cv=10, n_jobs=-1, verbose=2).fit(X_train,y_train)
lasso_gridcv_model.best_params_
lasso_tuned_model = Lasso(**lasso_gridcv_model.best_params_)
lasso_tuned_model.fit(X_train, y_train)
y_pred = lasso_tuned_model.predict(X_test)
lasso_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lasso_tuned_rmse
lasso_tuned_cv_rmse =  np.sqrt(np.mean(-cross_val_score(lasso_tuned_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
lasso_tuned_cv_rmse
elasticnet_params = {"l1_ratio": [0.1,0.4,0.5,0.6,0.8,1],
                     "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1],
                    }
elasticnet_model = ElasticNet()
elasticnet_gridcv_model = GridSearchCV(estimator=elasticnet_model, param_grid=elasticnet_params, cv=10, n_jobs=-1, verbose=2).fit(X_train,y_train)
elasticnet_gridcv_model.best_params_
elasticnet_tuned_model = ElasticNet(**elasticnet_gridcv_model.best_params_)
elasticnet_tuned_model.fit(X_train, y_train)
y_pred = elasticnet_tuned_model.predict(X_test)
elasticnet_tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
elasticnet_tuned_rmse
elasticnet_tuned_cv_rmse =  np.sqrt(np.mean(-cross_val_score(elasticnet_tuned_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
elasticnet_tuned_cv_rmse 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

def select_model(X,y):
   
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=46)
    
    models = [ 
        {
            "name": "RidgeRegression",
            "estimator": Ridge(),
            "hyperparameters":
                {
                 'alpha' :np.linspace(0,1,100) ,
                 'solver' : ['auto', 'svd', 'cholesky', 'lsqr']
                }
        },
        {
            "name": "LassoRegression",
            "estimator": Lasso(),
            "hyperparameters":
                {
                 'alpha' :np.linspace(0,1,100) ,
                }
        },
        {
            "name": "ElasticNetRegression",
            "estimator": ElasticNet(),
            "hyperparameters":
                {
                 "l1_ratio": np.linspace(0,1,30), # [0.1,0.4,0.5,0.6,0.8,1],
                 "alpha":np.linspace(0,1,100), # [0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]
                }
        },
       
    ]

    for model in models:
        print(model['name'])
        print('-'*len(model['name']))

        grid = GridSearchCV(model["estimator"],
                            param_grid=model["hyperparameters"],
                            cv=10,scoring="neg_mean_squared_error")
        grid.fit(X_train, y_train)
        
        model["best_params"] = grid.best_params_
        #model["best_score"] = grid.best_score_
        model["tuned_model"] = grid.best_estimator_
        
        model["train_rmse_score"] = np.sqrt(mean_squared_error(y_train, model["tuned_model"].fit(X_train,y_train).predict(X_train)))
        model["validation_rmse_score"] = np.sqrt(np.mean(-cross_val_score(model["tuned_model"], X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
        model["test_rmse_score"] = np.sqrt(mean_squared_error(y_test, model["tuned_model"].fit(X_train,y_train).predict(X_test)))
      
        #print("Best ......... Score: {}".format(model["best_score"]))
        print("Best TRAIN RMSE Score: {}".format(model["train_rmse_score"]))
        print("Best VALIDATION RMSE Score: {}".format(model["validation_rmse_score"]))
        print("Best TEST RMSE Score: {}".format(model["test_rmse_score"]))
        print("Best Parameters: {}\n".format(model["best_params"]))

select_model(X,y)
