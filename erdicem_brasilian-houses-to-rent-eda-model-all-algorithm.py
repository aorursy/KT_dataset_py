import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm 
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
df.head()
df.shape
df.info()
df.describe().T
df.floor.unique()
df["floor"] = df["floor"].apply(str.strip).replace("-", np.nan)
df["floor"] = pd.to_numeric(df["floor"], downcast="float")
df.info()
df.isnull().sum()*100/df.shape[0]
df["floor"].fillna(np.mean(df.floor), inplace=True)
df.isnull().sum()*100/df.shape[0]
plt.figure(figsize=(8,5))
plt.boxplot(df['total (R$)'])
plt.show()
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

title_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }
curve_font  = {'family': 'arial', 'color': 'darkblue','weight': 'bold','size': 10 }

plt.figure(figsize=(38,28))

columns=[ 'area', 'rooms', 'bathroom', 'parking spaces', 'floor',
         'hoa (R$)', 'rent amount (R$)',
       'property tax (R$)', 'fire insurance (R$)', 'total (R$)']

for i in range(10):
    plt.subplot(5, 10, i+1)
    plt.hist(df[columns[i]])
    plt.title(columns[i]+str("/Orjinal")  , fontdict=title_font)
for i in range(10):
    plt.subplot(5, 10, i+11)
    plt.hist(winsorize(df[columns[i]], (0, 0.10)))
    plt.title(columns[i]+str("/Winsorize")  , fontdict=title_font)
for i in range(10):
    plt.subplot(5, 10, i+21)
    plt.hist(np.log(df[columns[i]]+1))
    plt.title(columns[i]+str("/Log")  , fontdict=title_font)
for i in range(10):
    plt.subplot(5, 10, i+31)
    plt.hist(scale(df[columns[i]]))
    plt.title(columns[i]+str("/Scale")  , fontdict=title_font)
for i in range(10):
    plt.subplot(5, 10, i+41)
    plt.hist(normalize(np.array(df[columns[i]]).reshape(1,-1).reshape(-1,1)))
    plt.title(columns[i]+str("/Normalize")  , fontdict=title_font)
plt.figure(figsize=(38,28))

columns=[ 'area', 'rooms', 'bathroom', 'parking spaces', 'floor',
         'hoa (R$)', 'rent amount (R$)',
       'property tax (R$)', 'fire insurance (R$)', 'total (R$)']

for i in range(10):
    plt.subplot(5, 10, i+1)
    plt.boxplot(df[columns[i]])
    plt.title(columns[i]+str("/Orjinal")  , fontdict=title_font)
for i in range(10):
    plt.subplot(5, 10, i+11)
    plt.boxplot(winsorize(df[columns[i]], (0, 0.03))) # %95
    plt.title(columns[i]+str("/Winsorize")  , fontdict=title_font)
for i in range(10):
    plt.subplot(5, 10, i+21)
    plt.boxplot(np.log(df[columns[i]]+1))
    plt.title(columns[i]+str("/Log")  , fontdict=title_font)
for i in range(10):
    plt.subplot(5, 10, i+31)
    plt.boxplot(scale(df[columns[i]]))
    plt.title(columns[i]+str("/Scale")  , fontdict=title_font)
for i in range(10):
    plt.subplot(5, 10, i+41)
    plt.boxplot(normalize(np.array(df[columns[i]]).reshape(1,-1).reshape(-1,1)))
    plt.title(columns[i]+str("/Normalize")  , fontdict=title_font)
log_threshold_variables= pd.DataFrame()
variables = [ 'area', 'rooms', 'bathroom', 'parking spaces', 'floor',
         'hoa (R$)', 'rent amount (R$)',
       'property tax (R$)', 'fire insurance (R$)', 'total (R$)']
for j in variables:
    for threshold_worth in np.arange(1,5,1):
        
        #logarithm Transformed
        q75_log, q25_log = np.percentile(np.log(df[j]), [75 ,25])
        caa_log = q75_log - q25_log
        
        #Orjinal Data
        q75, q25 = np.percentile(df[j], [75 ,25])
        caa= q75 - q25
        
        # Winsorize Data
        q75_win, q25_win = np.percentile(winsorize(df[j],(0, 0.03)), [75 ,25])
        caa_win= q75 - q25
        
        #logarithm Transformed
        min_worth_log = q25_log - (caa_log*threshold_worth)
        max_worth_log = q75_log + (caa_log*threshold_worth)
        
        #Orjinal Data
        min_worth= q25 - (caa*threshold_worth) 
        max_worth = q75 + (caa*threshold_worth) 
        
        # Winsorize Data
        min_worth_win= q25_win - (caa_win*threshold_worth) 
        max_worth_win = q75_win + (caa_win*threshold_worth)
        
        number_of_outliers_log = len((np.where((np.log(df[j]) > max_worth_log)| 
                                               (np.log(df[j]) < min_worth_log))[0]))
        
        number_of_outliers = len((np.where((df[j] > max_worth)| 
                                               (df[j] < min_worth))[0]))
        
        number_of_outliers_win = len((np.where((winsorize(df[j],(0, 0.03)) > max_worth_win)| 
                                               (winsorize(df[j],(0, 0.03)) < min_worth_win))[0]))
        
        log_threshold_variables = log_threshold_variables.append({'threshold_worth': threshold_worth,
                                                            'number_of_outliers' : number_of_outliers, 
                                                            'number_of_outliers_log': number_of_outliers_log,
                                                            "number_of_outliers_win":number_of_outliers_win
                                                            }, ignore_index=True)
    print("-"*10,"",j,"-"*10)
    display(log_threshold_variables)
    log_threshold_variables = pd.DataFrame()
plt.boxplot(df['total (R$)'], whis=4) # Not log transformed whis = 4 
plt.show()
plt.boxplot(np.log(df['total (R$)']), whis=4) # log Transformed whis = 4 
plt.show()
df["city"]= df["city"].replace({"São Paulo":0, 'Porto Alegre':1, 'Rio de Janeiro':2, 'Campinas':3,'Belo Horizonte':4})
df['animal']= pd.get_dummies(df['animal'],drop_first=True)
df['furniture']= pd.get_dummies(df['furniture'],drop_first=True)
df_log= df[['area','rooms',"bathroom","parking spaces","floor","hoa (R$)","rent amount (R$)","property tax (R$)","fire insurance (R$)","total (R$)"]]

df_add=df[['city',"animal","furniture"]] # Not Transform Log because we apply get_dummies() these columns

df_log = np.log(df_log+1) # We don't want taking -inf values in dataframe. 


df_log=pd.concat([df_log,df_add],axis=1)
df_log.head()
# IQR* 4 Log Transform

q1 = df_log['total (R$)'].quantile(0.25)
q3 = df_log['total (R$)'].quantile(0.75)
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+4*iqr #acceptable range
low,high
# IQR* 4 Log Transform Boxplot

df_log['total (R$)']=np.where(df_log['total (R$)'] > high,high,df_log['total (R$)']) # upper limit

plt.boxplot(df_log['total (R$)'])
plt.show()
df_corr=df_log.corr()
df_corr
plt.figure(figsize=(18,10))
ax=sns.heatmap(df_corr, square=True, annot=True, linewidths=.5, vmin=0, vmax=1, cmap='viridis')
ax.set_ylim(13,0)
plt.title("Correlation Matrix", fontdict=title_font)

plt.show()
df_log=df_log.drop(["rent amount (R$)","property tax (R$)","fire insurance (R$)","hoa (R$)"], axis=1)
df_log.head()
df_win = df.copy()
df_win['total (R$)'] = winsorize(df['total (R$)'], (0, 0.03))
df_win=df_win.drop(["rent amount (R$)","property tax (R$)","fire insurance (R$)","hoa (R$)"], axis=1)
df_win.head()
df_win_1 = df.copy()
df_win_1['total (R$)'] = winsorize(df['total (R$)'], (0, 0.03))
df_win_1=df_win_1.drop(["rent amount (R$)","property tax (R$)","fire insurance (R$)","hoa (R$)"], axis=1)
df_win_1.head()
# IQR* 1 Log Transform
q1 = df_win_1['total (R$)'].quantile(0.25)
q3 = df_win_1['total (R$)'].quantile(0.75)
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+1*iqr #acceptable range
low,high
df_win_1['total (R$)']=np.where(df_win_1['total (R$)'] > high,high,df_win_1['total (R$)']) # upper limit
df_win_2 = df.copy()
df_win_2['total (R$)'] = winsorize(df['total (R$)'], (0, 0.03))
df_win_2=df_win_2.drop(["rent amount (R$)","property tax (R$)","fire insurance (R$)","hoa (R$)"], axis=1)
df_win_2.head()
# IQR* 1 Log Transform
q1 = df_win_2['total (R$)'].quantile(0.25)
q3 = df_win_2['total (R$)'].quantile(0.75)
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+2*iqr #acceptable range
low,high
df_win_2['total (R$)']=np.where(df_win_2['total (R$)'] > high,high,df_win_2['total (R$)']) # upper limit
df_log.to_csv('Log_Brazil')
df_win.to_csv('Winsorize_Brazil')
from sklearn.metrics import mean_squared_error ,r2_score,explained_variance_score,max_error
from sklearn.model_selection import train_test_split, cross_val_score ,cross_val_predict,GridSearchCV, cross_validate
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm 
def create_model(X,y,model,tip):
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=111)
    model.fit(X_train, y_train)
    
    prediction_train=model.predict(X_train)
    prediction_test=model.predict(X_test)
    
    cv = cross_validate(estimator=model,X=X,y=y,cv=10,return_train_score=True)
    
    d = pd.Series({'mean_squared_error_train':mean_squared_error(y_train,prediction_train),
                   'mean_squared_error_test':mean_squared_error(y_test,prediction_test),
                   'RMSE Train':np.sqrt(mean_squared_error(y_train,prediction_train)),
                   'RMSE Test':np.sqrt(mean_squared_error(y_test,prediction_test)),
                   'r2_score_train':r2_score(y_train,prediction_train),
                   'r2_score_test':r2_score(y_test,prediction_test),
                   'explained_variance_score_train':explained_variance_score(y_train,prediction_train),
                   'explained_variance_score_test':explained_variance_score(y_test,prediction_test),
                   'max_error_train':max_error(y_train,prediction_train),
                   'max_error_test':max_error(y_test,prediction_test),
                   "Cross_val_train":cv['train_score'].mean(),
                   "Cross_val_test":cv['test_score'].mean()
                  },name=tip)
    return d
X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

X_train, X_test, y_train, y_test =  train_test_split(X_scl, y, test_size=0.20, random_state=111) 
      
lm = LinearRegression()
lm.fit(X_train, y_train)
metrics=pd.DataFrame()
metrics=metrics.append(create_model(X_scl,y,lm,tip='Log_IQR*4'))
metrics
X = df_win.drop(["total (R$)"], axis=1)
y = df_win['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

lm = LinearRegression()

metrics=metrics.append(create_model(X_scl,y,lm,tip='Winsorize_IQR*0'))
metrics
X = df_win_1.drop(["total (R$)"], axis=1)
y = df_win_1['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

lm = LinearRegression()

metrics=metrics.append(create_model(X_scl,y,lm,tip='Winsorize_IQR*1'))
metrics
X = df_win_2.drop(["total (R$)"], axis=1)
y = df_win_2['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

lm = LinearRegression()

metrics=metrics.append(create_model(X_scl,y,lm,tip='Winsorize_IQR*2'))
metrics
X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

lm = LinearRegression()

models= pd.DataFrame()
models=models.append(create_model(X_scl,y,lm,tip='Linear_Model'))
models
X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

Knn=KNeighborsRegressor(n_neighbors=5)

models=models.append(create_model(X_scl,y,Knn,tip='Knn_model'))
models
Knn=KNeighborsRegressor()
k_range = list(range(1,25))
parameter = dict(n_neighbors=k_range)
grid = GridSearchCV(Knn, parameter, cv=10, scoring='r2')
Grds = grid.fit(X,y)
print('The best parameters:', Grds.best_estimator_)
print('The best score:', Grds.best_score_)
Knn = KNeighborsRegressor(n_neighbors=23)

models=models.append(create_model(X_scl,y,Knn,tip='Knn_model_Tuning'))
models
X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

cart_model = DecisionTreeRegressor()

models=models.append(create_model(X_scl,y,cart_model,tip='cart_model'))
models
cart_params= {'min_samples_split':range(2,20), 
             "max_leaf_nodes":range(2,10),
             "max_features":range(0,5)}

cart_cv_model = GridSearchCV(cart_model, cart_params, cv=10)

cart_cv_model.fit(X_train,y_train)
print("The best Parameters"+str(cart_cv_model.best_params_))
cart_model = DecisionTreeRegressor(max_features=4 , max_leaf_nodes=9 , min_samples_split=6 )

models=models.append(create_model(X_scl,y,cart_model,tip='cart_model_tuning'))
models
X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

random_model = RandomForestRegressor(n_estimators=25, random_state=2)

models=models.append(create_model(X_scl,y,random_model,tip='random_model'))
models
rf_params = {'max_depth': [2,3,5,8,10],
            "max_features":[1,2,3,4],
            "min_samples_split":[2,5,40]}

rf_cv_model= GridSearchCV(random_model, rf_params , cv=10)

rf_cv_model.fit(X_train,y_train)
print("The best paramters :"+str(rf_cv_model.best_params_))
random_model = RandomForestRegressor(n_estimators=25, random_state=2,max_depth=10, max_features=4 ,min_samples_split=2)

models=models.append(create_model(X_scl,y,random_model,tip='random_model_tuning'))
models
plt.figure(figsize=(20,10))
importance_level = pd.Series(data=random_model.feature_importances_,
                        index= X.columns)

importance_level_sorted = importance_level.sort_values()

importance_level_sorted.plot(kind='barh', color='darkblue')
plt.title('Importance Level of the Features')
plt.show()
from sklearn.svm import SVR

X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

svm_model = SVR()

models=models.append(create_model(X_scl,y,svm_model,tip='svm_model'))
models
svr_params = {"C": np.arange(0.1, 2, 0.1)}

svr_cv_model = GridSearchCV(svm_model,svr_params, cv=10 ).fit(X_train,y_train)

print("The Best Parameters :"+str(svr_cv_model.best_params_))
svm_model = SVR(C=1.9000000000000001)

models=models.append(create_model(X_scl,y,svm_model,tip='svm_model_Tuning'))
models
from sklearn.ensemble import BaggingRegressor

X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

bag_model = BaggingRegressor(bootstrap_features=True)

models=models.append(create_model(X_scl,y,bag_model,tip='bag_model'))
models
bag_params = {"n_estimators": range(2,20)}

bag_cv_model = GridSearchCV(bag_model,bag_params, cv=10 ).fit(X_train,y_train)

bag_cv_model.best_params_
bag_model = BaggingRegressor(n_estimators=19, random_state=45)

models=models.append(create_model(X_scl,y,bag_model,tip='bag_model_tuning'))
models
from xgboost import XGBRegressor

X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler= StandardScaler()
X_scl= scaler.fit_transform(X)

xgb = XGBRegressor(base_score=0.5, verbose=False)

models=models.append(create_model(X_scl,y,xgb,tip='xgb_model'))
models
xgb_params = {'colsample_bytree':[0.4,0.5,0.6, 0.9, 1],
             "n_estimators":[100,200,500,1000],
             "max_depth":[2,3,4,5,6],
             "learning_rate":[0.1,0.01,0.5]}

xgb_cv_model = GridSearchCV(xgb,xgb_params, cv=10, n_jobs= -1 , verbose=False )

xgb_cv_model.fit(X_train,y_train)
print("The Best Parameters :"+str(xgb_cv_model.best_params_))
xgb = XGBRegressor(base_score=0.5,colsample_bytree=0.5,learning_rate=0.01
                   ,max_depth=6,n_estimators=1000,verbose=False )

models=models.append(create_model(X_scl,y,xgb,tip='xgb_model_tuning'))
models
from lightgbm import LGBMRegressor

X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler= StandardScaler()
X_scl= scaler.fit_transform(X)

lgbm = LGBMRegressor()

models=models.append(create_model(X_scl,y,lgbm,tip='lgbm_model'))
models
lgbm_params = {'colsample_bytree':[0.4,0.5,0.6, 0.9, 1],
             "n_estimators":[100,200,500,1000],
             "max_depth":[2,3,4,5,6],
             "learning_rate":[0.1,0.01,0.5]}

lgbm_cv_model = GridSearchCV(lgbm,lgbm_params, cv=10, n_jobs= -1 , verbose=False )

lgbm_cv_model.fit(X_train,y_train)
print("The Best Parameters :"+str(lgbm_cv_model.best_params_))
lgbm = LGBMRegressor(colsample_bytree=0.6,learning_rate=0.01
                   ,max_depth=6,n_estimators=1000 )

models=models.append(create_model(X_scl,y,lgbm,tip='lgbm_model_tuning'))
models
from catboost import CatBoostRegressor

X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler= StandardScaler()
X_scl= scaler.fit_transform(X)

catboost_model = CatBoostRegressor(verbose=False)

models=models.append(create_model(X_scl,y,catboost_model,tip='catboost_model'))
models
catb_params = {
    "iterations":[200,500,1000],
    "learning_rate":[0.01,0.03,0.05,0.1],
    "depth":[3,4,5,6,7,8]}

cat_cv_model = GridSearchCV(catboost_model,catb_params, cv=10, n_jobs= -1 ,verbose=False )

cat_cv_model.fit(X_train,y_train)
print("The Best Parameters :"+str(cat_cv_model.best_params_))
catboost_model = CatBoostRegressor(depth=7 , iterations=500, learning_rate= 0.05,verbose=False)

models=models.append(create_model(X_scl,y,catboost_model,tip='catboost_model_tuning'))
models
X = df_log.drop(["total (R$)"], axis=1)
y = df_log['total (R$)']

scaler= StandardScaler()
X_scl= scaler.fit_transform(X)

X_train, X_test, y_train, y_test =  train_test_split(X_scl, y, test_size=0.20, random_state=111)

catboost_model = CatBoostRegressor(depth=7 , iterations=500, learning_rate= 0.05,verbose=False).fit(X,y)

print(catboost_model.get_scale_and_bias())
catboost_model.get_feature_importance(data=None,prettified=False,thread_count=-1,verbose=False,)
print("Total = '8.243122100830078 + area * 26.03681985 + rooms * 5.60436687 + bathroom * 13.64171516 + parking spaces * 7.6913577 + floor * 18.73900675 +" 
      ,"city *17.16975768 + animal * 1.34374804 + furniture * 9.77322794  ")
plt.figure(figsize=(20,10))
importance_level = pd.Series(data=catboost_model.feature_importances_,
                        index= X.columns)

importance_level_sorted = importance_level.sort_values()

importance_level_sorted.plot(kind='barh', color='darkblue')
plt.title('Importance Level of the Features')
plt.show()