#1.0 Clear memory
%reset -f



# 1.1 Call data manipulation libraries
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

# 1.3 Dimensionality reduction
from sklearn.decomposition import PCA

from sklearn import preprocessing 
# 1.3 Data transformation classes
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler 
 


# 1.4 Data splitting and model parameter search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBClassifier

# 1.6 Model pipelining
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer


# 1.7 Model evaluation metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix

# 1.8
import matplotlib.pyplot as plt
from xgboost import plot_importance



# 1.9 RandomForest modeling
from sklearn.ensemble import RandomForestClassifier 

# 2.0 Misc
import os, gc

from scipy.stats import uniform

#Graphing
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import LogNorm

# to display all outputs of one cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#hide warning
import warnings
warnings.filterwarnings('ignore')
os.chdir("/kaggle/input/rossmann-store-sales")
os.listdir()
dftrain=pd.read_csv("train.csv")
dfstore=pd.read_csv('store.csv')
dftrain.head()
dfstore.head()

dftrain.columns[dftrain.isnull().any()]

#no column has null value so need to fix null values


dfstore.columns[dfstore.isnull().any()]
dfstore.isnull().sum()
df_train_store = dftrain.merge(dfstore, on = 'Store', copy = False)
df_train_store=df_train_store[df_train_store.Open != 0]
df_train_store=df_train_store[df_train_store.Sales > 0]
df_train_store.shape
#decreased by 2 lakh approx
#get year month day from date column & drop column
import calendar
df_train_store['Date']=pd.to_datetime(df_train_store['Date'])
df_train_store['Year']=df_train_store['Date'].dt.year
df_train_store['month']=df_train_store['Date'].dt.month
df_train_store['weekofyear']=df_train_store['Date'].dt.weekofyear
df_train_store['month_name']=df_train_store['month'].apply(lambda x: calendar.month_abbr[x])


#df_train_store.drop(['Date'], axis = 1, inplace= True)
df_train_store['CompetitionOpen'] = 12 * (df_train_store.Year - df_train_store.CompetitionOpenSinceYear) + (df_train_store.month - df_train_store.CompetitionOpenSinceMonth)
df_train_store['CompetitionOpen'] = df_train_store.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
df_train_store.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis = 1,  inplace = True)

df_train_store['PromoOpen'] = 12 * (df_train_store.Year - df_train_store.Promo2SinceYear) + (df_train_store.weekofyear - df_train_store.Promo2SinceWeek) / float(4)
df_train_store['PromoOpen'] = df_train_store.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
df_train_store.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis = 1,  inplace = True)


#drop na values
df_train_store.dropna(inplace = True)

#convert PromoInterval and month_name in string
df_train_store['PromoInterval']=df_train_store['PromoInterval'].astype(str)


def checkpromomonth(row):
 if (row['month_name'] in row['PromoInterval']):
    return 1
 else:
    return 0
df_train_store['IsPromoMonth'] =  df_train_store.apply(lambda row: checkpromomonth(row),axis=1)

#Drop Date,month_name,PromoInterval
df_train_store.drop(['Date', 'month_name','PromoInterval'], axis = 1,  inplace = True)
#convert num columns into float 32
df_train_store.dtypes.value_counts()
num_columns= df_train_store.select_dtypes(exclude=[object]).columns 
cat_columns=df_train_store.select_dtypes(include=[object]).columns 
for col in num_columns:
    df_train_store[col]=df_train_store[col].astype('float32')

le = preprocessing.LabelEncoder()
from sklearn import preprocessing
for col in cat_columns:
    df_train_store[col]=le.fit_transform(df_train_store[col].astype('str'))


import math
plt.figure(figsize=(15,18))
noofrows= math.ceil(len(num_columns)/3)


#set false.Other wise error if  bandwidth =0 
sns.distributions._has_statsmodels=False

for i in range(len(num_columns)):
 plt.subplot(noofrows,3,i+1)
 out=sns.distplot(df_train_store[num_columns[i]]) 
    
plt.tight_layout()


df_train_store.Sales.mean()
df_train_store.loc[(df_train_store.Sales >= df_train_store.Sales.mean()),'aboveAvgSale']=1
df_train_store.loc[(df_train_store.Sales < df_train_store.Sales.mean()),'aboveAvgSale']=0
df_train_store.aboveAvgSale.value_counts()
#define y
y=df_train_store.aboveAvgSale.astype('int')
y1=np.log1p(df_train_store['Sales'])
df_train_store.drop(['Sales','aboveAvgSale'], axis = 1,  inplace = True)

num_columns=num_columns.drop(labels=['Sales'])
ct=ColumnTransformer([
    ('abc',RobustScaler(),num_columns),
    ('abc1',OneHotEncoder(),cat_columns),
    ],
    remainder="passthrough"
    )
ct.fit_transform(df_train_store)
X=df_train_store
#store features
colnames = X.columns.tolist()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30)
#credit : https://www.kaggle.com/tushartilwankar/sklearn-rf
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def RMSPE(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 15)
rf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score


y_pred = rf.predict(X_test)
error = (RMSPE(y_test,y_pred))
error
print("RMSPE of Random Forest %",error * 100)

#top 10 features of Random Forest
feat_importances_rf = pd.Series(rf.feature_importances_, index=colnames)
feat_importances_rf.nlargest(10).sort_values(ascending = True).plot(kind='barh')
plt.xlabel('importance')
plt.title('Feature Importance in Random Forest')
ss=preprocessing.StandardScaler
steps_xg = [('sts', ss() ),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2)        # Specify other parameters here
            )
            ]
# Instantiate Pipeline object
pipe_xg = Pipeline(steps_xg)
# What parameters in the pipe are available for tuning
pipe_xg.get_params()

parameters = {'xg__learning_rate':  [0.03, 0.05], # learning rate decides what percentage
                                                  #  of error is to be fitted by
                                                  #   by next boosted tree.
                                                  # See this answer in stackoverflow:
                                                  # https://stats.stackexchange.com/questions/354484/why-does-xgboost-have-a-learning-rate
                                                  # Coefficients of boosted trees decide,
                                                  #  in the overall model or scheme, how much importance
                                                  #   each boosted tree shall have. Values of these
                                                  #    Coefficients are calculated by modeling
                                                  #     algorithm and unlike learning rate are
                                                  #      not hyperparameters. These Coefficients
                                                  #       get adjusted by l1 and l2 parameters
              'xg__n_estimators':   [200,  300],  # Number of boosted trees to fit
                                                  # l1 and l2 specifications will change
                                                  # the values of coeff of boosted trees
                                                  # but not their numbers

              'xg__max_depth':      [4,6],
              'pca__n_components' : [10,15]
              }  


import time
clfgs = GridSearchCV(pipe_xg,            # pipeline object
                   parameters,         # possible parameters
                   n_jobs = 2,         # USe parallel cpu threads
                   cv =2 ,             # No of folds
                   verbose =2,         # Higher the value, more the verbosity
                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance
                   refit = 'roc_auc'   # Refitting final model on what parameters?
                                       # Those which maximise auc
                   )
#Start fitting data to pipeline
start = time.time()
clfgs.fit(X_train, y_train)
   
y_pred = clfgs.predict(X_test)
y_pred

# 7.5 Accuracy
accuracy = accuracy_score(y_test, y_pred)
f"Accuracy using GridSearchCV: {accuracy * 100.0}"             

# 7.6 Confusion matrix


from sklearn.metrics import confusion_matrix
cm_gs = pd.DataFrame(confusion_matrix(y_test, y_pred))
sns.heatmap(cm_gs, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Predicted vs Actual -GridSearchCV ')

# 7.7 F1 score
f1_score(y_test,y_pred, pos_label = 1)      
f1_score(y_test,y_pred, pos_label = 0)      

# 7.8 ROC curve
plot_roc_curve(clfgs, X_test, y_test)


# Get feature importances from GridSearchCV best fitted 'xg' model
#     See stackoverflow: https://stackoverflow.com/q/48377296
clfgs.best_estimator_.named_steps["xg"].feature_importances_
clfgs.best_estimator_.named_steps["xg"].feature_importances_.shape


#Hyperparameters to tune and their ranges
parameters = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,300),
              'xg__max_depth':      range(3,10),
              'pca__n_components' : range(10,17)}



# 8.1 Tune parameters using random search
#     Create the object first
rs = RandomizedSearchCV(pipe_xg,
                        param_distributions=parameters,
                        scoring= ['roc_auc', 'accuracy'],
                        n_iter=15,          
                                            
                        verbose = 3,
                        refit = 'roc_auc',
                        n_jobs = 2,          # Use parallel cpu threads
                        cv = 2               
                                             
                        )


# 
rs.fit(X_train, y_train)

y_pred = rs.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
f"Accuracy using randomized search: {accuracy * 100.0}"        
f1_score(y_test,y_pred, pos_label = 1) 
model_gs = XGBClassifier(
                    learning_rate = clfgs.best_params_['xg__learning_rate'],
                    max_depth = clfgs.best_params_['xg__max_depth'],
                    n_estimators=clfgs.best_params_['xg__max_depth']
                    )

# 9.1 Model with parameters of random search
model_rs = XGBClassifier(
                    learning_rate = rs.best_params_['xg__learning_rate'],
                    max_depth = rs.best_params_['xg__max_depth'],
                    n_estimators=rs.best_params_['xg__max_depth']
                    )


# Modeling with both parameters

model_gs.fit(X_train, y_train)
model_rs.fit(X_train, y_train)

#Predictions with both models
y_pred_gs = model_gs.predict(X_test)
y_pred_rs = model_rs.predict(X_test)

#Accuracy from both models
accuracy_gs = accuracy_score(y_test, y_pred_gs)
accuracy_rs = accuracy_score(y_test, y_pred_rs)
print("Accuracy with GridSearch XGB model:",accuracy_gs*100)
print("Accuracy with Random search XGB model:",accuracy_rs*100)

rmspe_gs = RMSPE(y_pred_gs,y_test)
rmspe_rs = RMSPE(y_pred_rs,y_test)
print("RMSPE of GridSearch XGB modelt %",rmspe_gs * 100)
print("RMSPE of Random search XGB modelt %",rmspe_rs * 100)


#  Plt now

%matplotlib inline
model_gs.feature_importances_
model_rs.feature_importances_
# Importance type: 'weight'
plot_importance(
                model_gs,
                importance_type = 'weight'   # default
                )
#  Importance type: 'gain'
#        # Normally use this
plot_importance(
                model_rs,
                importance_type = 'gain', 
                title = "Feature impt by gain"
                )
plt.show()
