from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import preprocessing
from scipy.stats import norm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
from scipy import stats
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os
warnings.filterwarnings('ignore')
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_red=pd.read_csv("/kaggle/input/wine-quality-selection/winequality-red.csv")
df_white=pd.read_csv("/kaggle/input/wine-quality-selection/winequality-white.csv")
df_white.info()
df_red.info()
df_red.describe()
df_white.describe()
# Combining the red and white wine data
df_wine = df_red.append(df_white)
print(df_wine.shape)
df_wine.describe()
# Features for the wine data
sns.set()
pd.DataFrame.hist(df_wine, figsize = [15,15], color='green')
plt.show()
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(df_wine.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

dependent_all=df_wine['quality']
independent_all=df_wine.drop(['quality'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(independent_all,dependent_all,test_size=0.3,random_state=100)
xgboost = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05)
xgboost.fit(x_train,y_train)
#XGBoost modelon the train set
XGB_prediction = xgboost.predict(x_train)
XGB_score= accuracy_score(y_train,XGB_prediction)
XGB_score
#XGBoost model on the test
XGB_prediction = xgboost.predict(x_test)
XGB_score= accuracy_score(y_test,XGB_prediction)
XGB_score
rfc2=RandomForestClassifier()
rfc2.fit(x_train,y_train)
#model on train using all the independent values in df
rfc_prediction = rfc2.predict(x_train)
rfc_score= accuracy_score(y_train,rfc_prediction)
print(rfc_score)
#model on test using all the indpendent values in df
rfc_prediction = rfc2.predict(x_test)
rfc_score= accuracy_score(y_test,rfc_prediction)
print(rfc_score)
log =LogisticRegression()
log.fit(x_train,y_train)
#model on train using all the independent values in df
log_prediction = log.predict(x_train)
log_score= accuracy_score(y_train,log_prediction)
print(log_score)
#model on train using all the independent values in df
log_prediction = log.predict(x_test)
log_score= accuracy_score(y_test,log_prediction)
print(log_score)
dec=DecisionTreeClassifier()
dec.fit(x_train,y_train)
#model on train using all the independent values in df
dec_prediction = dec.predict(x_train)
dec_score= accuracy_score(y_train,dec_prediction)
print(dec_score)
#model on test using all the independent values in df
dec_prediction = dec.predict(x_test)
dec_score= accuracy_score(y_test,dec_prediction)
print(dec_score)
etc=ExtraTreeClassifier()
etc.fit(x_train,y_train)
#model on train using all the independent values in df
etc_prediction = etc.predict(x_train)
etc_score= accuracy_score(y_train,etc_prediction)
print(etc_score)
#model on test using all the independent values in df
etc_prediction = etc.predict(x_test)
etc_score= accuracy_score(y_test,etc_prediction)
print(etc_score)
ada =AdaBoostClassifier()
ada.fit(x_train,y_train)
#model on train using all the independent values in df
ada_prediction = ada.predict(x_train)
ada_score= accuracy_score(y_train,ada_prediction)
print(ada_score)
#model on test using all the independent values in df
ada_prediction = ada.predict(x_test)
ada_score= accuracy_score(y_test,ada_prediction)
print(ada_score)
bca =BaggingClassifier()
bca.fit(x_train,y_train)
#model on train using all the independent values in df
bca_prediction = bca.predict(x_train)
bca_score= accuracy_score(y_train,bca_prediction)
print(bca_score)
#model on test using all the independent values in df
bca_prediction = bca.predict(x_test)
bca_score= accuracy_score(y_test,bca_prediction)
print(bca_score)
estimator = [] 
estimator.append(('LR',  
                  LogisticRegression(solver ='lbfgs',  
                                     multi_class ='multinomial',  
                                     max_iter = 200))) 
estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 
estimator.append(('DTC', DecisionTreeClassifier()))
vc=VotingClassifier(estimators = estimator, voting ='hard') 
vc.fit(x_train,y_train)
#model on train using all the independent values in df
vc_prediction = vc.predict(x_train)
vc_score= accuracy_score(y_train,vc_prediction)
print(vc_score)
#model on test using all the independent values in df
vc_prediction = vc.predict(x_test)
vc_score= accuracy_score(y_test,vc_prediction)
print(vc_score)
vc=VotingClassifier(estimators = estimator, voting ='soft') 
vc.fit(x_train,y_train)
#model on train using all the independent values in df
vc_prediction = vc.predict(x_train)
vc_score= accuracy_score(y_train,vc_prediction)
print(vc_score)
#model on test using all the independent values in df
vc_prediction = vc.predict(x_test)
vc_score= accuracy_score(y_test,vc_prediction)
print(vc_score)
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
#model on train using all the independent values in df
gbc_prediction = gbc.predict(x_train)
gbc_score= accuracy_score(y_train,gbc_prediction)
print(gbc_score)
#model on test using all the independent values in df
gbc_prediction =gbc.predict(x_test)
gbc_score= accuracy_score(y_test,gbc_prediction)
print(gbc_score)
ettc=ExtraTreesClassifier()
ettc.fit(x_train,y_train)
#model on train using all the independent values in df
ettc_prediction = ettc.predict(x_train)
ettc_score= accuracy_score(y_train,ettc_prediction)
print(ettc_score)
#model on test using all the independent values in df
ettc_prediction =ettc.predict(x_test)
ettc_score= accuracy_score(y_test,ettc_prediction)
print(ettc_score)
sgdc=SGDClassifier()
sgdc.fit(x_train,y_train)
#model on train using all the independent values in df
sgdc_prediction = sgdc.predict(x_train)
sgdc_score= accuracy_score(y_train,sgdc_prediction)
print(sgdc_score)
#model on test using all the independent values in df
sgdc_prediction =sgdc.predict(x_test)
sgdc_score= accuracy_score(y_test,sgdc_prediction)
print(sgdc_score)
pac=PassiveAggressiveClassifier()
pac.fit(x_train,y_train)
#model on train using all the independent values in df
pac_prediction = pac.predict(x_train)
pac_score= accuracy_score(y_train,pac_prediction)
print(pac_score)
#model on test using all the independent values in df
pac_prediction =pac.predict(x_test)
pac_score= accuracy_score(y_test,pac_prediction)
print(pac_score)
rc=RidgeClassifier()
rc.fit(x_train,y_train)
#model on train using all the independent values in df
rc_prediction = rc.predict(x_train)
rc_score= accuracy_score(y_train,rc_prediction)
print(rc_score)
#model on test using all the independent values in df
rc_prediction =rc.predict(x_test)
rc_score= accuracy_score(y_test,rc_prediction)
print(rc_score)
clf = RandomForestClassifier()
grid_values = {'max_features':['auto','sqrt','log2'],'max_depth':[None, 10, 5, 3, 1],
              'min_samples_leaf':[1, 5, 10, 20, 50]}
clf
grid_clf = GridSearchCV(clf, param_grid=grid_values, cv=10, scoring='accuracy')
grid_clf.fit(x_train, y_train) # fit and tune model
grid_clf.best_params_
clf = RandomForestClassifier().fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Training Accuracy :: ', accuracy_score(y_train, clf.predict(x_train)))
print('Test Accuracy :: ', accuracy_score(y_test, y_pred))
df_wine
df_wine_unsupervised=df_wine.drop(['quality'],axis=1)
df_wine_unsupervised.dtypes
from scipy import stats
df_wine_z_score=stats.zscore(df_wine_unsupervised, axis = 1)
cols = list(df_wine_unsupervised.columns)
df_wine_unsupervised[cols]
# now iterate over the remaining columns and create a new zscore column
for col in cols:
    col_zscore = col + '_zscore'
    df_wine_unsupervised[col_zscore] = (df_wine_unsupervised[col] - df_wine_unsupervised[col].mean())/df_wine_unsupervised[col].std(ddof=0)
df_wine_unsupervised.head()
df_wine_unsupervised.columns
x_norm=df_wine_unsupervised[['fixed acidity_zscore',
       'volatile acidity_zscore', 'citric acid_zscore',
       'residual sugar_zscore', 'chlorides_zscore',
       'free sulfur dioxide_zscore', 'total sulfur dioxide_zscore',
       'density_zscore', 'pH_zscore', 'sulphates_zscore', 'alcohol_zscore']]
from sklearn.cluster import DBSCAN
model=DBSCAN()
model.fit(x_norm)
labels=model.labels_
from sklearn.cluster import KMeans
model = KMeans(n_clusters=8)
model.fit(x_norm)
labels=model.labels_
df_wine_unsupervised['predicted_quality']=labels
df_wine_unsupervised
df_wine_unsupervised['quality']=df_wine['quality']
import pandas as pd
%matplotlib inline
#do code to support model
#"data" is the X dataframe and model is the SKlearn object

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_train.columns, rfc2.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)


