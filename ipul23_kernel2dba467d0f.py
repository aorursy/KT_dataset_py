#import Library
%matplotlib inline
import math
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.combine import SMOTEENN ,SMOTETomek
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
# disable chained assignments
from eli5.sklearn import PermutationImportance
from catboost import CatBoostClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from sklearn.linear_model import SGDClassifier
import pandas as pd 
from sklearn.feature_selection import RFECV
pd.options.mode.chained_assignment = None 
from scipy import stats
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from imblearn.combine import SMOTEENN 
# Ensemble
from mlens.ensemble import SuperLearner
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
from sklearn.metrics import accuracy_score
import copy
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.svm import SVC,SVR
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsClassifier
import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from eli5.sklearn import PermutationImportance
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
df_train = pd.read_csv("../input/zomato.csv",encoding='latin-1')
kf = KFold(n_splits=10,random_state=42)
df_train.shape
df_train.head().T
sns.boxplot(x='Rating text',y='Votes',data=df_train)
df_train[df_train['Currency'] == 'Indonesian Rupiah(IDR)'].head()
#City,Has,Has,Is,Average Cost for two,Convert to USD
df_select = df_train[['City','Locality','Average Cost for two','Has Table booking','Has Online delivery','Is delivering now','Switch to order menu']] 
df_select.head()
df_train['Currency'].unique()
idx = df_train[df_train['Currency'] == 'Botswana Pula(P)']['Average Cost for two'].index 
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Botswana Pula(P)']['Average Cost for two'].divide(10.62)
idx = df_train[df_train['Currency'] == 'Brazilian Real(R$)']['Average Cost for two'].index 
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Brazilian Real(R$)']['Average Cost for two'].divide(3.91)
idx = df_train[df_train['Currency'] == 'Emirati Diram(AED)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Emirati Diram(AED)']['Average Cost for two'].divide(3.67)
idx =df_train[df_train['Currency'] == 'Indian Rupees(Rs.)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Indian Rupees(Rs.)']['Average Cost for two'].divide(71.38)
idx = df_train[df_train['Currency'] == 'Indonesian Rupiah(IDR)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Indonesian Rupiah(IDR)']['Average Cost for two'].divide(14572.75)
idx = df_train[df_train['Currency'] == 'NewZealand($)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'NewZealand($)']['Average Cost for two'].divide(1.45)
idx = df_train[df_train['Currency'] == 'Pounds(\x8c£)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Pounds(\x8c£)']['Average Cost for two'].divide(0.79)
idx = df_train[df_train['Currency'] == 'Qatari Rial(QR)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Qatari Rial(QR)']['Average Cost for two'].divide(3.64)
idx = df_train[df_train['Currency'] == 'Rand(R)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Rand(R)']['Average Cost for two'].divide(14.20)
idx = df_train[df_train['Currency'] == 'Sri Lankan Rupee(LKR)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Sri Lankan Rupee(LKR)']['Average Cost for two'].divide(178.94)
idx = df_train[df_train['Currency'] == 'Turkish Lira(TL)']['Average Cost for two'].index
df_train['Average Cost for two'].iloc[idx] = df_train[df_train['Currency'] == 'Turkish Lira(TL)']['Average Cost for two'].divide(5.28) 

df_train.head()
len(df_train['City'].unique())
#City,Has,Has,Is,Average Cost for two,Convert to USD
df_select = df_train[['Price range','Average Cost for two','Has Table booking','Has Online delivery','Is delivering now','Switch to order menu']].copy() 
df_select.head()
#sns.countplot(df_select['Rating text'])
sns.countplot(df_select['Has Table booking'])
plt.show()
sns.countplot(df_select['Has Online delivery'])
plt.show()
sns.countplot(df_select['Is delivering now'])
plt.show()
sns.countplot(df_select['Switch to order menu'])
plt.show()
df_select.drop(['Is delivering now','Switch to order menu'],axis=1,inplace=True)
df_select.head()
df_select['Has Table booking'].replace({'Yes':1,'No':0},inplace=True)
df_select['Has Online delivery'].replace({'Yes':1,'No':0},inplace=True)
df_transform = df_select
df_transform.head()
df_train['Rating text'].unique()
df_train['Rating text'].replace({'Excellent':1,'Very Good':1,'Good':1,'Average':0,'Not rated':0,'Poor':0},inplace=True)
y = df_train['Rating text'].copy()
model = pd.DataFrame()
from sklearn import tree
clf1  = tree.DecisionTreeClassifier()
res = cross_val_score(clf1,df_transform,y,cv=kf,scoring='accuracy')
model['Decision Tree'] = res
#clf.fit(df_transform,y)
#import graphviz
#dot_data = tree.export_graphviz(clf, out_file=None, feature_names=df_transform.columns,  class_names='Rating text',  filled=True, rounded=True,   special_characters=True) 
#graph = graphviz.Source(dot_data)  
#graph 
from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()
sc = cross_val_score(clf2,df_transform,y,cv=kf,scoring='accuracy')
print(sc)
model['Naive Bayes'] = sc
from sklearn.neighbors import KNeighborsClassifier
clf3 = KNeighborsClassifier()
sc = cross_val_score(clf3,df_transform,y,cv=kf,scoring='accuracy')
print(sc)
model['kNN'] = sc
model.plot(title = "Nilai AUC setiap model dengan 10 fold cross-validation")
print("Decision tree mean AUC : ",model['Decision Tree'].mean())
print("Naive Bayes mean AUC : ",model['Naive Bayes'].mean())
print("kNN mean AUC : ",model['kNN'].mean())
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
eclf1 = VotingClassifier(estimators=[('dt', clf1), ('svm', clf2), ('knn', clf3)], voting='hard')
res = cross_val_score(eclf1,df_transform,y,cv=kf)
print(res)
print(y.sum())
print(len(y))
from mpl_toolkits.basemap import Basemap
%matplotlib inline

# Set up plot
plt.figure(1, figsize=(20,10))

# Mercator of World
m1 = Basemap(projection='merc',
             llcrnrlat=-60,
             urcrnrlat=65,
             llcrnrlon=-180,
             urcrnrlon=180,
             lat_ts=0,
             resolution='c')
m2 = Basemap(projection='merc',
             llcrnrlat=-60,
             urcrnrlat=65,
             llcrnrlon=-180,
             urcrnrlon=180,
             lat_ts=0,
             resolution='c')
m1.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m1.drawmapboundary(fill_color='#000000')                # black background
m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
m2.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m2.drawmapboundary(fill_color='#000000')                # black background
m2.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
good_longitude = df_train[df_train['Rating text'] == 1]['Longitude']
good_latitude = df_train[df_train['Rating text'] == 1]['Latitude']
bad_longitude = df_train[df_train['Rating text'] == 0]['Longitude']
bad_latitude = df_train[df_train['Rating text'] == 0]['Latitude']
# Plot the data
mxy = m1(good_longitude.tolist(), good_latitude.tolist())
m1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=1, zorder=5)
# Plot the data
mxy1 = m2(bad_longitude.tolist(), bad_latitude.tolist())
m2.scatter(mxy1[0], mxy1[1], s=3, c="red", lw=0, alpha=1, zorder=5)

plt.title("Good and bad rating around the world")
plt.show()
sns.distplot(df_train['Average Cost for two'])
from mpl_toolkits.basemap import Basemap
%matplotlib inline

# Set up plot
plt.figure(1, figsize=(20,10))

# Mercator of World
m1 = Basemap(projection='merc',
             llcrnrlat=-60,
             urcrnrlat=65,
             llcrnrlon=-180,
             urcrnrlon=180,
             lat_ts=0,
             resolution='c')
m2 = Basemap(projection='merc',
             llcrnrlat=-60,
             urcrnrlat=65,
             llcrnrlon=-180,
             urcrnrlon=180,
             lat_ts=0,
             resolution='c')
m1.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m1.drawmapboundary(fill_color='#000000')                # black background
m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
m2.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m2.drawmapboundary(fill_color='#000000')                # black background
m2.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
good_longitude = df_train[df_train['Price range'] > 2]['Longitude']
good_latitude = df_train[df_train['Price range'] > 2]['Latitude']
bad_longitude = df_train[df_train['Price range'] <= 2]['Longitude']
bad_latitude = df_train[df_train['Price range'] <= 2]['Latitude']
# Plot the data
mxy = m1(good_longitude.tolist(), good_latitude.tolist())
m1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=1, zorder=5)
# Plot the data
mxy1 = m2(bad_longitude.tolist(), bad_latitude.tolist())
m2.scatter(mxy1[0], mxy1[1], s=3, c="red", lw=0, alpha=1, zorder=5)

plt.title("Price range around the world")
plt.show()
df_train.head()
len(df_train)
from sklearn.metrics import roc_auc_score
def cross_validation_split(X,y,kf):
    res = []
    for train_index,test_index in kf.split(X,y):
        X_train,X_test = X[train_index], X[test_index]
        y_train,y_test = y[train_index], y[test_index]
        DT = tree.DecisionTreeClassifier()
        NB = GaussianNB()
        kNN = KNeighborsClassifier()
        DT.fit(X_train,y_train)
        NB.fit(X_train,y_train)
        kNN.fit(X_train,y_train)
        DT_res = DT.predict_proba(X_test)
        NB_res = NB.predict_proba(X_test)
        kNN_res = kNN.predict_proba(X_test)
        print(DT_res[:,1])
        #predict = (DT_res.T[:,1] + NB_res[:,1] + kNN_res[:,1])/3
        #res.append(roc_auc_score(y_test,predict))
    return res
res = cross_validation_split(df_transform.values,y,kf)
model['kNN+NaiveBayes+DecisionTree'] = res 
model.plot(title = "Nilai AUC setiap model dengan 10 fold cross-validation")
print("Decision tree mean AUC : ",model['Decision Tree'].mean())
print("Naive Bayes mean AUC : ",model['Naive Bayes'].mean())
print("kNN mean AUC : ",model['kNN'].mean())
print("kNN+NaiveBayes+DecisionTree mean AUC : ",model['kNN+NaiveBayes+DecisionTree'].mean())
from sklearn.model_selection import GridSearchCV
tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
clf = GridSearchCV(tree.DecisionTreeClassifier(), tree_para, cv=kf)
clf.fit(df_transform, y)
print(clf.best_params_)
model = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 4)
model.fit(df_transform, y)
import graphviz
dot_data = tree.export_graphviz(model, out_file=None, feature_names=df_transform.columns, class_names="FT",  filled=True, rounded=True,  special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
