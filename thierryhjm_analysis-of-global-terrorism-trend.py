import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# from mpl_toolkits.basemap import Basemap
import seaborn as sns

import matplotlib.animation as animation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')


import io
import requests
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rc('figure', figsize=(12,10))
%matplotlib inline

#specific to time series analysis
import scipy.stats as st
from statsmodels.tsa import stattools as stt
from statsmodels import tsa
import statsmodels.api as smapi
import datetime

from statsmodels.tsa.arima_model import ARIMA

import warnings
warnings.simplefilter(action='ignore')
try:
    terrorism = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
    print('File load: Success')
except:
    print('File load: Failed')
# terrorism = pd.read_csv('global terrorism.csv', encoding='ISO-8859-1')
terrorism.head(10)
terrorism.info()
terrorism.describe() ##describes only numeric data
rename = terrorism.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terrorism = terrorism[['Year','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terrorism['casualities']=terrorism['Killed']+terrorism['Wounded']
terrorism.head(10)
terrorism.dropna(how = 'all', inplace=True)
print('Size After Dropping Rows with NaN in All Columns:', terrorism.shape)
terrorism.isnull().sum()
print('Country with Highest Terrorist Attacks:',terrorism['Country'].value_counts().index[0])
print('Regions with Highest Terrorist Attacks:',terrorism['Region'].value_counts().index[0])
print('Maximum people killed in an attack are:',terrorism['Killed'].max(),'that took place in',terrorism.loc[terrorism['Killed'].idxmax()].Country)
plt.subplots(figsize=(13,6))
sns.countplot('Year',data=terrorism,palette='RdYlGn_r',edgecolor=sns.color_palette('husl',8))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
people_killed_eachyr = terrorism[["Year","casualities"]].groupby('Year').sum()

plt.subplots(figsize = (15,6))
sns.barplot(x=people_killed_eachyr.index, 
            y=[i[0] for i in people_killed_eachyr.values], data = people_killed_eachyr, palette='RdYlGn_r',edgecolor=sns.color_palette("Set2", 10))
plt.xticks(rotation=90)
plt.title('Number Of people were killed of wouded by terrorism each year')
plt.show()
terrorism.to_csv('terr.csv', index = False)
dateparse = lambda d: pd.datetime.strptime(d, '%Y')

f='terr.csv'
terrorism_ts = pd.read_csv(f,
                   parse_dates=['Year'], 
                   index_col='Year', 
                   date_parser=dateparse,
                   )
terrorism_ts.head()
terrorism_ts = terrorism_ts.iloc[:, 0]
terrorism_ts.head()
type(terrorism_ts)

plt.subplots(figsize=(13,6))
sns.countplot('Region',data=terrorism, palette='RdYlGn_r',edgecolor=sns.color_palette('husl',8))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Region')
sns.set(font_scale=1)
plt.show()
terror_region=pd.crosstab(terrorism.Year,terrorism.Region)
terror_region.plot(color=sns.color_palette('Set2',12))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
plt.subplots(figsize=(15,6))
sns.barplot(terrorism['Country'].value_counts()[:15].index,terrorism['Country'].value_counts()[:15].values,palette='inferno')
plt.title('Top Affected Countries')
sns.set(font_scale=1)
plt.show()
pd.crosstab(terrorism.Region,terrorism.AttackType).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',9))
fig=plt.gcf()
fig.set_size_inches(12,8)
sns.set(font_scale=0.5)
plt.show()


plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=terrorism,palette='inferno',order=terrorism['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
sns.set(font_scale=2)
plt.show()
# pd.crosstab(terrorism.AttackType,terrorism.Group).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',9))
# fig=plt.gcf()
# fig.set_size_inches(12,8)
# sns.set(font_scale=0.5)
# plt.show()
# plt.subplots(figsize=(13,6))
# sns.countplot('Target',data=terrorism,palette='RdYlGn_r',edgecolor=sns.color_palette('husl',8))
# plt.xticks(rotation=90)
# plt.title('Number Of Terrorist Activities Each Target')
# plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import itertools


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
####7.1 Preprocessing

# Before we do the classification, we may want to delete some columns like city and motive since some of them may have
# linear relationship with other columns like killed and some of them has too many missing value like motive

terrorism_cm = terrorism.drop(columns = ['Motive','Target','Killed','Wounded','Summary','city'])
# Encode category predictors into numbers, facilitating later work
labelEncoding = LabelEncoder()
terrorism_cm['Country'] = labelEncoding.fit_transform(terrorism_cm['Country'])
terrorism_cm['AttackType'] = labelEncoding.fit_transform(terrorism_cm['AttackType'])
terrorism_cm['Target_type'] = labelEncoding.fit_transform(terrorism_cm['Target_type'])
terrorism_cm['Weapon_type'] = labelEncoding.fit_transform(terrorism_cm['Weapon_type'])
terrorism_cm['Region'] = labelEncoding.fit_transform(terrorism_cm['Region'])
terrorism_cm['Group'] = labelEncoding.fit_transform(terrorism_cm['Group'])


terrorism_cm['casualities'] = terrorism_cm['casualities'].apply(lambda x: 0 if x == 0 else 1)
terrorism_cm.head(5)
len(terrorism_cm)
# We drop na to avoid misinformation
terrorism_cm = terrorism_cm.dropna()
len(terrorism_cm)
len(terrorism_cm[terrorism_cm['casualities'] == 0])
####7.2 Cross Validation

# Split data for training data and validation data
X = terrorism_cm[['Year','Country','Region','latitude','longitude','AttackType','Group','Target_type','Weapon_type']]
valid = terrorism_cm['casualities']

X_train, X_test, valid_train, valid_test = train_test_split(X, valid, test_size=0.3)

####7.3 Compute the feature importances with random forest
forest = ExtraTreesClassifier(n_estimators=20,
                              random_state=0)

forest.fit(X, valid)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
fnames = [['Year','Country','Region','latitude','longitude','AttackType','Group','Target_type','Weapon_type'][i] for i in indices]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), fnames, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
####7.4 Train the model
X = terrorism_cm[['Year','Country','latitude','longitude','AttackType','Group','Target_type','Weapon_type']]
valid = terrorism_cm['casualities']

X_train, X_test, valid_train, valid_test = train_test_split(X, valid, test_size=0.3)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, valid_train)
pred = model.predict(X_test)
np.mean(pred == valid_test)
####7.5 Confusion Matrix
# reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
cnf_matrix = confusion_matrix(valid_test, pred)
 

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
# Compute confusion matrix
np.set_printoptions(precision=2)
 
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      title='Confusion matrix, without normalization')
 
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True,
                      title='Normalized confusion matrix')
 
plt.show()
####7.6 ROC curve
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import roc_curve
score = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(pred, score, pos_label=1)
auc = np.trapz(tpr, fpr)

plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.legend()
plt.show() 