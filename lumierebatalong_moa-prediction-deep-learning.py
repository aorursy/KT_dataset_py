# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from scipy.stats import normaltest, norm

import scipy as sp

import holoviews as hv

from holoviews import opts

hv.extension('bokeh')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict, cross_validate

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, LabelEncoder, RobustScaler, Normalizer

from sklearn.pipeline import Pipeline

from xgboost import XGBRFClassifier

from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import multilabel_confusion_matrix, label_ranking_loss, log_loss, roc_auc_score

from sklearn.linear_model import LogisticRegression, Perceptron

from sklearn.neural_network import MLPClassifier

from sklearn.multioutput import MultiOutputClassifier

from sklearn.decomposition import PCA 

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
file1 = '/kaggle/input/lish-moa/train_features.csv'

file2 = '/kaggle/input/lish-moa/train_targets_scored.csv'
data = pd.read_csv(file1)

target = pd.read_csv(file2)
data.tail()
target.head()
data.info()
data.isnull().sum()[data.isnull().sum()>0]
target.info()
print('We have {} different drugs id.'.format(data.sig_id.nunique()))
data.cp_type.unique() #two types compound trt_cp give MoA but ctl_vehicle have not MoA
data.cp_dose.unique()# D1 is low dose and D2 is high dose.
data.describe()
def outlier_feature():

    """

        This function seek the feature that contain outlier.

    """

    

    #take all feature

    cols = [x for x in data.columns if data[x].dtype != object]

    outlier = [] #list of feature that contains outlier

    

    for u in cols:

        

        desc = data[u].describe()# 

        

        Q1 = desc['25%']# 25 percentile

        Q3 = desc['75%']# 75 percentile

        IQR = Q3 - Q1# Interval percentile

        

        low = Q1 - 1.5*IQR #low limit

        upper = Q3 + 1.5*IQR #upper limit

        

        for x in data[u].values:

            if ( (x > upper) or (x < low)): # all x respects this condition is an outlier.

                

                outlier.append(u)# take all feature

                

    return np.unique(outlier)
out_feature = outlier_feature()
print('Number of feature containing outlier is {}'.format(len(out_feature)))
#we plot some.

fig = plt.figure(figsize=(20,20))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(1,21):

    ax = fig.add_subplot(5, 4, i)

    

    sns.distplot(data[out_feature[i-1]], ax=ax)

    ax.set_title(out_feature[i-1]) 
#we plot some.

fg = plt.figure(figsize=(20,20))

fg.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(1,21):

    ax = fg.add_subplot(5, 4, i)

    

    sns.boxplot(data.iloc[:, i+10], ax=ax) 
plt.figure(figsize=(15,5))

ax1 = plt.subplot(121)

ax2 = plt.subplot(122)

sns.countplot(x='cp_type', data=data, ax=ax1)

sns.countplot(x='cp_dose', data=data, ax=ax2)

plt.show()
targ = target.iloc[:, 1:].sum().sort_values(ascending=True)
plt.figure(dpi=80)

targ[targ>100].plot(kind='barh', figsize=(10,20))

plt.title('Multi MoA label', weight='bold')

plt.grid(True)
df_cor = data.corr()
df_cor
vifs = pd.DataFrame(np.linalg.inv(df_cor.values).diagonal(), index=df_cor.index, columns=['VIF'])
vifs.tail()
#we take a value where vif is greater than 15.

greater_vifs = vifs.where(vifs>15)

greater_vifs = greater_vifs.dropna()
cols_remove = greater_vifs.index
cols_remove #This is a feature that have highly correlated with any number of the other variables.
new_data = data.drop(columns=cols_remove) # we drop these columns highly correlated
new_data.head()
new_data['cp_type'] = new_data['cp_type'].astype('category')

new_data['cp_type'].cat.categories = [0, 1]

new_data['cp_type'] = new_data['cp_type'].astype("int")
new_data['cp_dose'] = new_data['cp_dose'].astype('category')

new_data['cp_dose'].cat.categories = [0, 1]

new_data['cp_dose'] = new_data['cp_dose'].astype("int")
train = new_data.drop(columns=['sig_id'])
train.tail()
xtrain, xvalid, ytrain, yvalid = train_test_split(train, target.drop(columns='sig_id') , test_size=0.2, random_state=42)
#plot pca components and cumsum explained variance ratio

pipeline =  Pipeline([('scaler', MinMaxScaler()), ('pca', PCA())]).fit(xtrain)
pca = pipeline['pca']

plt.figure(figsize=(15, 8))

plt.plot(range(len(pca.explained_variance_)), np.cumsum(pca.explained_variance_ratio_))

plt.title('Principal Components')

plt.xlabel('n_components')

plt.ylabel('cumsum explained variance ratio')

plt.grid('on')
#PCA and standardize

pipe = Pipeline([('scaler', MinMaxScaler()), ('pca', PCA(n_components=0.95))])
pca_xtrain = pipe.fit_transform(xtrain)
pca_xtrain.shape
pca_xvalid = pipe.transform(xvalid)
pca_xvalid.shape
mlp = MLPClassifier(hidden_layer_sizes=(100, ytrain.shape[1]), verbose=0, random_state=42, max_iter=500,

                    activation='tanh', learning_rate='invscaling', learning_rate_init=0.0001, beta_1=0.5)
scaler = MinMaxScaler()
xtrain_scale = scaler.fit_transform(xtrain)

xvalid_scale = scaler.transform(xvalid)
mlp.fit(xtrain_scale, ytrain.values)
plt.figure(figsize=(15,8))

plt.plot(range(mlp.n_iter_), mlp.loss_curve_)

plt.xlabel('Iteration')

plt.ylabel('log_loss')

plt.grid('on')
print('best_loss for MLP: {}'.format(mlp.best_loss_))
ypred = mlp.predict_proba(xvalid_scale)
log_loss(np.ravel(yvalid), np.ravel(ypred))
file3 = '/kaggle/input/lish-moa/test_features.csv'
test = pd.read_csv(file3)
test.head()
test = test.drop(columns=cols_remove) # remove cols
test['cp_type'] = test['cp_type'].astype('category')

test['cp_type'].cat.categories = [0, 1]

test['cp_type'] = test['cp_type'].astype("int")
test['cp_dose'] = test['cp_dose'].astype('category')

test['cp_dose'].cat.categories = [0, 1]

test['cp_dose'] = test['cp_dose'].astype("int")
test.tail()
xtest = scaler.transform(test.drop(columns='sig_id'))
pca_xtest = pipe.transform(test.drop(columns='sig_id'))
ypredict = mlp.predict_proba(xtest)
ypredict
col_targ = target.columns
file4 = '/kaggle/input/lish-moa/sample_submission.csv'
sample = pd.read_csv(file4)
sample.iloc[:,1:] = ypredict
sample.tail()
sample.to_csv('submission.csv', index=False)