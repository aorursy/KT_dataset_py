# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import np_utils

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split,GridSearchCV      # import GridSearchCV

from sklearn.pipeline import make_pipeline        # import pipeline

from sklearn.preprocessing import StandardScaler 

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/winequality-red.csv")

data.shape
#shuffling the dataset

np.random.seed(123)

data = data.reindex(np.random.permutation(data.index))
#before dividing the dataset, we will run some preliminary analysis

data.head()
data_X = data.iloc[:,:-1]

data_y = data.iloc[:,-1]



data_y.value_counts(normalize=True)   #checking proportion of different ratings

data_y  = data_y.astype('category')

data_y1 = np_utils.to_categorical(data_y)

data_y1
data_X.isnull().sum()

# No missing values found
data_X.describe()
#correlation matrix

corrmat = data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
data_X['non_free_sulfur_dioxide'] = data_X['total sulfur dioxide'] - data_X['free sulfur dioxide']

data_X = data_X.drop(['total sulfur dioxide'],axis=1)

data_X['fixed_acidity_proportion_citric'] = data_X['citric acid']/data_X['fixed acidity']
data_X['non_free_sulfur_dioxide'].describe()
m=1

plt.figure(figsize = (20,20))

for i in data_X.columns:

    plt.subplot(3,4,m)

    sns.boxplot(data_X[i])

    m = m+1
for col in data_X.columns:

    percentiles = data_X[col].quantile([0.01,0.99]).values

    data_X[col][data_X[col] <= percentiles[0]] = percentiles[0]

    data_X[col][data_X[col] >= percentiles[1]] = percentiles[1]
# plot histograms to see skewness

m=1

plt.figure(figsize = (15,15))

for i in data_X.columns:

    plt.subplot(3,4,m)

    sns.distplot(data_X[i],kde = True)

    m = m+1

data_X.dtypes[data_X.dtypes != "object"].index
from scipy.stats import skew



#finding skewness of all variables

col = data_X.columns

skewed_feats = data_X[col].apply(lambda x: skew(x.dropna()))

#adjusting features having skewness >0.75

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

data_X[skewed_feats] = np.log1p(data_X[skewed_feats])
data_X.head()
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

data_y= pd.cut(data_y, bins = bins, labels = group_names)
from sklearn.preprocessing import LabelEncoder

label_quality = LabelEncoder()

#Bad becomes 0 and good becomes 1 

data_y = label_quality.fit_transform(data_y)
'''

from imblearn.combine import SMOTETomek



smt = SMOTETomek(sampling_strategy = 0.3)

X_smt, y_smt = smt.fit_sample(data_X, data_y)



print('Before applying oversampling: ', data_y.shape)

unique, counts = np.unique(data_y, return_counts=True)   #y_smt is now ndarray, we can't apply value_counts()

print(np.asarray((unique, counts)).T)



print('After applying oversampling: ', y_smt.shape)

unique, counts = np.unique(y_smt, return_counts=True)   #y_smt is now ndarray, we can't apply value_counts()

print(np.asarray((unique, counts)).T)

'''
#we will divide data into train, test and validation data

#since our data is shuffled we can select say top n rows for training

ntrain = int(data.shape[0]*0.9)

train_data = data_X.iloc[:ntrain,:]            # 90% of total data

train_data_y = data_y[:ntrain]



validation_data = data_X.iloc[ntrain: ,:]             #10% of total data

val_data_y = data_y[ntrain:]



print(train_data.shape[0])

print(validation_data.shape[0])
from sklearn.model_selection import train_test_split



#  split X between training and testing set

x_train, x_test, y_train, y_test = train_test_split(train_data,train_data_y, test_size=0.25, shuffle=True)
ss = StandardScaler()

ss.fit(x_train)

x_train = ss.transform(x_train)

x_test = ss.transform(x_test)

validation_data = ss.transform(validation_data)


def report(y_actuals,y_pred):

    print(classification_report(y_actuals, y_pred))
# Using DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

dtc_pred = dtc.predict(x_test)

report(y_test, dtc_pred)
# Using RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=400)

rfc.fit(x_train, y_train)

rfc_pred = rfc.predict(x_test)

report(y_test, rfc_pred)
features = data_X.columns

importances = rfc.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
# Using XGBClassifier

xgb = XGBClassifier(n_estimators=900)

xgb.fit(x_train, y_train)

xgb_pred = xgb.predict(x_test)

report(y_test, xgb_pred)
rfc = RandomForestClassifier(n_jobs =-1)

parameters = {'n_estimators' : [200,400,600], 'criterion':['gini'], 'max_depth':[5,10,30,50,100]}

grid_rf = GridSearchCV(rfc, parameters , scoring='accuracy', cv=5)
grid_rf.fit(x_train, y_train)

#Best parameters for our svc model

grid_rf.best_params_
#Let's run our RFC again with the best parameters.

rfc = RandomForestClassifier(n_estimators=400,max_depth =10, n_jobs =-1)

rfc.fit(x_train, y_train)

rfc_pred = rfc.predict(x_test)

report(y_test, rfc_pred)
rf_acc_score = accuracy_score(y_test, rfc_pred)

rf_acc_score