# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

import math

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import ExtraTreesClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train = train.drop(['Id'], axis=1)

y = train.iloc[:,-1]

train = train.iloc[:,:-1]

test = pd.read_csv("../input/test.csv")

test_Id = test.iloc[:,0]

test = test.drop(['Id'], axis=1)

all_data = train.append(test)

train.head()
train.describe()
train = train.drop(['Soil_Type7','Soil_Type15'], axis=1)
sns.boxplot('Horizontal_Distance_To_Hydrology', data = all_data)
#correlation matrix

corrmat = all_data.iloc[:,:10].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
plt.figure(figsize = (20,8))

plt.subplot(1,2,1)

sns.scatterplot(x = 'Horizontal_Distance_To_Hydrology', y = 'Vertical_Distance_To_Hydrology', data = all_data)

plt.subplot(1,2,2)

sns.scatterplot(x = 'Hillshade_3pm', y = 'Aspect', data = all_data)
a = all_data['Horizontal_Distance_To_Hydrology']

b = all_data['Vertical_Distance_To_Hydrology']

all_data['distance_to_hydrology'] = np.sqrt(np.power(a,2) + np.power(b,2))

all_data['Horizontal_distance'] = (all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Roadways']  + all_data['Horizontal_Distance_To_Fire_Points'])/3

all_data['average_hillshade'] = (all_data['Hillshade_3pm'] + all_data['Hillshade_Noon'] + all_data['Hillshade_9am'])/3

# high negative correlation, therefore making new feature

all_data['Aspect_hillshade'] = (all_data['Aspect']*all_data['Hillshade_9am'])/255

all_data['slope_hillshade'] = (all_data['Slope']*all_data['Hillshade_Noon'])/255

all_data['Elevation'] = [math.floor(v/50.0) for v in all_data['Elevation']]
all_data['EVDtH'] = all_data.Elevation-all_data.Vertical_Distance_To_Hydrology



all_data['EHDtH'] = all_data.Elevation-all_data.Horizontal_Distance_To_Hydrology*0.2



all_data['Distanse_to_Hydrolody'] = (all_data['Horizontal_Distance_To_Hydrology']**2+all_data['Vertical_Distance_To_Hydrology']**2)**0.5



all_data['Hydro_Fire_1'] = all_data['Horizontal_Distance_To_Hydrology']+all_data['Horizontal_Distance_To_Fire_Points']



all_data['Hydro_Fire_2'] = abs(all_data['Horizontal_Distance_To_Hydrology']-all_data['Horizontal_Distance_To_Fire_Points'])



all_data['Hydro_Road_1'] = abs(all_data['Horizontal_Distance_To_Hydrology']+all_data['Horizontal_Distance_To_Roadways'])



all_data['Hydro_Road_2'] = abs(all_data['Horizontal_Distance_To_Hydrology']-all_data['Horizontal_Distance_To_Roadways'])



all_data['Fire_Road_1'] = abs(all_data['Horizontal_Distance_To_Fire_Points']+all_data['Horizontal_Distance_To_Roadways'])



all_data['Fire_Road_2'] = abs(all_data['Horizontal_Distance_To_Fire_Points']-all_data['Horizontal_Distance_To_Roadways'])

all_data.head()
num_labels = [i for i in all_data.columns[0:10]]

b = ['distance_to_hydrology','Horizontal_distance','average_hillshade','Aspect_hillshade','slope_hillshade','EVDtH',

      'EHDtH','Distanse_to_Hydrolody', 'Hydro_Fire_1','Hydro_Fire_2','Hydro_Road_1', 'Hydro_Road_2','Fire_Road_1','Fire_Road_2']

num_labels.extend(b)
train_data = all_data.iloc[:train.shape[0],:]



test_data = all_data.iloc[train.shape[0]:,:]
'''

#Outlier treatment for only continuous variables

for col in num_labels:

    percentiles = train_data[col].quantile([0.01,0.99]).values

    train_data[col][train_data[col] <= percentiles[0]] = percentiles[0]

    train_data[col][train_data[col] >= percentiles[1]] = percentiles[1]

    test_data[col][test_data[col] <= percentiles[0]] = percentiles[0]

    test_data[col][test_data[col] >= percentiles[1]] = percentiles[1]

'''
# Scaling

rs = RobustScaler()

rs.fit(train_data)

train_data = rs.transform(train_data)

test_data = rs.transform(test_data)

train_data = pd.DataFrame(train_data, columns = all_data.columns)

test_data = pd.DataFrame(test_data, columns = all_data.columns)
'''

# plot histograms to see skewness

m=1

plt.figure(figsize = (20,30))

for i in num_labels:

    plt.subplot(8,3,m)

    sns.distplot(train_data[i],kde=False)

    m = m+1

'''
'''

from scipy.stats import skew



#finding skewness of all variables

skewed_feats = train_data[num_labels].apply(lambda x: skew(x.dropna()))

#adjusting features having skewness >0.75

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

train_data[skewed_feats] = np.log1p(train_data[skewed_feats])

test_data[skewed_feats] = np.log1p(test_data[skewed_feats])

'''
x_train, x_test, y_train, y_test = train_test_split(train_data, y, test_size = 0.1, shuffle= True)
def fit(model, X, y, X_test):

    model.fit(X, y)

    pred = model.predict(X_test)

    return pred



def accuracy(y_actuals, y_predicted):

    print(accuracy_score(y_actuals, y_predicted))

etc = ExtraTreesClassifier(n_estimators=400)

print(cross_val_score(etc,x_train,y_train,cv = 5).mean())
lgb = LGBMClassifier(num_leaves = 70)

#lgb_predictions = fit(lgb, x_train, y_train, x_test)

#accuracy(y_test, lgb_predictions)
print(cross_val_score(lgb,x_train,y_train,cv = 5).mean())
etc.fit(train_data, y)

test_predictions = etc.predict(test_data)
solutions = pd.DataFrame({'Id':test_Id, 'Cover_Type':test_predictions})

solutions.to_csv('submission.csv',index=False)