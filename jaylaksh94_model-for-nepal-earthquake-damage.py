import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings('ignore')
train_values = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/train_values.csv')

train_labels = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/train_labels.csv')

test_values = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/test_values.csv')

submission = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/submission_format.csv')

train_values.head()
test_values.head()
print('Teaining feature shape: ',train_values.shape,'\nTraining label shape :',train_labels.shape)
# Let's drop the id columns

train_values = train_values[train_values.columns[4:]]

test_values = test_values[test_values.columns[4:]]

train_values.info()
#Check for the missing values



# (train_values.isnull().sum()/train_values.shape[0])*100

train_labels = train_labels[['damage_grade']]

train_labels.info()
sns.countplot(train_labels['damage_grade'])

plt.title('Number of Buildings with Each Damage Grade');
selected_features = ['foundation_type', 

                     'area_percentage', 

                     'height_percentage',

                     'count_floors_pre_eq',

                     'land_surface_condition',

                     'has_superstructure_cement_mortar_stone']



train_values_subset = train_values[selected_features]



sns.pairplot(train_values_subset.join(train_labels), 

             hue='damage_grade')
plt.figure(figsize=(15,13))

data_corr = train_values.corr()

sns.heatmap(data_corr)

plt.show();
train_values_new = pd.get_dummies(train_values)

test_values_new = pd.get_dummies(test_values)



print('Training dataset :',train_values_new.shape)

print()

print('Test dataset :',test_values_new.shape)
from xgboost import plot_importance

from xgboost import XGBClassifier



model = XGBClassifier()

model.fit(train_values_new,train_labels)
# Print the name and the gini importance of each features

# for features in zip(train_values_new.columns,model.feature_importances_):

#     print(features)



# Horizontal bar chart for feature Importance

feat_importances = pd.Series(model.feature_importances_,index= train_values_new.columns)

feat_importances = feat_importances.nlargest(18)

plt.figure(figsize=(10,8))

feat_importances.plot(kind='barh')

plt.style.use('fivethirtyeight')

plt.xlabel('Score')

plt.title('Feature Importance Score')

plt.show();
import_features = list(feat_importances.index)



train_values_df = train_values_new[import_features]

test_values_df = test_values_new[import_features]



print('Training dataset :',train_values_df.shape)

print()

print('Test dataset :',test_values_df.shape)
model = XGBClassifier()

model.fit(train_values_df,train_labels)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score



XGBoost_y_pred = model.predict(train_values_df)



print(accuracy_score(train_labels,XGBoost_y_pred))



print(f1_score(train_labels, XGBoost_y_pred, average='micro'))
from sklearn.model_selection import StratifiedKFold



cv_model = XGBClassifier()



kfold = StratifiedKFold(n_splits=5,random_state=12)

result = cross_val_score(cv_model,train_values_df,train_labels,cv=kfold)

result
result.mean()*100
y_pred = model.predict(test_values_df)
submission['damage_grade'] = y_pred
submission.head()
submission.to_csv('submission.csv')