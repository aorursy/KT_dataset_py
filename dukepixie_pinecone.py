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
import pandas as pd

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



from xgboost import XGBRegressor
train = pd.read_csv('/kaggle/input/learn-together/train.csv')

train.shape
train.columns.tolist()
test = pd.read_csv('/kaggle/input/learn-together/test.csv')

test.shape
test.columns.tolist()
plt.figure(figsize=(16, 9))

sns.scatterplot(x="Horizontal_Distance_To_Hydrology", y="Elevation",

                hue="Cover_Type", alpha=.8, palette="rainbow", data=train)
plt.figure(figsize=(16, 9))

sns.scatterplot(x="Horizontal_Distance_To_Roadways", y="Elevation",

                hue="Cover_Type", alpha=.8, palette="rainbow", data=train)
plt.figure(figsize=(16, 9))

sns.scatterplot(x="Horizontal_Distance_To_Fire_Points", y="Elevation",

                hue="Cover_Type", alpha=.8, palette="rainbow", data=train)
plt.figure(figsize=(16, 9))

sns.swarmplot(x="Cover_Type", y="Elevation",

            palette='rainbow', data=train)
features = ['Elevation', 'Aspect', 'Slope',

            'Horizontal_Distance_To_Hydrology',

            'Vertical_Distance_To_Hydrology',

            'Horizontal_Distance_To_Roadways',

            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',

            'Horizontal_Distance_To_Fire_Points']

X = train[features]
sns.pairplot(X)
plt.figure(figsize=(16, 9))

sns.heatmap(X.corr(), annot=True, cmap = sns.diverging_palette(10, 220, as_cmap=True))
X = train.drop(['Id', 'Cover_Type'], axis=1)

y = train.Cover_Type

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
rf_model = RandomForestClassifier(n_estimators=500)

rf_model.fit(x_train, y_train)

rf_pred = rf_model.predict(x_test)



print(mean_absolute_error(y_test, rf_pred))
rf_preds = rf_model.predict(test.drop('Id', axis=1))



output = pd.DataFrame({'Id': test.Id,

                       'Cover_Type': rf_preds})

output.to_csv('rfc_submission.csv', index=False)
forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(x_train, y_train)

forest_pred = forest_model.predict(x_test)



print(mean_absolute_error(y_test, forest_pred))
forest_preds = forest_model.predict(test.drop('Id', axis=1))



output = pd.DataFrame({'Id': test.Id,

                       'Cover_Type': forest_preds.round(0)})

output.to_csv('forest_submission.csv', index=False)
XGB_model = XGBRegressor(n_estimators=500)

XGB_model.fit(x_train, y_train)

XGB_pred = XGB_model.predict(x_test)



print(mean_absolute_error(y_test, XGB_pred))
XGB_preds = XGB_model.predict(test.drop('Id', axis=1))



output = pd.DataFrame({'Id': test.Id,

                       'Cover_Type': XGB_preds.round(0).astype(int)})

output.to_csv('XGB_submission.csv', index=False)