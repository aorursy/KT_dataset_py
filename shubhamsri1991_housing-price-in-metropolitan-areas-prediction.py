# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
bangalore_dataset = pd.read_csv('../input/housing-prices-in-metropolitan-areas-of-india/Bangalore.csv')

delhi_dataset = pd.read_csv('../input/housing-prices-in-metropolitan-areas-of-india/Delhi.csv')

kolkata_dataset = pd.read_csv('../input/housing-prices-in-metropolitan-areas-of-india/Kolkata.csv')

mumbai_dataset = pd.read_csv('../input/housing-prices-in-metropolitan-areas-of-india/Mumbai.csv')
bangalore_dataset.shape, delhi_dataset.shape, kolkata_dataset.shape, mumbai_dataset.shape
pd.pandas.set_option('display.max_columns', None)

bangalore_dataset.head()
pd.pandas.set_option('display.max_columns', None)

delhi_dataset.head()
pd.pandas.set_option('display.max_columns', None)

kolkata_dataset.head()
pd.pandas.set_option('display.max_columns', None)

mumbai_dataset.head()
bangalore_dataset.info()
delhi_dataset.info()
kolkata_dataset.info()
mumbai_dataset.info()
bangalore_dataset.duplicated().sum()
delhi_dataset.duplicated().sum()
kolkata_dataset.duplicated().sum()
mumbai_dataset.duplicated().sum()
bangalore_dataset.describe()
delhi_dataset.describe()
kolkata_dataset.describe()
mumbai_dataset.describe()
plt.figure(figsize=(12,12))

plt.subplot(221)

sns.boxplot(bangalore_dataset['Area'])

plt.title('Outliers In Area In Bangalore Dataset')

plt.subplot(222)

sns.boxplot(delhi_dataset['Area'])

plt.title('Outliers In Area In Delhi Dataset')

plt.subplot(223)

sns.boxplot(kolkata_dataset['Area'])

plt.title('Outliers In Area In Kolkata Dataset')

plt.subplot(224)

sns.boxplot(mumbai_dataset['Area'])

plt.title('Outliers In Area In Mumbai Dataset')

plt.show()
plt.figure(figsize=(12,12))

plt.subplot(221)

sns.boxplot(bangalore_dataset['No. of Bedrooms'])

plt.title('Outliers In No. of Bedrooms In Bangalore Dataset')

plt.subplot(222)

sns.boxplot(delhi_dataset['No. of Bedrooms'])

plt.title('Outliers In No. of Bedrooms In Delhi Dataset')

plt.subplot(223)

sns.boxplot(kolkata_dataset['No. of Bedrooms'])

plt.title('Outliers In No. of Bedrooms In Kolkata Dataset')

plt.subplot(224)

sns.boxplot(mumbai_dataset['No. of Bedrooms'])

plt.title('Outliers In No. of Bedrooms In Mumbai Dataset')

plt.show()
plt.figure(figsize=(12,12))

plt.subplot(221)

sns.distplot(bangalore_dataset['Price'], bins=50)

plt.title('Price Distribution In Bangalore')

plt.subplot(222)

sns.distplot(delhi_dataset['Price'], bins=50)

plt.title('Price Distribution In Delhi')

plt.subplot(223)

sns.distplot(kolkata_dataset['Price'], bins=50)

plt.title('Price Distribution In Kolkata')

plt.subplot(224)

sns.distplot(mumbai_dataset['Price'], bins=50)

plt.title('Price Distribution In Mumbai')

plt.show()
plt.figure(figsize=(12,12))

plt.subplot(221)

plt.hist(bangalore_dataset['Area'], bins=50)

plt.title('Area Distribution In Bangalore')

plt.subplot(222)

plt.hist(delhi_dataset['Area'], bins=50)

plt.title('Area Distribution In Delhi')

plt.subplot(223)

plt.hist(kolkata_dataset['Area'], bins=50)

plt.title('Area Distribution In Kolkata')

plt.subplot(224)

plt.hist(mumbai_dataset['Area'], bins=50)

plt.title('Area Distribution In Mumbai')

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(221)

bangalore_dataset['No. of Bedrooms'].value_counts().plot(kind='bar',figsize=(12,10))

plt.title('No. of Bedrooms Distribution In Bangalore')

plt.subplot(222)

delhi_dataset['No. of Bedrooms'].value_counts().plot(kind='bar',figsize=(12,10))

plt.title('No. of Bedrooms Distribution In Delhi')

plt.subplot(223)

kolkata_dataset['No. of Bedrooms'].value_counts().plot(kind='bar',figsize=(12,10))

plt.title('No. of Bedrooms Distribution In Kolkata')

plt.subplot(224)

mumbai_dataset['No. of Bedrooms'].value_counts().plot(kind='bar',figsize=(12,10))

plt.title('No. of Bedrooms Distribution In Mumbai')

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(221)

bangalore_dataset['Gymnasium'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('Gymnasium Count In Bangalore')

plt.subplot(222)

bangalore_dataset['SwimmingPool'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('SwimmingPool Count In Bangalore')

plt.subplot(223)

bangalore_dataset['LandscapedGardens'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('LandscapedGardens Count In Bangalore')

plt.subplot(224)

bangalore_dataset['JoggingTrack'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('JoggingTrack Count In Bangalore')

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(221)

delhi_dataset['Gymnasium'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('Gymnasium Count In Delhi')

plt.subplot(222)

delhi_dataset['SwimmingPool'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('SwimmingPool Count In Delhi')

plt.subplot(223)

delhi_dataset['LandscapedGardens'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('LandscapedGardens Count In Delhi')

plt.subplot(224)

delhi_dataset['JoggingTrack'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('JoggingTrack Count In Delhi')

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(221)

kolkata_dataset['Gymnasium'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('Gymnasium Count In Kolkata')

plt.subplot(222)

kolkata_dataset['SwimmingPool'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('SwimmingPool Count In Kolkata')

plt.subplot(223)

kolkata_dataset['LandscapedGardens'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('LandscapedGardens Count In Kolkata')

plt.subplot(224)

kolkata_dataset['JoggingTrack'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('JoggingTrack Count In Kolkata')

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(221)

mumbai_dataset['Gymnasium'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('Gymnasium Count In Mumbai')

plt.subplot(222)

mumbai_dataset['SwimmingPool'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('SwimmingPool Count In Mumbai')

plt.subplot(223)

mumbai_dataset['LandscapedGardens'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('LandscapedGardens Count In Mumbai')

plt.subplot(224)

mumbai_dataset['JoggingTrack'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(12,10))

plt.title('JoggingTrack Count In Mumbai')

plt.show()
plt.figure(figsize=(12,10))

plt.subplot(221)

sns.distplot(bangalore_dataset['Price'][bangalore_dataset['Gymnasium']==1])

plt.title('Gymnasium Vs Bangalore')

plt.subplot(222)

sns.distplot(bangalore_dataset['Price'][bangalore_dataset['SwimmingPool']==1])

plt.title('SwimmingPool Vs Bangalore')

plt.subplot(223)

sns.distplot(bangalore_dataset['Price'][bangalore_dataset['LandscapedGardens']==1])

plt.title('LandscapedGardens Vs Bangalore')

plt.subplot(224)

sns.distplot(bangalore_dataset['Price'][bangalore_dataset['JoggingTrack']==1])

plt.title('JoggingTrack Vs Bangalore')

plt.show()
plt.figure(figsize=(12,10))

plt.subplot(221)

sns.distplot(delhi_dataset['Price'][delhi_dataset['Gymnasium']==1])

plt.title('Gymnasium Vs Delhi')

plt.subplot(222)

sns.distplot(delhi_dataset['Price'][delhi_dataset['SwimmingPool']==1])

plt.title('SwimmingPool Vs Delhi')

plt.subplot(223)

sns.distplot(delhi_dataset['Price'][delhi_dataset['LandscapedGardens']==1])

plt.title('LandscapedGardens Vs Delhi')

plt.subplot(224)

sns.distplot(delhi_dataset['Price'][delhi_dataset['JoggingTrack']==1])

plt.title('JoggingTrack Vs Delhi')

plt.show()
plt.figure(figsize=(12,10))

plt.subplot(221)

sns.distplot(kolkata_dataset['Price'][kolkata_dataset['Gymnasium']==1])

plt.title('Gymnasium Vs Kolkata')

plt.subplot(222)

sns.distplot(kolkata_dataset['Price'][kolkata_dataset['SwimmingPool']==1])

plt.title('SwimmingPool Vs Kolkata')

plt.subplot(223)

sns.distplot(kolkata_dataset['Price'][kolkata_dataset['LandscapedGardens']==1])

plt.title('LandscapedGardens Vs Kolkata')

plt.subplot(224)

sns.distplot(kolkata_dataset['Price'][kolkata_dataset['JoggingTrack']==1])

plt.title('JoggingTrack Vs Kolkata')

plt.show()
plt.figure(figsize=(12,10))

plt.subplot(221)

sns.distplot(mumbai_dataset['Price'][mumbai_dataset['Gymnasium']==1])

plt.title('Gymnasium Vs Mumbai')

plt.subplot(222)

sns.distplot(mumbai_dataset['Price'][mumbai_dataset['SwimmingPool']==1])

plt.title('SwimmingPool Vs Mumbai')

plt.subplot(223)

sns.distplot(mumbai_dataset['Price'][mumbai_dataset['LandscapedGardens']==1])

plt.title('LandscapedGardens Vs Mumbai')

plt.subplot(224)

sns.distplot(mumbai_dataset['Price'][mumbai_dataset['JoggingTrack']==1])

plt.title('JoggingTrack Vs Mumbai')

plt.show()
bangalore_dataset.drop_duplicates(inplace=True)

delhi_dataset.drop_duplicates(inplace=True)

kolkata_dataset.drop_duplicates(inplace=True)

mumbai_dataset.drop_duplicates(inplace=True)

bangalore_dataset.shape, delhi_dataset.shape, kolkata_dataset.shape, mumbai_dataset.shape
outlier_features = ['Area','No. of Bedrooms']

for feature in outlier_features:

    IQR = bangalore_dataset[feature].quantile(0.75) - bangalore_dataset[feature].quantile(0.25)

    lower_boundary = bangalore_dataset[feature].quantile(0.25) - (IQR*1.5)

    upper_boundary = bangalore_dataset[feature].quantile(0.75) + (IQR*1.5)

    bangalore_dataset.loc[bangalore_dataset[feature]<=lower_boundary, feature] = lower_boundary

    bangalore_dataset.loc[bangalore_dataset[feature]>=upper_boundary, feature] = upper_boundary

    

    IQR = delhi_dataset[feature].quantile(0.75) - delhi_dataset[feature].quantile(0.25)

    lower_boundary = delhi_dataset[feature].quantile(0.25) - (IQR*1.5)

    upper_boundary = delhi_dataset[feature].quantile(0.75) + (IQR*1.5)

    delhi_dataset.loc[delhi_dataset[feature]<=lower_boundary, feature] = lower_boundary

    delhi_dataset.loc[delhi_dataset[feature]>=upper_boundary, feature] = upper_boundary

    

    IQR = kolkata_dataset[feature].quantile(0.75) - kolkata_dataset[feature].quantile(0.25)

    lower_boundary = kolkata_dataset[feature].quantile(0.25) - (IQR*1.5)

    upper_boundary = kolkata_dataset[feature].quantile(0.75) + (IQR*1.5)

    kolkata_dataset.loc[kolkata_dataset[feature]<=lower_boundary, feature] = lower_boundary

    kolkata_dataset.loc[kolkata_dataset[feature]>=upper_boundary, feature] = upper_boundary

    

    IQR = mumbai_dataset[feature].quantile(0.75) - mumbai_dataset[feature].quantile(0.25)

    lower_boundary = mumbai_dataset[feature].quantile(0.25) - (IQR*1.5)

    upper_boundary = mumbai_dataset[feature].quantile(0.75) + (IQR*1.5)

    mumbai_dataset.loc[mumbai_dataset[feature]<=lower_boundary, feature] = lower_boundary

    mumbai_dataset.loc[mumbai_dataset[feature]>=upper_boundary, feature] = upper_boundary
plt.figure(figsize=(12,12))

plt.subplot(221)

sns.boxplot(bangalore_dataset['Area'])

plt.title('Outliers In Area In Bangalore Dataset')

plt.subplot(222)

sns.boxplot(delhi_dataset['Area'])

plt.title('Outliers In Area In Delhi Dataset')

plt.subplot(223)

sns.boxplot(kolkata_dataset['Area'])

plt.title('Outliers In Area In Kolkata Dataset')

plt.subplot(224)

sns.boxplot(mumbai_dataset['Area'])

plt.title('Outliers In Area In Mumbai Dataset')

plt.show()
plt.figure(figsize=(12,12))

plt.subplot(221)

sns.boxplot(bangalore_dataset['No. of Bedrooms'])

plt.title('Outliers In No. of Bedrooms In Bangalore Dataset')

plt.subplot(222)

sns.boxplot(delhi_dataset['No. of Bedrooms'])

plt.title('Outliers In No. of Bedrooms In Delhi Dataset')

plt.subplot(223)

sns.boxplot(kolkata_dataset['No. of Bedrooms'])

plt.title('Outliers In No. of Bedrooms In Kolkata Dataset')

plt.subplot(224)

sns.boxplot(mumbai_dataset['No. of Bedrooms'])

plt.title('Outliers In No. of Bedrooms In Mumbai Dataset')

plt.show()
houseing_price_dataset = pd.concat([bangalore_dataset, delhi_dataset, kolkata_dataset, mumbai_dataset], axis=0)

houseing_price_dataset.shape
plt.figure(figsize=(35,35))

sns.heatmap(houseing_price_dataset.corr(), annot=True, cmap='RdYlGn')

plt.show()
features = houseing_price_dataset.drop(['Price', 'Location'], axis=1)

label = houseing_price_dataset['Price']

features.shape, label.shape
plt.figure(figsize=(12,5))

from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor()

etr.fit(features, label)

feature_importance = pd.Series(etr.feature_importances_, index=features.columns)

feature_importance.nlargest(20).plot(kind='barh')

plt.show()
houseing_price_dataset = pd.get_dummies(houseing_price_dataset)

houseing_price_dataset.shape
houseing_price_dataset.columns
features = ['MaintenanceStaff','Gymnasium','SwimmingPool','LandscapedGardens','JoggingTrack','RainWaterHarvesting',

            'IndoorGames','ShoppingMall','Intercom','SportsFacility','ATM','ClubHouse','School','24X7Security',

            'PowerBackup','CarParking','StaffQuarter','Cafeteria','MultipurposeRoom','Hospital','WashingMachine',

            'AC','Wifi','BED','VaastuCompliant','Microwave','GolfCourse','TV','DiningTable',

            'Sofa','Refrigerator']

selected_features = houseing_price_dataset.drop(houseing_price_dataset[features], axis=1)

from sklearn.model_selection import train_test_split

features_train, features_test, label_train, label_test = train_test_split(selected_features, label, test_size=0.3, random_state=40)

features_train.shape, features_test.shape, label_train.shape, label_test.shape
#Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]



#Number of features to consider in every split

max_features = ['auto', 'sqrt']



#Maximum number of levels in a tree

max_depth = [int(x) for x in np.linspace(start=5, stop=30, num=6)]



#Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]



#Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
#Random Grid

random_grid = {'n_estimators' : n_estimators,

              'max_features' : max_features,

              'max_depth' : max_depth,

              'min_samples_split' : min_samples_split,

              'min_samples_leaf' : min_samples_leaf}
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()

random_forest_model = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid, n_jobs=1, random_state=42,

                                        cv=5, n_iter=10, verbose=2, scoring='neg_mean_squared_error')

random_forest_model.fit(features_train, label_train)
label_pred = random_forest_model.predict(features_test)

label_pred
from sklearn.metrics import r2_score

r2_score(label_train, random_forest_model.predict(features_train))
plt.figure(figsize=(12,5))

sns.distplot(label_train-random_forest_model.predict(features_train))

plt.show()
r2_score(label_test, random_forest_model.predict(features_test))
plt.figure(figsize=(12,5))

sns.distplot(label_test-random_forest_model.predict(features_test))

plt.show()