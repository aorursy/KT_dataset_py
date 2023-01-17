# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeClassifier 

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_df = pd.read_csv('/kaggle/input/traningSet.csv')
raw_df.head(10).T
raw_df['year'] = 2019 - raw_df['Created_Timestamp'].str[:4].astype(int)



continuous_features = ['Open_Issues_Count', 'Watchers_Count', 'Contributors_Count', 'Forks_Count', 'Size', \

           'year','Stars_Count' ]

binary_features = ['Fork','Issues_enabled', 'Wiki_enabled','Pages_enabled' ]

categorical_features = ['Language','Host_Type']
raw_df.isna().sum()
missing_value_features = [ 'Open_Issues_Count', 'Watchers_Count', 'Size']

for feature in missing_value_features:

    raw_df[feature].fillna(raw_df[feature].mode()[0], inplace=True)

raw_df['Pages_enabled'].fillna(0, inplace=True)

raw_df.head(10).T
for feature in binary_features:

    raw_df[feature] = raw_df[feature].astype(int)
raw_df['has_readme'] = raw_df['Readme_filename'].notna() * 1

raw_df['has_description'] = raw_df['Description'].notna() * 1

binary_features.extend(['has_readme','has_description'])
# Double check if any missing in continuous_features

raw_df[continuous_features + binary_features + categorical_features].isna().sum()
# for feature in continuous_features + binary_features:

#     raw_df.plot.scatter(feature, 'New_Stars_Count')
languages = list(raw_df['Language'].value_counts()[1:11].index)

for language in languages:

    raw_df[language] = raw_df.Language.apply(lambda x : int(x == language))

raw_df['other_language'] = (raw_df[languages].sum(axis=1) == 0).astype(int)  
for Host_Type in ['GitLab', 'GitHub']:

    raw_df[Host_Type] = raw_df.Host_Type.apply(lambda x : int(x == Host_Type))
# missing handle timestamp...





features_notNeed = ['ID', 'Host_Type', 'Name_with_Owner', 'Description', 'Created_Timestamp', 'Updated_Timestamp', \

                   'Last_pushed_Timestamp', 'Homepage_URL', 'Language', 'Mirror_URL', 'Default_branch', 'UUID', \

                   'Fork_Source_Name_with_Owner', 'License', 'Readme_filename', 'Changelog_filename', \

                   'Contributing_guidelines_filename', 'License_filename', 'Last_Synced_Timestamp', 'SourceRank', \

                   'Display_Name', 'duplicate_ID']

df = raw_df.drop(features_notNeed, axis=1)
features = list(df.columns)

features.remove('New_Stars_Count')

k_fold = KFold(n_splits=2, shuffle=True, random_state=11)
def get_cv_results(regressor):

    

    results = []

    for train, test in k_fold.split(df):

        regressor.fit(df.loc[train, features], df.loc[train, 'New_Stars_Count'])

        y_predicted = regressor.predict(df.loc[test, features])

        accuracy = mean_squared_error(df.loc[test, 'New_Stars_Count'], y_predicted)

        results.append(accuracy)



    return np.mean(results), np.std(results)
gbr = GradientBoostingRegressor(

    max_depth=5,

    n_estimators=100

)



get_cv_results(gbr)
rforest = RandomForestRegressor(

    random_state=11, 

    max_depth=10,

    n_estimators=200

)

get_cv_results(rforest)
rforest.fit(df[features], df['New_Stars_Count'])  

for feature,score in sorted(zip(features,rforest.feature_importances_), key=lambda x:x[1], reverse=True):

    print(feature, ' ', score)
regtreemo = DecisionTreeRegressor(

    random_state=1, 

    max_depth=None,

    min_samples_leaf=1,

    max_features=None,

    max_leaf_nodes=None )



regtreemo.fit(df[features], df['New_Stars_Count'])
get_cv_results(regtreemo)
newregtreemo = DecisionTreeRegressor(

    random_state=1, 

    max_depth=6,

    min_samples_leaf=3 )



newregtreemo.fit(df[features], df['New_Stars_Count'])
get_cv_results(newregtreemo)
for feature,score in zip(features,newregtreemo.feature_importances_):

    print(feature, ' ', score)
hp_values = range(1,50,2)

all_mu = []

all_sigma = []



for m in hp_values:



    dtree=DecisionTreeClassifier(

        criterion='entropy', 

        random_state=1, 

        max_depth=m,

        min_samples_leaf=m,

    )



    mu, sigma = get_cv_results(dtree)

    all_mu.append(mu)

    all_sigma.append(sigma)

    

    print(m, mu, sigma)
plt.figure(figsize=(14, 5))

plt.plot(hp_values, all_mu)

plt.ylabel('Cross Validation Accuracy')

plt.xlabel('Max Depth')
plt.figure(figsize=(14, 5))

plt.plot(hp_values, all_sigma)

plt.ylabel('Cross Validation Std Dev.')

plt.xlabel('Max Depth')