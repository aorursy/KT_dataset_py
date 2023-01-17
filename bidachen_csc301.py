import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.neighbors import KNeighborsRegressor

import tensorflow as tf

from keras import models, layers, optimizers, regularizers

from keras import regularizers

import missingno as msno 
df = pd.read_csv("../input/dataset.csv")

df.columns
df.shape
number_features = ['branches', 'commits','createdAt', 'diskUsage', 'followers', 'forkCount', 'iClosedComments', 'iClosedParticipants', \

                'iOpenComments', 'iOpenParticipants', 'issuesClosed', 'issuesOpen', 'members', 'prClosed', 'prClosedComments', \

                'prClosedCommits', 'prMerged', 'prMergedComments', 'prMergedCommits', 'prOpen', 'prOpenComments', 'prOpenCommits', \

                'pushedAt', 'readmeCharCount', 'readmeLinkCount', 'readmeSize', 'readmeWordCount', 'releases', 'subscribersCount']



catogorical_feature = ['primaryLanguage']



boolean_features = ['hasWikiEnabled', 'isArchived', 'websiteUrl']



columns_not_need = ['Unnamed: 0', 'description', 'following', 'gistComments', 'gistStar', 'gists', 'license', 'location', 'login', \

                   'organizations', 'reponame', 'repositories', 'siteAdmin', 'tags', 'type', 'updatedAt']



target = ['stars']



features_need_process = ['createdAt', 'pushedAt', \

                        'hasWikiEnabled', 'isArchived', 'websiteUrl' \

                        'primaryLanguage'] 
df.drop(columns_not_need, axis=1, inplace=True)
msno.bar(df, color=[0.5,0.5,0.3])
df.followers.fillna(0, inplace=True)

df.members.fillna(0, inplace=True)

df.commits.fillna(df.commits.mean(), inplace=True)

df.primaryLanguage.fillna('others', inplace=True)
df['hasWikiEnabled'] = df.hasWikiEnabled.apply(lambda x : int(x == 'True'))

df['isArchived'] = df.isArchived.apply(lambda x : int(x == 'True'))

df['websiteUrl'] = df['websiteUrl'].notna() * 1
df['createdAt'] = 2019 - df['createdAt'].str[:4].astype(int)

df['pushedAt'] = 2019 - df['pushedAt'].str[:4].astype(int)
list(df['primaryLanguage'].value_counts()[:11].index)
languages = list(df['primaryLanguage'].value_counts()[:11].index)

languages.remove('others')

for language in languages:

    df[language] = df.primaryLanguage.apply(lambda x : int(x == language))

df['other_language'] = (df[languages].sum(axis=1) == 0).astype(int)
boolean_features.extend(languages)

boolean_features.append('other_language')
df.drop('primaryLanguage', axis=1, inplace=True)
features = list(df.columns)

features.remove('stars')

k_fold = KFold(n_splits=5, shuffle=True, random_state=11)
df.shape
def get_cv_results(regressor):

    

    results = []

    for train, test in k_fold.split(df):

        regressor.fit(df.loc[train, features], df.loc[train, 'stars'])

        y_predicted = regressor.predict(df.loc[test, features])

        accuracy = mean_squared_error(df.loc[test, 'stars'], y_predicted)

        results.append(accuracy)



    print("mean square error: ",np.mean(results))
rfr = RandomForestRegressor(

    random_state=11, 

    max_depth=7,

    n_estimators=120

)

get_cv_results(rfr)
mul_reg = LinearRegression()

get_cv_results(mul_reg)
knn = KNeighborsRegressor(n_neighbors=10)

get_cv_results(knn)
scaler = StandardScaler()

X = scaler.fit_transform(df[features])

Y = df['stars'].values.reshape(-1,1)



train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
svr = svm.SVR(epsilon=0.2)

svr.fit(train_x, train_y)

print("mean square error: ", mean_squared_error(test_y, svr.predict(test_x)))
net = models.Sequential()

net.add(layers.Dense(64, input_dim=train_x.shape[1], kernel_regularizer=regularizers.l1(0.1), activation='relu'))

net.add(layers.Dropout(0.3))



net.add(layers.Dense(128, kernel_regularizer=regularizers.l1(0.1), activation='relu'))

net.add(layers.Dropout(0.3))



net.add(layers.Dense(256, kernel_regularizer=regularizers.l1(0.1), activation='relu'))

net.add(layers.Dropout(0.3))



net.add(layers.Dense(512, kernel_regularizer=regularizers.l1(0.1), activation='relu'))

net.add(layers.Dropout(0.3))



net.add(layers.Dense(1, activation='linear'))



net.compile(loss='mean_squared_error',

            optimizer='adam',

            metrics=['mean_squared_error'])
net.fit(train_x, train_y, epochs=50, batch_size=64, validation_split = 0.1)
print("mean square error: ", mean_squared_error(test_y, net.predict(test_x)))