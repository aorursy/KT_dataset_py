import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) # getting future warnings for SGD classifier



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from sklearn.svm import LinearSVC



X = pd.read_csv('../input/winemag-data-130k-v2.csv')

Xunmod = X

descript=X.description

X.head()
# For the moment it looks like a number of the columns will be redundant, however I will re-introduce

# them slowly at later points to see how they impact classification. 

X = X.drop(['Unnamed: 0','description','designation', 'province','region_1','region_2','taster_name',

            'taster_twitter_handle','title','winery'],axis=1)



X.head()
## Country

X.country = X.country.fillna('US') # Fill NaNs with the mode



countries = X.country.value_counts()

countries = countries[countries>1000]

plt.xticks(rotation=90)

sns.barplot(countries.index,countries.values);
## Price

X.price = X.price.fillna(np.mean(X.price))

#print(X.price.unique())

sns.distplot(X.price[X.price<200],kde=False);
X.variety = X.variety.fillna('Other')

X.variety = X.variety.astype('category').cat.codes

X.info()
#Look at which country has the most expensive wine, and also which country scores the most points 

#(on average)



countryVsPoints = X.groupby(by=['country'], as_index=False)['points'].mean()

countryVsPoints = countryVsPoints.sort_values(['points'], ascending=False)

fig, ax = plt.subplots(figsize=(10,10))

ax.set_ylim([80,100])

plt.xticks(rotation=90)

sns.barplot(countryVsPoints['country'],countryVsPoints['points'])



countryVsPrice = X.groupby(by=['country'], as_index=False)['price'].mean()

countryVsPrice = countryVsPrice.sort_values(['price'], ascending=False)

fig, ax = plt.subplots(figsize=(10,10))

plt.xticks(rotation=90)

sns.barplot(countryVsPrice['country'],countryVsPrice['price'])



countryVsPoints['points'].corr(countryVsPrice['price'])
# For later



def encoder(data):

    '''Map the categorical variables to numbers to work with scikit learn'''

    for col in data.columns:

        if data.dtypes[col] == "object":

            le = preprocessing.LabelEncoder()

            le.fit(data[col])

            data[col] = le.transform(data[col])

    return data

X = encoder(X)

#correlation map

fig,ax = plt.subplots()

sns.heatmap(X.corr(), annot=True, linewidths=.5,ax=ax)
# It seems like the only think that really correlates so far is price and points, which makes sense. 

# Lets take a look at this distribution and see if we can't fit some regression to it.

ax = sns.jointplot(x="price", y="points", data=X);
# This would be too difficult to plot any kind of distribution to.

# Lets work with the description statistic and see if this will correlate to something else

# Can I predict whether or not the wine will be rated >90 by the review?

tempX = Xunmod.drop('price', axis=1)

tempy = X.price

trainX, testX, trainY, testY = train_test_split(tempX,tempy)



clf = Pipeline([('vect', TfidfVectorizer()),

                ('clf', SGDClassifier(alpha=4e-6))

])



# Use a cross-validation grid search to find the best parameters

parameters = {}

gs_clf = GridSearchCV(clf, parameters)



gs_clf.fit(trainX.description,(trainY>90))

results=pd.DataFrame(gs_clf.cv_results_)

CVscore = np.mean(cross_val_score(gs_clf,testX.description,(testY>90)))

print(results)
print("The mean cross validated score is {0:.1f}%".format(100*CVscore))