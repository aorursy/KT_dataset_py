import numpy as np

import pandas as pd

from pylab import *



train = pd.read_csv('../input/train.csv', dtype={'Survived': bool})



def cleanup(df):



    df['Deck'] = df['Cabin'].apply(lambda s: s[0] if type(s)==str else 'N/A')

    df = df.drop(['Cabin'], axis=1)



    name_to_title = lambda s: next(ss[:-1] for ss in s.split() if ss.endswith('.'))

    df['Title'] = df['Name'].apply(name_to_title)

    df.loc[df['Title'].isin(['Don', 'Dr', 'Master', 'Rev', 'Jonkheer', 'Sir', 'Countess']), 'Title'] = 'HighStatus'

    df.loc[df['Title'].isin(['Capt', 'Col', 'Major']), 'Title'] = 'Officer'

    df.loc[df['Title'].isin(['Lady', 'Mlle', 'Ms']), 'Title'] = 'Miss'

    df.loc[df['Title'].isin(['Mme']), 'Title'] = 'Mrs'



    df.loc[df['Fare']==0, 'Fare'] = np.nan

    df['LogFare'] = df['Fare'].apply(np.log)



    return df



train = cleanup(train)



train.keys()
def dummies(df, column):

    df = pd.get_dummies(df[column]).astype(float)

    df.columns = [column + '_' + str(s) for s in df.columns]

    return df



def dummify(df):



    df['IsMale'] = df['Sex']=='male'

    df = df.drop('Sex', axis=1)



    df = df.join(dummies(df, 'Title'))

    df = df.drop('Title', axis=1)



    df = df.join(dummies(df, 'Deck'))

    df = df.drop('Deck', axis=1)



    df = df.join(dummies(df, 'Pclass'))

    df = df.drop('Pclass', axis=1)



    df = df.join(dummies(df, 'Embarked'))

    df = df.drop('Embarked', axis=1)



    df['AgeIsEstimated'] = (df['Age']==df['Age'].apply(np.floor)+.5).astype(float)

    df['AgeIsMissing'] = df['Age'].isnull()



    # Not sure what to do with tickets. Some have letters in them, most don't.

    # Include a column for the case that ticket is only numeric.

    def ticket_is_numeric(string):

        try:

            int(string)

        except ValueError:

            return False

        else:

            return True

    df['TicketIsNumeric'] = df['Ticket'].apply(ticket_is_numeric)

    df = df.drop('Ticket', axis=1)



    df = df.drop(['PassengerId', 'Name'], axis=1)



    return df



train = dummify(train)

train.keys()
# Data normalization

# Scikit-learn's scaling functions don't like NaN,

# so I have to do this manually

def normalize_with_nan(X):

    means = np.nanmean(X, axis=0)[None, :]

    stds = np.nanstd(X, axis=0)[None, :]

    stds[stds==0] = 1

    X -= means

    X /= stds

    to_units = lambda x: x*stds + means

    return X, to_units
from sklearn.cluster import KMeans

#kmeans = KMeans(n_clusters=10, random_state=0).fit(X)

#kmeans.cluster_centers_



# Copied from ali_m (MIT License):

# https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data



def kmeans_missing(X, n_clusters, max_iter=10):

    """Perform K-Means clustering on data with missing values.



    Args:

      X: An [n_samples, n_features] array of data to cluster.

      n_clusters: Number of clusters to form.

      max_iter: Maximum number of EM iterations to perform.



    Returns:

      labels: An [n_samples] vector of integer labels.

      centroids: An [n_clusters, n_features] array of cluster centroids.

      X_hat: Copy of X with the missing values filled in.

    """



    # Initialize missing values to their column means

    missing = ~np.isfinite(X)

    mu = np.nanmean(X, 0, keepdims=1)

    X_hat = np.where(missing, mu, X)



    for i in range(max_iter):

        if i > 0:

            # initialize KMeans with the previous set of centroids. this is much

            # faster and makes it easier to check convergence (since labels

            # won't be permuted on every iteration), but might be more prone to

            # getting stuck in local minima.

            model = KMeans(n_clusters, init=prev_centroids)

        else:

            # do multiple random initializations in parallel

            model = KMeans(n_clusters, n_jobs=-1)



        # perform clustering on the filled-in data

        labels = model.fit_predict(X_hat)

        centroids = model.cluster_centers_



        # fill in the missing values based on their cluster centroids

        X_hat[missing] = centroids[labels][missing]



        # when the labels have stopped changing then we have converged

        if i > 0 and np.all(labels == prev_labels):

            break



        prev_labels = labels

        prev_centroids = model.cluster_centers_



    return labels, centroids, X_hat
# Find closest centroids for given data

def fill_to_closest_cluster(X, centroids):

    test_labels = np.argmin(

        np.nansum((X[None, ...] - centroids[:, None, :])**2, 2), axis=0)

    nans = ~np.isfinite(X)

    X[nans] = centroids[test_labels][nans]

    return X
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier





# Use everything except the observations themselves

predictors = [c for c in train.columns if c not in ('Survived',)]



# Split into train and test

X_train, X_cv, y_train, y_cv = train_test_split(

    train[predictors], train['Survived'], test_size=0.25)



# Scale to near unity

X_train = X_train.astype(float).values

X_train, train_to_units = normalize_with_nan(X_train)

X_cv = X_cv.astype(float).values

X_cv, test_to_units = normalize_with_nan(X_cv)



# Clustering-based imputation

labels, centroids, X_hat = kmeans_missing(X_train, n_clusters=6)

centroid_df = pd.DataFrame(data=centroids, columns=predictors)

X_train[np.isnan(X_train)] = X_hat[np.isnan(X_train)]

X_cv = fill_to_closest_cluster(X_cv, centroids)



# Fit

# The parameters are manually picked using the Stetson--Harrison method

clf = RandomForestClassifier(

    criterion='entropy', max_depth=5, min_samples_leaf=3, n_estimators=60, bootstrap=False)

clf = clf.fit(X_train, y_train)



# Variables that have the highest "feature importance"

pd.Series(clf.feature_importances_, predictors).sort_values(ascending=False)


cv_pred = clf.predict(X_cv)

print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_cv, cv_pred)))

print("F1:{0:.3f}".format(metrics.f1_score(y_cv, cv_pred)))
test = pd.read_csv('../input/test.csv', dtype={'Survived': bool})

test_data = cleanup(test)

test_data = dummify(test_data)

test_data, _ = test_data.align(train, axis=1, join='right')

test_data = test_data[predictors]

X_test = test_data.astype(float).values

X_test, _ = normalize_with_nan(X_test)

X_test = fill_to_closest_cluster(X_test, centroids)

test_pred = clf.predict(X_test)

submission = test[['PassengerId']]

submission['Survived'] = test_pred.astype(int)

submission = submission.set_index('PassengerId')

submission.to_csv('random_forest_submission.csv')

submission.head()
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=0)

pca.fit(X_train, y_train)

X_train_trans = pca.transform(X_train)

X_train_trans_survived = pca.transform(X_train[y_train])

X_train_trans_perished = pca.transform(X_train[~y_train])

pca_components = pd.DataFrame(pca.components_.T, index=predictors)

sorted_squares = ((pca_components**2).sum(axis=1)).sort_values(ascending=False)

pca_components = pca_components.reindex(sorted_squares.index)



# Most "variant" dimensions 0 and 1

print(pca_components.head(10))
figure(figsize=(7,7))

scatter(*X_train_trans_survived.T, c='g')

scatter(*X_train_trans_perished.T, c='b')

scatter(*np.einsum('ij,kj->ik', centroids, pca.components_).T, c='k') # Centroid points

legend(['Survived', 'Perished'])

gca().set_xlabel('Wealth')

gca().set_ylabel('Femaleness')