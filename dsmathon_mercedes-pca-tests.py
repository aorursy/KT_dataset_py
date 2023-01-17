# Importing main packages and settings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, OrthogonalMatchingPursuitCV
# Function for plotting the scores for different alphas used in Ridge regression

def display_plot(cv_scores, cv_scores_std):

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    ax.plot(alpha_space, cv_scores)



    std_error = cv_scores_std / np.sqrt(10)



    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)

    ax.set_ylabel('CV Score +/- Std Error')

    ax.set_xlabel('Alpha')

    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')

    ax.set_xlim([alpha_space[0], alpha_space[-1]])

    ax.set_xscale('log')

    plt.show()
# Loading the training dataset

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# turning object features into dummy variables

df_train_dummies = pd.get_dummies(df_train, drop_first=True)

df_test_dummies = pd.get_dummies(df_test, drop_first=True)



# dropping ID and the target variable

df_train_dummies = df_train_dummies.drop(['ID','y'], axis=1)

df_test_dummies = df_test_dummies.drop('ID', axis=1)



print("Clean Train DataFrame With Dummy Variables: {}".format(df_train_dummies.shape))

print("Clean Test DataFrame With Dummy Variables: {}".format(df_test_dummies.shape))
# concatenate to only include columns in both data sets

# the number should be based on the number of columns. Original is 30471. Now set to 15471 after outlier handling etc.

df_temp = pd.concat([df_train_dummies, df_test_dummies], join='inner')

df_temp_train = df_temp[:len(df_train.index)]

df_temp_test = df_temp[len(df_train.index):]



# check shapes of combined df and split out again

print(df_temp.shape)

print(df_temp_train.shape)

print(df_temp_test.shape)
# defining X and y

X = df_temp_train

test_X = df_temp_test

y = df_train['y']
X.head()
# Create a PCA instance: pca

pca = PCA()



# Fit the pca to 'samples'

pca.fit(X)



# Plot the explained variances

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.xticks(features)

plt.show()
# Create a PCA instance: pca

pca2 = PCA(n_components=50)



# Fit the pca to 'samples'

pca2.fit(X)



pca_X = pca2.transform(X)

pca_test_X = pca2.transform(test_X)



# Plot the explained variances

features = range(pca2.n_components_)

plt.bar(features, pca2.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.xticks(features)

plt.show()
print(pca_features.shape)
# Setup the array of alphas and lists to store scores

alpha_space = np.logspace(-4, 0, 20)

ridge_scores = []

ridge_scores_std = []



# Create a ridge regressor: ridge

ridge = Ridge(normalize=True)



# Compute scores over range of alphas

for alpha in alpha_space:



    # Specify the alpha value to use: ridge.alpha

    ridge.alpha = alpha

    

    # Perform 10-fold CV: ridge_cv_scores

    ridge_cv_scores = cross_val_score(ridge, X, y, cv=5)

    

    # Append the mean of ridge_cv_scores to ridge_scores

    ridge_scores.append(np.mean(ridge_cv_scores))

    

    # Append the std of ridge_cv_scores to ridge_scores_std

    ridge_scores_std.append(np.std(ridge_cv_scores))



# Display the plot

display_plot(ridge_scores, ridge_scores_std)
# Setup the array of alphas and lists to store scores

alpha_space = np.logspace(-4, 0, 20)

ridge_scores = []

ridge_scores_std = []



# Create a ridge regressor: ridge

ridge = Ridge(normalize=True)



# Compute scores over range of alphas

for alpha in alpha_space:



    # Specify the alpha value to use: ridge.alpha

    ridge.alpha = alpha

    

    # Perform 10-fold CV: ridge_cv_scores

    ridge_cv_scores = cross_val_score(ridge, pca_X, y, cv=5)

    

    # Append the mean of ridge_cv_scores to ridge_scores

    ridge_scores.append(np.mean(ridge_cv_scores))

    

    # Append the std of ridge_cv_scores to ridge_scores_std

    ridge_scores_std.append(np.std(ridge_cv_scores))



# Display the plot

display_plot(ridge_scores, ridge_scores_std)
# instantiating different regressors

rcv = RidgeCV()

ecv = ElasticNetCV()

ompcv = OrthogonalMatchingPursuitCV()
# bad for but just for now:

import warnings

warnings.filterwarnings("ignore")



# Compute 10-fold cross-validation scores: cv_scores

cv_scores_rcv = cross_val_score(rcv, X, y, cv=5)

cv_scores_ecv = cross_val_score(ecv, X, y, cv=5)

cv_scores_ompcv = cross_val_score(ompcv, X, y, cv=5)



# Compute 10-fold cross-validation scores: cv_scores

cv_scores_pca_rcv = cross_val_score(rcv, pca_X, y, cv=5)

cv_scores_pca_ecv = cross_val_score(ecv, pca_X, y, cv=5)

cv_scores_pca_ompcv = cross_val_score(ompcv, pca_X, y, cv=5)



# Print the 10-fold cross-validation scores

print(cv_scores_rcv)

print(cv_scores_ecv)

print(cv_scores_ompcv)

print(cv_scores_pca_rcv)

print(cv_scores_pca_ecv)

print(cv_scores_pca_ompcv)



print("Average 5-Fold RidgeCV CV Score: {}".format(np.mean(cv_scores_rcv)))

print("Average 5-Fold ElasticNetCV CV Score: {}".format(np.mean(cv_scores_ecv)))

print("Average 5-Fold OrthogonalMatchingPursuitCV CV Score: {}".format(np.mean(cv_scores_ompcv)))

print("Average 5-Fold PCA RidgeCV CV Score: {}".format(np.mean(cv_scores_pca_rcv)))

print("Average 5-Fold PAC ElasticNetCV CV Score: {}".format(np.mean(cv_scores_pca_ecv)))

print("Average 5-Fold PCA OrthogonalMatchingPursuitCV CV Score: {}".format(np.mean(cv_scores_pca_ompcv)))
# Create a PCA instance: pca

pca10 = PCA(n_components=10)

pca20 = PCA(n_components=20)

pca50 = PCA(n_components=50)

pca100 = PCA(n_components=100)

pca200 = PCA(n_components=200)

pca300 = PCA(n_components=300)



# Fit the pca to 'samples'

pca10.fit(X)

pca20.fit(X)

pca50.fit(X)

pca100.fit(X)

pca200.fit(X)

pca300.fit(X)



pca10_X = pca10.transform(X)

pca20_X = pca20.transform(X)

pca50_X = pca50.transform(X)

pca100_X = pca100.transform(X)

pca200_X = pca200.transform(X)

pca300_X = pca300.transform(X)
# bad for but just for now:

import warnings

warnings.filterwarnings("ignore")



# Compute 5-fold cross-validation scores: cv_scores

cv_scores_pca10_ecv = cross_val_score(ecv, pca10_X, y, cv=5)

cv_scores_pca20_ecv = cross_val_score(ecv, pca20_X, y, cv=5)

cv_scores_pca50_ecv = cross_val_score(ecv, pca50_X, y, cv=5)

cv_scores_pca100_ecv = cross_val_score(ecv, pca100_X, y, cv=5)

cv_scores_pca200_ecv = cross_val_score(ecv, pca200_X, y, cv=5)

cv_scores_pca300_ecv = cross_val_score(ecv, pca300_X, y, cv=5)

cv_scores_nopca_ecv = cross_val_score(ecv, X, y, cv=5)



print("Average 5-Fold 10 PCA ElasticNetCV CV Score: {}".format(np.mean(cv_scores_pca10_ecv)))

print("Average 5-Fold 20 PCA ElasticNetCV CV Score: {}".format(np.mean(cv_scores_pca20_ecv)))

print("Average 5-Fold 50 PCA ElasticNetCV CV Score: {}".format(np.mean(cv_scores_pca50_ecv)))

print("Average 5-Fold 100 PCA ElasticNetCV CV Score: {}".format(np.mean(cv_scores_pca100_ecv)))

print("Average 5-Fold 200 PCA ElasticNetCV CV Score: {}".format(np.mean(cv_scores_pca200_ecv)))

print("Average 5-Fold 300 PCA ElasticNetCV CV Score: {}".format(np.mean(cv_scores_pca300_ecv)))

print("Average 5-Fold No PCA ElasticNetCV CV Score: {}".format(np.mean(cv_scores_nopca_ecv)))