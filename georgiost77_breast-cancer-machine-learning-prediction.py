import pandas as pd

from IPython.display import HTML



df = pd.read_csv("../input/data.csv")

HTML(df.to_html())

print(df.shape)

print(df.dtypes)

df.head(n=3)
# Remove unnecessary columns

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
# Find missing values

print('Missing values:\n{}'.format(df.isnull().sum()))



# Find duplicated records

print('\nNumber of duplicated records: {}'.format(df.duplicated().sum()))



# Find the unique values of 'diagnosis'.

print('\nUnique values of "diagnosis": {}'.format(df['diagnosis'].unique()))
total = df['diagnosis'].count()

malignant = df[df['diagnosis'] == "M"]['diagnosis'].count()

print("Malignant: ", malignant)

print("Benign: ", total - malignant)
# Generate statistics

df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



g = sns.PairGrid(df.iloc[:, 0:11], hue="diagnosis", palette="Set2")

g = g.map_diag(plt.hist, edgecolor="w")

g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)

plt.show()
df_corr = df.iloc[:, 1:11].corr()

plt.figure(figsize=(8,8))

sns.heatmap(df_corr, cmap="Blues", annot=True)

plt.show()
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
from sklearn.model_selection import train_test_split



array = df.values



# Define the independent variables as features.

features = array[:,1:]



# Define the target (dependent) variable as labels.

labels = array[:,0]



# Create a train/test split using 30% test size.

features_train, features_test, labels_train, labels_test = train_test_split(features,

                                                                            labels,

                                                                            test_size=0.3,

                                                                            random_state=42)



# Check the split printing the shape of each set.

print(features_train.shape, labels_train.shape)

print(features_test.shape, labels_test.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from time import time



def print_ml_results():

    t0 = time()



    # Create classifier.

    clf = KNeighborsClassifier()



    # Fit the classifier on the training features and labels.

    t0 = time()

    clf.fit(features_train, labels_train)

    print("Training time:", round(time()-t0, 3), "s")



    # Make predictions.

    t1 = time()

    predictions = clf.predict(features_test)

    print("Prediction time:", round(time()-t1, 3), "s")



    # Evaluate the model.

    accuracy = clf.score(features_test, labels_test)

    report = classification_report(labels_test, predictions)



    # Print the reports.

    print("\nReport:\n")

    print("Accuracy: {}".format(accuracy))

    print("\n", report)

    print(confusion_matrix(labels_test, predictions))



print_ml_results()
df_new = df[['diagnosis', 'radius_mean', 'texture_mean', 'smoothness_mean',

            'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean',

            'radius_se', 'texture_se', 'smoothness_se',

            'compactness_se', 'concave points_se', 'symmetry_se',

            'fractal_dimension_se', 'concavity_worst', 'symmetry_worst',

            'fractal_dimension_worst']]



array = df_new.values



# Define the independent variables as features.

features_new = array[:,1:]



# Define the target (dependent) variable as labels.

labels_new = array[:,0]



# Create a train/test split using 30% test size.

features_train, features_test, labels_train, labels_test = train_test_split(features_new, labels_new, \

                                                                            test_size=0.3, random_state=42)



print_ml_results()
from sklearn.preprocessing import MinMaxScaler

import numpy as np



np.set_printoptions(precision=2, suppress=True)



scaler = MinMaxScaler(feature_range=(0,1))

features_scaled = scaler.fit_transform(features)

print("Unscaled data\n", features_train)

print("\nScaled data\n", features_scaled)
from sklearn.decomposition import PCA



pca = PCA(30)

projected = pca.fit_transform(features)

pca_inversed_data = pca.inverse_transform(np.eye(30))



plt.style.use('seaborn')



def plot_pca():

    plt.figure(figsize=(10, 4))

    plt.plot(pca_inversed_data.mean(axis=0), '--o', label = 'mean')

    plt.plot(np.square(pca_inversed_data.std(axis=0)), '--o', label = 'variance')

    plt.ylabel('Feature Contribution')

    plt.xlabel('Feature Index')

    plt.legend(loc='best')

    plt.xticks(np.arange(0, 30, 1.0))

    plt.show()



    plt.figure(figsize = (10, 4))

    plt.plot(np.cumsum(pca.explained_variance_ratio_), '--o')

    plt.xlabel('Principal Component')

    plt.ylabel('Cumulative Explained Variance')

    plt.xticks(np.arange(0, 30, 1.0))

    plt.show()



plot_pca()
projected_scaled = pca.fit_transform(features_scaled)

pca_inversed_data = pca.inverse_transform(np.eye(30))



plot_pca()
from sklearn.feature_selection import SelectKBest



select = SelectKBest()

select.fit(features_train, labels_train)

scores = select.scores_

# Show the scores in a table

feature_scores = zip(df.columns.values.tolist(), scores)

ordered_feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)

for feature, score in ordered_feature_scores:

    print(feature, score)
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import GridSearchCV



# Create the scaler.

scaler = MinMaxScaler(feature_range=(0,1))



# Scale down all the features (both train and test dataset).

features = scaler.fit_transform(features)



# Create a train/test split using 30% test size.

features_train, features_test, labels_train, labels_test = train_test_split(features,

                                                                            labels,

                                                                            test_size=0.3,

                                                                            random_state=42)



# Create the classifier.

clf = KNeighborsClassifier()



# Create the pipeline.

pipeline = Pipeline([('reduce_dim', PCA()),

                     ('clf', clf)])



# Create the parameters.

n_features_options = [1, 3, 5, 7]

n_neighbors = [2, 4, 6]

algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']



parameters = [

    {

        'reduce_dim': [PCA(iterated_power=7)],

        'reduce_dim__n_components': n_features_options,

        'clf__n_neighbors': n_neighbors,

        'clf__algorithm': algorithm

    },

    {

        'reduce_dim': [SelectKBest()],

        'reduce_dim__k': n_features_options,

        'clf__n_neighbors': n_neighbors,

        'clf__algorithm': algorithm

    }]



# Create a function to find the best estimator.

def get_best_estimator(n_splits):



    t0 = time()



    # Create Stratified ShuffleSplit cross-validator.

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=3)



    # Create grid search.

    grid = GridSearchCV(pipeline, param_grid=parameters, scoring=('f1'), cv=sss, refit='f1')



    # Fit pipeline on features_train and labels_train.

    grid.fit(features_train, labels_train)



    # Make predictions.

    predictions = grid.predict(features_test)



    # Test predictions using sklearn.classification_report().

    report = classification_report(labels_test, predictions)



    # Find the best parameters and scores.

    best_parameters = grid.best_params_

    best_score = grid.best_score_



    # Print the reports.

    print("\nReport:\n")

    print(report)

    print("Best f1-score:")

    print(best_score)

    print("Best parameters:")

    print(best_parameters)

    print(confusion_matrix(labels_test, predictions))

    print("Time passed: ", round(time() - t0, 3), "s")

    

    return grid.best_estimator_



get_best_estimator(n_splits=20)
# Build the estimator from PCA and univariate selection.

combined_features = FeatureUnion([('pca', PCA()), ('univ_select', SelectKBest())])



# Do grid search over k, n_components and K-NN parameters.

pipeline = Pipeline([('features', combined_features),

                     ('clf', clf)])



# Create the parameters.

n_features_options = [1, 3, 5, 7]

n_neighbors = [2, 4, 6]

algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']



parameters = [

    {

        'features__pca': [PCA(iterated_power=7)],

        'features__pca__n_components': n_features_options,

        'features__univ_select__k': n_features_options,

        'clf__n_neighbors': n_neighbors,

        'clf__algorithm': algorithm

    }]



get_best_estimator(20)