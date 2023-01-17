# Import pandas

import pandas as pd



# Load dataset

df = pd.read_csv('../input/caravan-insurance-challenge.csv')
# Inspect data

print(df.head())
# Summary statistics 

print(df.describe())
# DataFrame information

print(df.info())
%matplotlib inline



# Import seaborn library

import seaborn as sns



# Set Seaborn

sns.set()



# Box Plots for the first 18 features

import matplotlib.pyplot as plt

df.iloc[:,1:10].plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)

plt.show()
# Correlation matrix between the first 9 features

g = sns.clustermap(df.iloc[:, 1:10].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
# import required libraries

from collections import Counter

import numpy as np



# Outlier detection 

def detect_outliers(df, n, features):

    

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col], 75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than n outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   
# detect outliers from columns

Outliers_to_drop = detect_outliers(df, 6, list(df.drop(['ORIGIN', 'CARAVAN'], axis=1)))
# Rows with more than 6 outliers

df.loc[Outliers_to_drop].shape
# Removing rows with outliers from the DataFrame

df.drop(Outliers_to_drop, axis=0, inplace=True)
df.shape
# Segregate features and labels into separate variables

X, y = df.drop(['ORIGIN', 'CARAVAN'], axis=1), df['CARAVAN']



# Import the StandardScaler

from sklearn.preprocessing import StandardScaler



# Convert data with input dtype int64 to float64

X = X.astype(float)



# Instantiate StandardScaler and use it to features

scaler = StandardScaler()

rescaledX = scaler.fit_transform(X)
# Import the plotting module and PCA class

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



# Get our explained variance ratios from PCA using all features

pca = PCA()

pca.fit(rescaledX)

exp_var = pca.explained_variance_ratio_

# plot the explained variance using a barplot

fig, ax = plt.subplots()

ax.bar(range(pca.n_components_),exp_var)

ax.set_xlabel('Principal Component')

plt.show()
# Calculate the cumulative explained variance

cum_exp_var = np.cumsum(exp_var)
# Plot the cumulative explained variance and draw a dashed line at 0.90.

fig, ax = plt.subplots()

ax.plot(range(85),cum_exp_var)

ax.axhline(y=0.9, linestyle='--')

plt.show()
n_components = 38



# Perform PCA with the chosen number of components and project data onto components

pca = PCA(n_components, random_state=10)

pca.fit(rescaledX)

pca_projection = pca.transform(rescaledX)
# import train_test_split

from sklearn.model_selection import train_test_split



# split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(pca_projection, y, test_size=0.3, random_state=42)
# import selected calssifiers from sklearn 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.linear_model.stochastic_gradient import SGDClassifier

from sklearn.neighbors.classification import KNeighborsClassifier

from sklearn.neighbors import NearestCentroid

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB  

from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# Import the needed packages

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve





# Instantiate the classifiers

classifiers = [

    ('Logistic Regression', LogisticRegression(solver='lbfgs', max_iter=1e3)),

    ('SVC', SVC(kernel='poly', degree=3, gamma='auto')), 

    ('SGD Hinge', SGDClassifier(loss='hinge', max_iter=200, tol=1e-3)), 

    ('SGD Logistic', SGDClassifier(loss='log', max_iter=200, tol=1e-3)), 

    ('SGD Smoothed Hinge', SGDClassifier(loss='log', max_iter=200, tol=1e-3)),

    ('Nearest Neighbors', KNeighborsClassifier(n_neighbors=3)), 

    ('Nearest Centroid', NearestCentroid()), 

    ('Decision Tree', DecisionTreeClassifier()), 

    ('Random Forest', RandomForestClassifier(n_estimators=100)),

    ('Extra Tree', ExtraTreeClassifier()),

    ('AdaBoost', AdaBoostClassifier()), 

    ('Gradient Boosting', GradientBoostingClassifier()), 

    ('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=1)),

    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()), 

    ('Gaussian NB', GaussianNB()), 

    ('Bernoulli NB', BernoulliNB()),

]

# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=5)



# empty lists

clf_names = []

cv_accuracies = []

cv_precisions = []

cv_recalls = []

cv_f1scores = []



# Modeling step Test differents algorithms 

for name, clf in classifiers:

    clf_names.append(name)

    cv_accuracies.append(cross_val_score(clf, X_train, y_train, scoring = 'accuracy', cv = kfold, n_jobs=4))

    cv_precisions.append(cross_val_score(clf, X_train, y_train, scoring = 'precision', cv = kfold, n_jobs=4))

    cv_recalls.append(cross_val_score(clf, X_train, y_train, scoring = 'recall', cv = kfold, n_jobs=4))

    cv_f1scores.append(cross_val_score(clf, X_train, y_train, scoring = 'f1', cv = kfold, n_jobs=4))

# Empty lists

accuracy_means = []

accuracy_stds = []



precision_means = []

precision_stds = []



recall_means = []

recall_stds = []



f1score_means = []

f1score_stds = []



# Calculate the mean of acuuracies, precisions, recalls and f1scores

for cv_accuracy, cv_precision, cv_recall, cv_f1score in zip(cv_accuracies, cv_precisions, cv_recalls, cv_f1scores):

    accuracy_means.append(cv_accuracy.mean())

    accuracy_stds.append(cv_accuracy.std())

    precision_means.append(cv_precision.mean())

    precision_stds.append(cv_precision.std())

    recall_means.append(cv_recall.mean())

    recall_stds.append(cv_recall.std())

    f1score_means.append(cv_f1score.mean())

    f1score_stds.append(cv_f1score.std())
df_accuracy = pd.DataFrame({"CrossValMeans":accuracy_means,"CrossValerrors": accuracy_stds, "Algorithm": clf_names})



g = sns.barplot("CrossValMeans","Algorithm",data = df_accuracy, palette="Set3",orient = "h",**{'xerr':accuracy_stds})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
df_precision = pd.DataFrame({"CrossValMeans":precision_means,"CrossValerrors": precision_stds, "Algorithm": clf_names})



g = sns.barplot("CrossValMeans","Algorithm",data = df_precision, palette="Set3",orient = "h",**{'xerr':precision_stds})

g.set_xlabel("Mean Precision")

g = g.set_title("CV precisions")
df_recall = pd.DataFrame({"CrossValMeans":recall_means,"CrossValerrors": recall_stds, "Algorithm": clf_names})



g = sns.barplot("CrossValMeans", "Algorithm", data = df_recall, palette="Set3",orient = "h",**{'xerr':recall_stds})

g.set_xlabel("Mean Recall")

g = g.set_title("CV recalls")
df_f1score = pd.DataFrame({"CrossValMeans": f1score_means,"CrossValerrors": f1score_stds, "Algorithm": clf_names})





g = sns.barplot("CrossValMeans","Algorithm",data = df_f1score, palette="Set3",orient = "h",**{'xerr':f1score_stds})

g.set_xlabel("Mean F1-score")

g = g.set_title("CV F1 Scores")