import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn.preprocessing import power_transform

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/pulsar_stars.csv")



# Print the head of df

print(df.head())



# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)
plt.figure(figsize=(12,8))

sns.heatmap(df.describe()[1:].transpose(),

            annot=True,linecolor="w",

            linewidth=2,cmap=sns.color_palette("Blues"))

plt.title("Data summary")

plt.show()
# Display the histogram to undestand the data

f, axes = plt.subplots(2,4, figsize=(20, 12))

sns.distplot( df[" Mean of the integrated profile"], ax=axes[0,0])

sns.distplot( df[" Standard deviation of the integrated profile"], ax=axes[0,1])

sns.distplot( df[" Excess kurtosis of the integrated profile"], ax=axes[0,2])

sns.distplot( df[" Skewness of the integrated profile"], ax=axes[0,3])

sns.distplot( df[" Mean of the DM-SNR curve"], ax=axes[1,0])

sns.distplot( df[" Standard deviation of the DM-SNR curve"], ax=axes[1,1])

sns.distplot( df[" Excess kurtosis of the DM-SNR curve"], ax=axes[1,2])

sns.distplot( df[" Skewness of the DM-SNR curve"], ax=axes[1,3])
# Compute the correlation matrix

corr=df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.color_palette("Blues")



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(df,hue="target_class")

plt.title("pair plot for variables")

plt.show()
df_scale = df.copy()

columns =df.columns[:-1]

df_scale[columns] = power_transform(df.iloc[:,0:8],method='yeo-johnson')

df_scale.head()
# Display the histogram to undestand the data

f, axes = plt.subplots(2,4, figsize=(20, 12))

sns.distplot( df_scale[" Mean of the integrated profile"], ax=axes[0,0])

sns.distplot( df_scale[" Standard deviation of the integrated profile"], ax=axes[0,1])

sns.distplot( df_scale[" Excess kurtosis of the integrated profile"], ax=axes[0,2])

sns.distplot( df_scale[" Skewness of the integrated profile"], ax=axes[0,3])

sns.distplot( df_scale[" Mean of the DM-SNR curve"], ax=axes[1,0])

sns.distplot( df_scale[" Standard deviation of the DM-SNR curve"], ax=axes[1,1])

sns.distplot( df_scale[" Excess kurtosis of the DM-SNR curve"], ax=axes[1,2])

sns.distplot( df_scale[" Skewness of the DM-SNR curve"], ax=axes[1,3])
# Create feature and target arrays

X = df_scale.iloc[:,0:8]

y = df_scale.iloc[:,-1]



# Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123, stratify=y)

print('X_train:', X_train.shape)

print('X_test:', X_test.shape)
# Seleccion of the model to run

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
# Prepare models

models = []

models.append(('LR', LogisticRegression(solver='lbfgs')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='scale')))

models
# Evaluate each model by Accuracy

results = []

names = []

seed = 123

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=5, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
plt.figure(figsize=(12,8))

sns.boxplot(x=names, y=results, palette="Set3")

plt.title("Models Accuracy")

plt.show()
# Import necessary modules

from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [3, None],

              "max_features": randint(1, 9),

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}



# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeClassifier()



# Instantiate the Grid Search

tree_cv = RandomizedSearchCV(tree, param_dist, cv=5, scoring=scoring)



# Fit it to the data

tree_cv.fit(X_train, y_train)



# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))

print("Best score is {}".format(tree_cv.best_score_))
from sklearn.model_selection import cross_val_score



# Fit it to the data with new hyper-parameters

new_tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, 

                                  max_features = 7, min_samples_leaf = 3)

new_cv = cross_val_score(new_tree, X_train, y_train, cv=5, scoring=scoring)
# Merging the results with the old group of model to compare results

new_results = list(np.vstack((results, new_cv)))

names.append('CART_T')
plt.figure(figsize=(12,8))

sns.boxplot(x=names, y=new_results, palette="Set3")

plt.title("Models Accuracy")

plt.show()
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Instantiate the model

lg = LogisticRegression(solver='lbfgs')



# Fit the classifier to the training data

lg.fit(X_train, y_train)



# Predict the labels of the test data: y_pred

y_pred = lg.predict(X_test)



# Generate the confusion matrix and classification report

print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred),cbar=False,annot=True,cmap=cmap,fmt="d")
from sklearn.metrics import roc_curve



# Compute predicted probabilities: y_pred_prob

y_pred_prob = lg.predict_proba(X_test)[:,1]



# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
# Import necessary modules

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score



# Compute predicted probabilities: y_pred_prob

y_pred_prob = lg.predict_proba(X_test)[:,1]



# Compute and print AUC score

print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))



# Compute cross-validated AUC scores: cv_auc

cv_auc = cross_val_score(lg, X, y, cv=5, scoring='roc_auc')



# Print list of AUC scores

print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))