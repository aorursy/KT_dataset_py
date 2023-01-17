import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

%matplotlib inline

import scipy.stats                # statistics

from sklearn import preprocessing



import os

print(os.listdir("../input"))
# Read the data and store them in two objects



red = pd.read_csv('../input/winequality-red.csv')

white = pd.read_csv('../input/winequality-white.csv')
# Explore dimension of the datasets



print("There are {} red wines with {} attributes in red dataset. \n".format(red.shape[0],red.shape[1]))

print("There are {} red wines with {} attributes in white dataset. \n".format(white.shape[0],white.shape[1]))
# Let´s create a new variable that let us know if the wine is red(1) or white(0)

red['red']=1

white['red']=0



# Union of both datasets

wines = pd.concat([red,white])
wines.head()
wines.shape
# Let's see if we have duplicated records



twice = wines[wines.duplicated()]

twice.shape
twice.head()
sns.countplot(x="red", data=twice, palette="husl")
pd.DataFrame(twice['red'].value_counts(dropna=False)).head()
pd.DataFrame(wines['red'].value_counts(dropna=False)).head()
wine = wines.drop_duplicates(keep='first')

wine.shape
sns.countplot(x="red", data=wine, palette="RdPu")
wine.describe()
plt.figure(figsize=(15,10))

sns.boxplot(data=wine.drop(columns=['red']), orient='horizontal', palette='RdPu')
# Scaling the continuos variables

wine_scale = wine.copy()

scaler = preprocessing.StandardScaler()

columns = wine.columns[0:12]

wine_scale[columns] = scaler.fit_transform(wine_scale[columns])

wine_scale.head()
plt.figure(figsize=(15,10))

sns.boxplot(data=wine_scale.drop(columns=['red']), orient='horizontal', palette='RdPu')
g = sns.PairGrid(wine_scale.iloc[:,1:13], hue="red", palette="RdPu")

g.map(plt.scatter);
# Compute the correlation matrix

corr=wine_scale.iloc[:,1:13].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, annot=True, cmap='RdPu', center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.jointplot(x="total sulfur dioxide", y="free sulfur dioxide", data=wine, color='c')
sample = np.random.choice(wine_scale.index, size=int(len(wine_scale)*0.8), replace=False)

train_data, test_data = wine_scale.iloc[sample], wine_scale.drop(sample)



print("Number of training samples is", len(train_data))

print("Number of testing samples is", len(test_data))

print(train_data[:5])

print(test_data[:5])
f, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

sns.countplot(x="red", data=wine_scale, palette="RdPu", ax=axes[0])

sns.countplot(x="red", data=train_data, palette="RdPu", ax=axes[1])

sns.countplot(x="red", data=test_data, palette="RdPu", ax=axes[2])
features = train_data.drop('red', axis=1)

targets = train_data['red']

features_test = test_data.drop('red', axis=1)

targets_test = test_data['red']
from keras import models

from keras import layers



# Building the model

Nnetwork = models.Sequential()

Nnetwork.add(layers.Dense(40, activation='sigmoid', input_shape=(12,)))

Nnetwork.add(layers.Dense(1, activation='sigmoid'))
# Compiling the model

Nnetwork.compile(loss = 'binary_crossentropy',

                 optimizer='sgd',

                 metrics=['accuracy'])

Nnetwork.summary()
# Training the model

Nnetwork.fit(features, targets, epochs=10, batch_size=100, verbose=0)
test_loss, test_acc = Nnetwork.evaluate(features_test, targets_test)
print('test_acc:', test_acc, '\ntest_loss:', test_loss)
train_loss, train_acc = Nnetwork.evaluate(features, targets)
print('train_acc:', train_acc, '\ntrain_loss:', train_loss)
from sklearn.ensemble import RandomForestClassifier
Rforest = RandomForestClassifier(max_depth=4, n_estimators=10, max_features=2)
Rforest.fit(features, targets)
# Accuracy

score = Rforest.score(features_test, targets_test)
print(score)
# Accuracy

score_train = Rforest.score(features, targets)
print(score_train)
### Predictions

y_pred_rf = Rforest.predict(features_test)
### Probabilities

y_prob_rf = Rforest.predict_proba(features_test)

y_prob_rf = y_prob_rf.T[1]
from sklearn import metrics

# measure confusion matrix

cm_rf = metrics.confusion_matrix(targets_test, y_pred_rf, labels=[0, 1])

cm_rf = cm_rf.astype('float')

cm_rf_norm = cm_rf / cm_rf.sum(axis=1)[:, np.newaxis]

print("True Positive (rate): ", cm_rf[1,1], "({0:0.4f})".format(cm_rf_norm[1,1]))

print("True Negative (rate): ", cm_rf[0,0], "({0:0.4f})".format(cm_rf_norm[0,0]))

print("False Positive (rate):", cm_rf[1,0], "({0:0.4f})".format(cm_rf_norm[1,0]))

print("False Negative (rate):", cm_rf[0,1], "({0:0.4f})".format(cm_rf_norm[0,1]))
np.shape(y_prob_rf)
fpr, tpr, thresholds = metrics.roc_curve(targets_test, y_pred_rf)
# measure Area Under Curve (AUC)

auc_rf = metrics.roc_auc_score(targets_test, y_pred_rf)

print()

print("AUC:", auc_rf)
# ------------------------------------------------------------------------------

# Plot: Receiver-Operator Curve (ROC)

# ------------------------------------------------------------------------------



fig, axis1 = plt.subplots(figsize=(8,8))

plt.plot(fpr, tpr, 'r-', label='ROC')

plt.plot([0,1], [0,1], 'k--', label="1-1")

plt.title("Receiver Operator Characteristic (ROC)")

plt.xlabel("False positive (1 - Specificity)")

plt.ylabel("True positive (selectivity)")

plt.legend(loc='lower right')

plt.tight_layout()
# Accuracy of Random Forest

test_loss, test_acc_nn = Nnetwork.evaluate(features_test, targets_test)

print('Accuracy of NN in test data:', test_acc, '\ntest_loss:', test_loss)
# Accuracy of Random Forest

score = Rforest.score(features_test, targets_test)

print(score)
importances = Rforest.feature_importances_

std = np.std([tree.feature_importances_ for tree in Rforest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]
# Print the feature ranking

print("Feature ranking:")



for f in zip(features.columns, Rforest.feature_importances_):

    print(f)



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(features.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(features.shape[1]), indices)

plt.xlim([-1, features.shape[1]])

plt.show()