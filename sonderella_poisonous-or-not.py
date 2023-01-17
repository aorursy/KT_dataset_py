# Load dataset

import pandas as pd



data = pd.read_csv("../input/mushrooms.csv")

data.head()
# encode labels



from sklearn import preprocessing



le = preprocessing.LabelEncoder()

for col in data.columns:

    data[col] = le.fit_transform(data[col])

data.head()
# split out features and target labels



y = data['class']

X = data.drop(['class'], axis=1)
# split out training and test set

from sklearn.model_selection import train_test_split



# Shuffle and split the data into training and testing subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# apply MLP model

from sklearn.neural_network import MLPClassifier



# fit model

clf = MLPClassifier(hidden_layer_sizes=(100,), solver='adam',warm_start=False, random_state=None)

clf.fit(X_train, y_train)



# assess model accuracy

clf.score(X_test, y_test)
# create confusion matrix to illustrate accuracy of predictions



from sklearn.metrics import classification_report,confusion_matrix



y_pred = clf.predict(X_train)

cm = (confusion_matrix(y_train,y_pred))



import seaborn as sns

%matplotlib inline



sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Edible", "Poisonous"],yticklabels=["Edible","Poisonous"])
print(classification_report(y_train,y_pred))
# fit RandomForest model

from sklearn.ensemble import RandomForestClassifier 



tree = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)



tree.fit(X_train, y_train)



# assess model accuracy

tree.score(X_test, y_test)
# extract feature importances

import numpy as np



keyfeat = tree.feature_importances_
# rank features



df = pd.DataFrame(keyfeat)

df.index = np.arange(1, len(df) + 1)



featurenames = data.columns

featurenames = pd.DataFrame(data.columns)

featurenames.drop(featurenames.head(1).index, inplace=True)



dfnew = pd.concat([featurenames, df], axis=1)

dfnew.columns = ['featurenames', 'weight']

dfsorted = dfnew.sort_values(['weight'], ascending=[False])

dfsorted.head()
# plot feature importances

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(15, 10))



ax = sns.barplot(x=dfsorted['featurenames'], y=dfsorted['weight'])



ax.set(xlabel='feature names', ylabel='weight')



ax.set_title('Feature importances')



for item in ax.get_xticklabels():

    item.set_rotation(50)