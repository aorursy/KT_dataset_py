%%capture

!pip install umap-learn
import numpy as np

import pandas as pd

import umap

from sklearn import preprocessing, metrics, model_selection, neural_network, linear_model, ensemble

import matplotlib.pyplot as plt

from scipy import stats
# Load files and define features



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

features = list(train_df.columns)

features.remove('id')

features.remove('genus')

features.remove('species')
# First we check if the dataset is as balanced as advertised



fig, ax = plt.subplots(figsize=(20,10))

train_df['species'].value_counts().plot.bar(ax=ax)

plt.show()
# From this graph it might seem that the OP was lying, but we have to be careful!

# A Fully qualified species name is Genus + Species!



train_df['FQSN'] = train_df['genus'] + '_' + train_df['species']

test_df['FQSN'] = test_df['genus'] + '_' + test_df['species']



# Now we can check again the truthfulness of the dataset description



fig, ax = plt.subplots(figsize=(20,10))

train_df['FQSN'].value_counts().plot.bar(ax=ax)

plt.show()
# Perfectly balanced... you know the drill

# Now we can check the distribution of the genera



fig, ax = plt.subplots(figsize=(20,10))

(train_df['genus'].value_counts()/20).plot.bar(ax=ax)

plt.show()
# Scale dataset and encode the label



scaler = preprocessing.StandardScaler()

scaler.fit(train_df[features])

train_df[features] = pd.DataFrame(scaler.transform(train_df[features]), columns=features)

test_df[features] = pd.DataFrame(scaler.transform(test_df[features]), columns=features)



X = train_df[features].values

y_raw = train_df['FQSN'].values



le = preprocessing.LabelEncoder()



y = le.fit_transform(y_raw)



X_test = test_df[features].values

y_raw_test = test_df['FQSN'].values



y_test = le.transform(y_raw_test)
# Use umap (TSNE's true heir) to show the dataset in a space of reduced dimensions 



X_umap = umap.UMAP(n_neighbors=125, min_dist=0.05).fit_transform(X)



fig, ax = plt.subplots(figsize=(20,10))



plt.scatter(X_umap[:,0], X_umap[:,1])

plt.show()
# As we can see there are very noticeable clusters let's see if some model can learn to distinguish between the species



sss = model_selection.StratifiedShuffleSplit(n_splits=5,test_size=0.2)



log_reg_accs = []

log_reg_preds = np.zeros((5,len(X_test)))



rand_forest_accs = []

rand_forest_preds = np.zeros((5,len(X_test)))



nn_accs = []

nn_preds = np.zeros((5,len(X_test)))



i = 0



for train_index, val_index in sss.split(X, y):

    X_train, X_val = X[train_index], X[val_index]

    y_train, y_val = y[train_index], y[val_index]

    

    log_reg = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')

    log_reg.fit(X_train, y_train)

    y_val_pred = log_reg.predict(X_val)

    log_reg_accs.append(metrics.accuracy_score(y_val_pred, y_val))

    log_reg_preds[i] = log_reg.predict(X_test)

    

    rand_forest = ensemble.RandomForestClassifier(n_estimators=100)

    rand_forest.fit(X_train, y_train)

    y_val_pred = rand_forest.predict(X_val)

    rand_forest_accs.append(metrics.accuracy_score(y_val_pred, y_val))

    rand_forest_preds[i] = rand_forest.predict(X_test)

    

    

    nn = neural_network.MLPClassifier(activation='relu',hidden_layer_sizes=[128,128])

    nn.fit(X_train, y_train)

    y_val_pred = nn.predict(X_val)

    nn_accs.append(metrics.accuracy_score(y_val_pred, y_val))

    nn_preds[i] = nn.predict(X_test)

    

    i += 1

    

print("Accuracy of Logistic Regression on validation set: ", np.mean(log_reg_accs))

print("Accuracy of Random Forest on validation set: ", np.mean(rand_forest_accs))

print("Accuracy of Feed Forward Neural Network on validation set: ", np.mean(nn_accs))
log_reg_pred = stats.mode(log_reg_preds, axis=0)[0][0]

rand_forest_pred = stats.mode(rand_forest_preds, axis=0)[0][0]

nn_pred = stats.mode(nn_preds, axis=0)[0][0]



print("Accuracy of Logistic Regression on test set: ", metrics.accuracy_score(log_reg_pred, y_test))

print("Accuracy of Random Forest on test set: ", metrics.accuracy_score(rand_forest_pred, y_test))

print("Accuracy of Feed Forward Neural Network on test set: ", metrics.accuracy_score(nn_pred, y_test))