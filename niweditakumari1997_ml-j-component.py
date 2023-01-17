# Import all the libraries that we shall be using

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.cluster import KMeans

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



import xgboost as xgb



from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping
# Import labels (for the whole dataset, both training and testing)

y = pd.read_csv("../input/actual.csv")

print(y.shape)

y.head()
y['cancer'].value_counts()
# Recode label to numeric

y = y.replace({'ALL':0,'AML':1})

labels = ['ALL', 'AML'] # for plotting convenience later on
# Import training data

df_train = pd.read_csv('../input/data_set_ALL_AML_train.csv')

print(df_train.shape)



# Import testing data

df_test = pd.read_csv('../input/data_set_ALL_AML_independent.csv')

print(df_test.shape)
df_train.head()
df_test.head()


train_to_keep = [col for col in df_train.columns if "call" not in col]

test_to_keep = [col for col in df_test.columns if "call" not in col]



X_train_tr = df_train[train_to_keep]

X_test_tr = df_test[test_to_keep]
train_columns_titles = ['Gene Description', 'Gene Accession Number', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',

       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', 

       '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']



X_train_tr = X_train_tr.reindex(columns=train_columns_titles)
test_columns_titles = ['Gene Description', 'Gene Accession Number','39', '40', '41', '42', '43', '44', '45', '46',

       '47', '48', '49', '50', '51', '52', '53',  '54', '55', '56', '57', '58', '59',

       '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72']



X_test_tr = X_test_tr.reindex(columns=test_columns_titles)
X_train = X_train_tr.T

X_test = X_test_tr.T



print(X_train.shape) 

X_train.head()
# Clean up the column names for training and testing data

X_train.columns = X_train.iloc[1]

X_train = X_train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)



# Clean up the column names for Testing data

X_test.columns = X_test.iloc[1]

X_test = X_test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)



print(X_train.shape)

print(X_test.shape)

X_train.head()
# Split into train and test (we first need to reset the index as the indexes of two dataframes need to be the same before you combine them).



# Subset the first 38 patient's cancer types

X_train = X_train.reset_index(drop=True)

y_train = y[y.patient <= 38].reset_index(drop=True)



# Subset the rest for testing

X_test = X_test.reset_index(drop=True)

y_test = y[y.patient > 38].reset_index(drop=True)
X_train.describe()
# Convert from integer to float

X_train_fl = X_train.astype(float, 64)

X_test_fl = X_test.astype(float, 64)



# Apply the same scaling to both datasets

scaler = StandardScaler()

X_train_scl = scaler.fit_transform(X_train_fl)

X_test_scl = scaler.transform(X_test_fl) # note that we transform rather than fit_transform
pca = PCA()

pca.fit_transform(X_train)
total = sum(pca.explained_variance_)

k = 0

current_variance = 0

while current_variance/total < 0.90:

    current_variance += pca.explained_variance_[k]

    k = k + 1

    

print(k, " features explain around 90% of the variance. From 7129 features to ", k, ", not too bad.", sep='')



pca = PCA(n_components=k)

X_train.pca = pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



var_exp = pca.explained_variance_ratio_.cumsum()

var_exp = var_exp*100

plt.bar(range(k), var_exp);
pca3 = PCA(n_components=3).fit(X_train)

X_train_reduced = pca3.transform(X_train)



plt.clf()

fig = plt.figure(1, figsize=(10,6 ))

ax = Axes3D(fig, elev=-150, azim=110,)

ax.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], X_train_reduced[:, 2], c = y_train.iloc[:,1], cmap = plt.cm.Paired, linewidths=10)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])
fig = plt.figure(1, figsize = (10, 6))

plt.scatter(X_train_reduced[:, 0],  X_train_reduced[:, 1], c = y_train.iloc[:,1], cmap = plt.cm.Paired, linewidths=10)

plt.annotate('Note the Brown Cluster', xy = (30000,-2000))

plt.title("2D Transformation of the Above Graph ")
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train_scl)

km_pred = kmeans.predict(X_test_scl)



print('K-means accuracy:', round(accuracy_score(y_test.iloc[:,1], km_pred), 3))



cm_km = confusion_matrix(y_test.iloc[:,1], km_pred)



ax = plt.subplot()

sns.heatmap(cm_km, annot=True, ax = ax, fmt='g', cmap='Greens') 



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels') 

ax.set_title('K-means Confusion Matrix') 

ax.xaxis.set_ticklabels(labels) 

ax.yaxis.set_ticklabels(labels, rotation=360);
# Create a Gaussian classifier

nb_model = GaussianNB()



nb_model.fit(X_train, y_train.iloc[:,1])



nb_pred = nb_model.predict(X_test)



print('Naive Bayes accuracy:', round(accuracy_score(y_test.iloc[:,1], nb_pred), 3))



cm_nb =  confusion_matrix(y_test.iloc[:,1], nb_pred)



ax = plt.subplot()

sns.heatmap(cm_nb, annot=True, ax = ax, fmt='g', cmap='Greens') 



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels') 

ax.set_title('Naive Bayes Confusion Matrix') 

ax.xaxis.set_ticklabels(labels) 

ax.yaxis.set_ticklabels(labels, rotation=360);
# Parameter grid

svm_param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10], "kernel": ["linear", "rbf", "poly"], "decision_function_shape" : ["ovo", "ovr"]} 



# Create SVM grid search classifier

svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=3)



# Train the classifier

svm_grid.fit(X_train_pca, y_train.iloc[:,1])



print("Best Parameters:\n", svm_grid.best_params_)



# Select best svc

best_svc = svm_grid.best_estimator_



# Make predictions using the optimised parameters

svm_pred = best_svc.predict(X_test_pca)



print('SVM accuracy:', round(accuracy_score(y_test.iloc[:,1], svm_pred), 3))



cm_svm =  confusion_matrix(y_test.iloc[:,1], svm_pred)



ax = plt.subplot()

sns.heatmap(cm_svm, annot=True, ax = ax, fmt='g', cmap='Greens') 



# Labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels') 

ax.set_title('SVM Confusion Matrix') 

ax.xaxis.set_ticklabels(labels) 

ax.yaxis.set_ticklabels(labels, rotation=360);
xgb2_model = xgb.XGBClassifier()

xgb2_model.fit(X_train_pca, y_train.iloc[:,1])



xgb2_pred = xgb2_model.predict(X_test_pca)



print('Accuracy: ', round(accuracy_score(y_test.iloc[:,1], xgb2_pred), 3))



cm_xgb2 = confusion_matrix(y_test.iloc[:,1], xgb2_pred)



ax = plt.subplot()

sns.heatmap(cm_xgb2, annot=True, ax = ax, fmt='g', cmap='Greens') 



# Labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels') 

ax.set_title('XGB (PCA without Grid Search) Confusion Matrix') 

ax.xaxis.set_ticklabels(labels) 

ax.yaxis.set_ticklabels(labels, rotation=360);