# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix

# read csv
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
# shows percentages of each class value in the dataset
n_classes = df['Class'].value_counts()
total = df['Class'].count()
print(f'There are {total} values, from which {n_classes[0]} are not fraud and {n_classes[1]} are fraud.\n')
print(f'No fraud: {round(n_classes[0]*100/total,1)} %\nFraud:    {round(n_classes[1]*100/total,1)} %')
# selects the fraudulent observations
fraud = df.loc[df['Class']==1]

# takes a random sample from the non fraudulent observations, equal to the size of the fraudulent observations
no_fraud = df.loc[df['Class']==0].sample(n=fraud.shape[0], random_state=0)

df_balanced = fraud.append(no_fraud)
print('Now we have a new dataframe with the same number of fraudulent and non fraudulent observations.\n')

# shuffle the new dataframe
df_balanced = df_balanced.sample(frac=1, random_state=0)
df_balanced.reset_index(drop=True, inplace=True)

print(df_balanced['Class'].value_counts())
X = df_balanced.drop(columns='Class')
y = df_balanced['Class']
# save feature column names
feature_cols = X.columns

# standarize feature matrix
scaler = preprocessing.StandardScaler()
X_std = scaler.fit_transform(X)

# goes back from np array to dataframe
X_std = pd.DataFrame(X_std, columns=feature_cols)
# visualize correlations
corr = X_std.join(y).corr()
fig, hm = plt.subplots(figsize=(10,7))    
hm = sns.heatmap(corr, cmap='coolwarm_r')
hm.set_title('Correlations')
used_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']

# drops columns in both standarized and not standarized feature matrixes
X = X[used_columns]
X_std = X_std[used_columns]
# Create a KNN classifier
knn = KNeighborsClassifier(n_jobs=-1)

# Create standardizer
scaler = preprocessing.StandardScaler()

# Create a pipeline
pipeline = Pipeline([("scaler", scaler), ("knn", knn)])

# Create space of candidate values
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7]}]


# Create grid search
classifier = GridSearchCV(pipeline, search_space, cv=5, verbose=0)

# Fit grid search
model_knn = classifier.fit(X, y)
# drop unused columns of original dataframe, separate in X and and test model with original dataframe
df_X = df.drop(columns='Class')
df_X = df_X[used_columns]
df_y = df['Class']

y_predicted = model_knn.predict(df_X)
# Best neighborhood size (k)
K = model_knn.best_estimator_.get_params()["knn__n_neighbors"]
print(f'The best value of K is: {K}')
from sklearn.metrics import accuracy_score
print(f'Accuracy: {round(accuracy_score(df_y,y_predicted)*100,2)} %')
from sklearn.metrics import confusion_matrix

# Create confusion matrix
matrix = confusion_matrix(df_y, y_predicted)

# represents confusion matrix with heatmap
dataframe = pd.DataFrame(matrix, index=['No Fraud', 'Fraud'], columns=['No Fraud', 'Fraud'])
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='g', linewidths=1, linecolor='black')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
# supress annoying warning message
import warnings
warnings.filterwarnings("ignore")

# select the used columns from the features dataframe
df_X = df_X[used_columns]

sm = SMOTE(random_state=1)
X_res, y_res = sm.fit_resample(df_X,df_y)

X_res = pd.DataFrame(X_res, columns=df_X.columns)
y_res = pd.DataFrame(y_res, columns=['Class'])
print(y_res['Class'].value_counts())
# create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)

# create standardizer
scaler = preprocessing.StandardScaler()

# create a pipeline
pipeline = Pipeline([("scaler", scaler), ("knn", knn)])

# cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

cv_results = cross_val_score(pipeline, X_res, y_res, cv=kf, scoring="accuracy", n_jobs=-1)
# Calculate mean
print(f'Accuracy: {round(cv_results.mean()*100,2)} %')
model_smote = pipeline.fit(X_res,y_res)
y_predicted = model_smote.predict(df_X)
# create confusion matrix
matrix = confusion_matrix(df_y, y_predicted)

# represents confusion matrix with heatmap
dataframe = pd.DataFrame(matrix, index=['No Fraud', 'Fraud'], columns=['No Fraud', 'Fraud'])
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='g', linewidths=1, linecolor='black')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
from keras import models
from keras import layers

# number of features
n_inputs = X_res.shape[1]

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,random_state=0, test_size=0.2)

# standarize
scaler = preprocessing.StandardScaler()

# transform
scaler.fit(X_train)
X_std = scaler.transform(X_train)
X_test = scaler.transform(X_test)

NNmodel = models.Sequential()

NNmodel.add(layers.Dense(n_inputs, activation="relu", input_shape=(n_inputs, )))
NNmodel.add(layers.Dense(units=30, activation="relu"))
NNmodel.add(layers.Dense(units=1, activation="sigmoid"))

NNmodel.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = NNmodel.fit(X_std, y_train, epochs=7, validation_data=(X_test, y_test))
# get training and test loss histories
training_loss = history.history["loss"]
test_loss = history.history["val_loss"]

# create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# visualize loss history
plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
# get training and test accuracy histories
training_accuracy = history.history["accuracy"]
test_accuracy = history.history["val_accuracy"]
plt.plot(epoch_count, training_accuracy, "r--")
plt.plot(epoch_count, test_accuracy, "b-")

# visualize accuracy history
plt.legend(["Training Accuracy", "Test Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.show();
# make predictions
NN_pred = NNmodel.predict(scaler.transform(df_X))
# create confusion matrix
matrix = confusion_matrix(df_y, NN_pred.astype(int))

# represents confusion matrix with heatmap
dataframe = pd.DataFrame(matrix, index=['No Fraud', 'Fraud'], columns=['No Fraud', 'Fraud'])
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='g', linewidths=1, linecolor='black')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")