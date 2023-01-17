import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns 

warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_path = "/kaggle/input/titanic/train.csv"
train_df = pd.read_csv(train_path)

test_path = "/kaggle/input/titanic/test.csv"
test_df = pd.read_csv(test_path)
test_passenger_id = test_df.PassengerId.values

train_df.head()
train_df.describe()
print(train_df.dtypes)
print(train_df.shape)
for col in train_df.columns:
    print(col, " No. of categories: ", len(train_df[col].value_counts()))
train_df.isna().sum()
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
plt.show()
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.ylim(0,1)
plt.show()
# analyze influence of gender
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male')
plt.show()
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
datasets = [train_df, test_df]
labels = train_df[['Survived']]
train_df.drop('Survived', axis=1, inplace=True)
for i in range(len(datasets)):
    data = datasets[i]

    # select columns 
    data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    # encode categorical variables 
    Sex = pd.get_dummies(data.Sex)
    Pclass = pd.get_dummies(data.Pclass, prefix="class")
    Embarked = pd.get_dummies(data.Embarked)

    data = data.merge(Sex, how="left", left_index=True, right_index=True).merge(Pclass, how="left", left_index=True, right_index=True).merge(Embarked, how="left", left_index=True, right_index=True)
    data.drop(['Sex', 'Pclass', 'Embarked'], axis=1, inplace=True)
        
    # replace NAN values 
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    
    # datatypes 
    data = data.astype({'Fare': 'int32'}) 
    data = data.astype({'Age': 'int32'}) 
    
    feature_cols = ["Age","SibSp","Parch","Fare","female","male","class_1","class_2","class_3","C","Q","S"]
    
    # assign preprocessed dataset to train/test set
    if i == 0:
        train_df = data
        scaler.fit(train_df[feature_cols])
        x_train = scaler.transform(train_df[feature_cols])
        y_train = labels.values
    else: 
        test_df = data
        x_test = scaler.transform(test_df[feature_cols])
    
print("train and test set successfully pre-processed")
results=[]

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
            "kernel":["linear", "rbf", "poly"],
            "C":[4,5,6],
            "gamma": [0.09, 0.1, 0.11]
            }

cv = GridSearchCV(estimator=SVC(), param_grid=param_grid, n_jobs=-1, cv =5)
grid_result = cv.fit(x_train, y_train)

# summary
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with %r" % (mean, stdev, param))

results.append(grid_result.best_score_)
from sklearn.neighbors import KNeighborsClassifier

param_grid = {"n_neighbors": [2,3,4,5,6,7,8,9,10]}

cv = GridSearchCV(estimator = KNeighborsClassifier(), cv=5, param_grid=param_grid, n_jobs=-1)
grid_result = cv.fit(x_train, y_train)

# summary
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with %r" % (mean, stdev, param))

results.append(grid_result.best_score_)
from sklearn.ensemble import RandomForestClassifier

param_grid = {
            "n_estimators":[80, 90, 100],
            "max_depth":[10,12],
            "min_samples_split":[2,3],
            "max_features":[0.6, 0.7]
            }

cv = GridSearchCV(estimator = RandomForestClassifier(), cv=5, param_grid=param_grid, n_jobs=-1)
grid_result = cv.fit(x_train, y_train)

# summary
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with %r" % (mean, stdev, param))
    
results.append(grid_result.best_score_)
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
            "n_estimators":[75,80,85],
            "learning_rate":[0.005, 0.01, 0.015,],
            "max_depth":[6,7],
            "min_samples_split":[4,5],
            "max_features":[0.4, 0.5]
            }

cv = GridSearchCV(estimator = GradientBoostingClassifier(), cv=5, param_grid=param_grid, n_jobs=-1)
grid_result = cv.fit(x_train, y_train)

# summary
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with %r" % (mean, stdev, param))

results.append(grid_result.best_score_)
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras
import keras.backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# define input dim
input_dim = x_train.shape[1]
output_dim = 1

from keras.wrappers.scikit_learn import KerasClassifier


# Function to create model, required for KerasRegressor
def create_model(activation, neurons, dropout, learning_rate):
    
    adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(neurons))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim))
    model.add(Activation("sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])    
    return model

# create model
grid_model = KerasClassifier(build_fn=create_model, epochs=20, verbose=1)

# define the grid search parameters
neurons = [8, 16]
dropout= [0.1, 0.2, 0.3]
activation = ['tanh', 'relu']
learning_rate = [0.001, 0.0001]

# define grid with parameters to be tuned
param_grid = dict(neurons=neurons,activation=activation, dropout=dropout, learning_rate=learning_rate)

# instanciate GridSearchCV with defined scoring and cv
grid = GridSearchCV(estimator=grid_model, 
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    cv=3)
#fit grid model
grid_result = grid.fit(x_train, y_train)
    
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with %r" % (mean, stdev, param))
# define input dim
input_dim = x_train.shape[1]
output_dim = 1

adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# create model
model = Sequential()
model.add(Dense(16, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(output_dim))
model.add(Activation("sigmoid"))

# Compile model
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', precision_m, recall_m, f1_m])
print(model.summary())
# fit model
model_history = model.fit(x_train, y_train, 
                            epochs=200, 
                            validation_split=0.2,
                            shuffle=True,
                            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')])

print(model_history.history.keys())
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

axs[0].plot(model_history.history['accuracy'], color="b")
axs[0].plot(model_history.history['val_accuracy'], color="black")
axs[0].set_xlabel('\nEpoche', fontsize=14)
axs[0].set_title('ACC\n', fontsize=14)
axs[0].legend(['Training', 'Validation'], loc='best')

axs[1].plot(model_history.history['precision_m'], color="b")
axs[1].plot(model_history.history['val_precision_m'], color="black")
axs[1].set_xlabel('\nEpoche', fontsize=14)
axs[1].set_title('Precision\n', fontsize=14)
axs[1].legend(['Training', 'Validation'], loc='best')

axs[2].plot(model_history.history['recall_m'], color="b")
axs[2].plot(model_history.history['val_recall_m'], color="black")
axs[2].set_xlabel('\nEpoche', fontsize=14)
axs[2].set_title('Recall\n', fontsize=14)
axs[2].legend(['Training', 'Validation'], loc='best')
axs[2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

axs[3].plot(model_history.history['f1_m'], color="b")
axs[3].plot(model_history.history['val_f1_m'], color="black")
axs[3].set_xlabel('\nEpoche', fontsize=14)
axs[3].set_title('\nF1 Score\n', fontsize=14)
axs[3].legend(['Training', 'Validation'], loc='best')
axs[3].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.show()
nb_epochs = len(model_history.history['val_accuracy'])
val_acc = model_history.history['val_accuracy']
final_acc = val_acc[nb_epochs-1]

results.append(final_acc)
result_df = pd.DataFrame(data=results, columns=["Accuracy"])
result_df.index=["SVC", "KNN", "Random Forest", "Gradient Boosting", "ANN"]
result_df
plt.figure(figsize=(8,5))
plt.bar(result_df.index, result_df.Accuracy, color="black")
plt.ylabel("ACC\n")
plt.xticks(rotation=30)
plt.ylim(0.7,1)
plt.show()
from sklearn.metrics import confusion_matrix

# predictions
y_pred = model.predict_classes(x_train)

cm = confusion_matrix(y_train, y_pred)
print("Confusion Matrix Training Data: \n\n", cm)
# make predictions on test data
test_predictions = model.predict_classes(x_test)

df1 = pd.DataFrame(test_passenger_id, index=None, columns=["PassengerId"])
df2 = pd.DataFrame(data=test_predictions, columns=["Survived"])
final_data = df1.merge(df2, how="left", left_index=True, right_index=True)
final_data.to_csv('test.csv', header=True, index=False)
final_data