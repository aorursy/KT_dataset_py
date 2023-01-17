# Import packages and libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

import lightgbm as lgb

import keras
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout
# Import the data with pandas

data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
data_copy = data.copy() # Just in case

data.head()
# Dataset columns

print("The names of the columns are:", data.columns)
# Dataset statistical description

data.describe
# Missing values

total = data.isnull().sum().sort_values(ascending=False)
porcentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, porcentage], axis=1, keys=['Total', 'Porcentage'])
missing_data.head(20)
# Removing non-relevant information

not_featured_cols = ["RowNumber", "CustomerId", "Surname"]
data = data.drop(not_featured_cols, axis = 1)

data.head()
# Correlation plot

corr = data.corr()

sns.set()
fig, ax = plt.subplots(figsize = (15,15))
ax = sns.heatmap(corr, annot = True, linewidths = 1.0)
ax.set_title("Correlation Plot")
# Visualizing columns

fig = sns.countplot(data["Geography"])
fig.set_title("Geopgraphy Counting")

plt.show()
fig = sns.countplot(data["Gender"])
fig.set_title("Gender Counting")

plt.show()
# Display min and max age.

print("The maximum age is:", data["Age"].max())
print("The minimum age is:", data["Age"].min())
# Split the data

X = data.iloc[:, :10].values
y = data.iloc[:, 10].values
# Encoding categorical features

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # 'Geography' 
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # 'Gender'



transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling", # Name for transormation
        OneHotEncoder(categories='auto'), # Class we want transform
        [1] # Columns
        )
    ], remainder='passthrough'
)
X = transformer.fit_transform(X)
X = X[:, 1:] # Avoiding multicollinearity
# Last but not least, splitting the data in training and testing groups

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)
# Scaling data

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Building the model

classifier = Sequential()

# First layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
classifier.add(Dropout(rate = 0.1))

# Second layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.1))

# Output layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

# Compiler
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
# LET'S TRAIN!

classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)
# Evaluating the model

y_pred = classifier.predict(X_test) 
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)

def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    sns.set()
    ax= plt.subplot()
    sns.heatmap(df_confusion, annot=True, ax = ax, cmap='coolwarm')
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
plot_confusion_matrix(cm_df)
# Building the model specifically for LGBM

training_data = lgb.Dataset(data = X_train, label = y_train)
params = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
params['metric'] = ['auc', 'binary_logloss']
classifier = lgb.train(params = params,
                       train_set = training_data,
                       num_boost_round = 10)
# Making predictions with test set

prob_pred = classifier.predict(X_test)
y_pred = np.zeros(len(prob_pred))
for i in range(0, len(prob_pred)):
    if prob_pred[i] >= 0.5:
       y_pred[i] = 1
    else:  
       y_pred[i] = 0
# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)
plot_confusion_matrix(cm_df)
# Getting the accuracy

accuracy = accuracy_score(y_pred, y_test) * 100
print("Accuracy: {:.0f} %".format(accuracy))
# K-FOLD CROSS VALIDATION

params = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
params['metric'] = ['auc']
cv_results = lgb.cv(params = params,
                    train_set = training_data,
                    num_boost_round = 10,
                    nfold = 10)
average_auc = np.mean(cv_results['auc-mean'])
print("Average AUC: {:.0f} %".format(accuracy))
