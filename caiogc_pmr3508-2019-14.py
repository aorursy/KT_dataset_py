import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 
# Read training data

train_df = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

                       sep=r'\s*,\s*',

                       engine='python',

                       na_values="?")

train_df.shape
# Show the first inputs of the dataframe

train_df.head()
# Remove missing data from dataset

train_df = train_df.dropna()

train_df.shape
# Do the same for testing data

test_df = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

                       sep=r'\s*,\s*',

                       engine='python',

                       na_values="?")

test_df = test_df.dropna()

test_df.shape



test_df.head()
# Show general description of the dataset numerical features 

train_df.describe()
train_df["native.country"].value_counts()
train_df["workclass"].value_counts().plot(kind="bar")
train_df["education"].value_counts().plot(kind="bar")
train_df["sex"].value_counts().plot(kind="bar")
train_df["marital.status"].value_counts().plot(kind="bar")
# Function to plot the feature distribution with respect to the labels

def plot_class_hist(df, label_key, feature_key):

    labels = df[label_key].unique()

    temp = []

    n_labels = len(labels)

    for i in range(n_labels):

        temp.append(df[df[label_key] == labels[i]])

        temp[i] = np.array(temp[i][feature_key]).reshape(-1,1)

    fig = plt.figure(figsize= (15,8))

    for i in range(n_labels):

        plt.hist(temp[i], alpha = 0.7)

    plt.legend(labels)
plot_class_hist(train_df, 'income', 'sex')
plot_class_hist(train_df, 'income', 'marital.status')
plot_class_hist(train_df, 'income', 'workclass')
from sklearn import preprocessing
train_df_num = train_df.apply(preprocessing.LabelEncoder().fit_transform)

train_df_num.head()
test_df_num = test_df.apply(preprocessing.LabelEncoder().fit_transform)

test_df_num.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
# Define inputs and targets

X_train = train_df[['age','education.num','capital.gain','capital.loss','hours.per.week']]

y_train = train_df['income']
from tqdm import tqdm



def find_k_value(X_train, y_train, k_max = 30, n_folds = 10):

    '''

    Find best K-value using n-fold CV

    '''

    best_k = 1

    best_score = 0

    score_list = []

    for k in tqdm(range(1,k_max+1)):

        clf = KNeighborsClassifier(n_neighbors=k)

        scores = cross_val_score(clf, X_train, y_train, cv = n_folds)

        score_list.append(scores.mean())



        if scores.mean() > best_score:

            best_score = scores.mean()

            best_k = k

    return best_k, best_score, score_list
best_k, best_score, score_list = find_k_value(X_train, y_train)

print("Best value of k = ", best_k)

print("Best score = ", best_score)

plt.plot(score_list)

plt.show()
from sklearn.preprocessing import StandardScaler
X_train = train_df_num

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

y_train = train_df_num['income']
best_k, best_score, score_list = find_k_value(X_train, y_train)
print("Best value of k = ", best_k)

print("Best score = ", best_score)

plt.plot(score_list)

plt.show()
X_train = train_df[['age','education.num','capital.gain','capital.loss','hours.per.week']]

y_train = train_df['income']

clf = KNeighborsClassifier(n_neighbors = 26)

clf.fit(X_train, y_train)
X_test = test_df[['age','education.num','capital.gain','capital.loss','hours.per.week']]

y_pred = clf.predict(X_test)
pred_df = pd.DataFrame(y_pred, columns=['Income'])

pred_df
pred_df.to_csv("subimission.csv", index = True, index_label = 'Id')
