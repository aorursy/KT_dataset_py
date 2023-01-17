import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('../input/human-activity-recognition-with-smartphones/train.csv')
test_data = pd.read_csv('../input/human-activity-recognition-with-smartphones/test.csv')
train_data.shape
train_data.head()
train_data.isna().sum().sum()
train_data.Activity.value_counts()
import seaborn as sns

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (15, 5))

sns.countplot(x = 'Activity', 

              data = train_data, 

              #palette = "Blues_r",

              palette = 'winter',

              order = train_data['Activity'].value_counts().index

             )
# Plots distribution of 6 columns



def plot_distribution(data, col):

    fig, axes = plt.subplots(ncols = 3, nrows = 2, figsize = (15, 8))

    for i, ax in zip(range(6), axes.flat):

        sns.distplot(data[cols[i]], ax = ax)

    plt.show()
# Select some body acceleration attributes

cols = train_data.columns[:6]

plot_distribution(train_data, cols)
# Select some gravitational acceleration attributes

cols = train_data.columns[40:47]

plot_distribution(train_data, cols)
from sklearn.manifold import TSNE
tsne_data = train_data.copy()

tsne_data.drop(['Activity', 'subject'], axis = 1, inplace = True)
labels = train_data['Activity']

label_counts = labels.value_counts()
from sklearn.preprocessing import StandardScaler
def scale_data(data):

    scl = StandardScaler()

    return scl, scl.fit_transform(data)
scale_model, scaled_data = scale_data(tsne_data)

scaled_data.shape
tsne = TSNE(random_state = 0)

tsne_transformed = tsne.fit_transform(scaled_data)
fig1 = plt.figure(figsize = (25, 10))

colors = ['darkblue', 'mediumturquoise', 'darkgray', 'darkorchid', 'darkred', 'darkgreen']

for i, activity in enumerate(label_counts.index):

    mask = (labels == activity).values

    plt.scatter(x = tsne_transformed[mask][:,0],

                y = tsne_transformed[mask][:,1],

                color = colors[i],

                alpha = 0.4,

                label = activity)

plt.title('Visualisation using t-SNE')

plt.legend()

plt.show()
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import numpy as np
X_train = train_data.drop(['Activity', 'subject'], axis = 1)

y_train = train_data['Activity']



X_test = test_data.drop(['Activity', 'subject'], axis = 1)

y_test = test_data['Activity']
from sklearn.tree import DecisionTreeClassifier



clf_dt = DecisionTreeClassifier(max_depth = 30)
clf_dt.fit(X_train, y_train)
y_train_pred_dt = clf_dt.predict(X_train)
y_test_pred_dt = clf_dt.predict(X_test)
def plot_train_test_accuracy(y_train, y_train_pred, y_test, y_test_pred, title):

    acc_train = accuracy_score(y_train, y_train_pred)

    acc_test = accuracy_score(y_test, y_test_pred)

    

    print('Train accuracy = ', acc_train)

    print('Test accuracy = ', acc_test)



    ax = plt.figure()

    plt.bar(x = 'train accuracy', height = acc_train, color='darkblue')

    plt.bar(x = 'test accuracy', height = acc_test, color='lightblue')

    plt.xticks(['train accuracy', 'test accuracy'])

    plt.title(title)
plot_train_test_accuracy(y_train, y_train_pred_dt, y_test, y_test_pred_dt, 'Decision Tree Classifier')
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators = 100)
clf_rf.fit(X_train, y_train)
y_train_pred_rf = clf_rf.predict(X_train)
y_test_pred_rf = clf_rf.predict(X_test)
plot_train_test_accuracy(y_train, y_train_pred_rf, y_test, y_test_pred_rf, 'Random Forest Classifier')
from lightgbm import LGBMClassifier



lgbm = LGBMClassifier(max_depth = 3, n_estimators = 500, random_state = 0)

lgbm.fit(X_train, y_train)
y_train_pred_lgbm = lgbm.predict(X_train)

y_test_pred_lgbm = lgbm.predict(X_test)
plot_train_test_accuracy(y_train, y_train_pred_lgbm, y_test, y_test_pred_lgbm, 'LGBM Classifier')
from lightgbm import plot_importance

plot_importance(lgbm, max_num_features = 10)

plt.show()