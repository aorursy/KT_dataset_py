# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # plotting libraries

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC



#set style of sns

sns.set(style='whitegrid', color_codes=True)

sns.set(rc={'figure.figsize':(12,9)})



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load in the data and check out the first few rows

df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head()
#first, get some info about the data - note, first run appears to be a complete dataset with no null values

print(df.info())

print(df.describe())
#Class = 1 when there is a fraud detected; so lets investigate the proportion of the total that is fraudulent, and also correlations

prop_fraud = df['Class'].sum()/len(df['Class'])

prop_fraud
heatmap = sns.heatmap(df.corr()[['Class']].sort_values(by='Class', ascending=False), vmin=-1, vmax=1, annot=True, cmap="Blues")

heatmap.set_title('Correlation of features with fraudulent transactions', fontdict={'Fontsize': 18}, pad=16)
#there appears to be very little correlation between the factors, but a relationship does exist. Lets split the data into train and test

X = df.drop(columns='Class')

y = df['Class']



print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=35)
#As a classification exercise, using the suggestions here (https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html), so attempting SVC

svc = LinearSVC(random_state=32, class_weight='balanced')

svc.fit(X_train, y_train)

pred_svc = svc.predict(X_test)
#lets try out a Kneighbors and a random forest model as well, to see if we can improve any further
neigh = KNeighborsClassifier()

neigh.fit(X_train, y_train)

pred_neigh = neigh.predict(X_test)
forest = RandomForestClassifier(class_weight='balanced', random_state=32)

forest.fit(X_train, y_train)

pred_forest = forest.predict(X_test)
#accuracy

print('SVC accuracy : ', accuracy_score(y_test, pred_svc, normalize=True))

print('K Neighbors accuracy : ', accuracy_score(y_test, pred_neigh, normalize=True))

print('Random Forest accuracy : ', accuracy_score(y_test, pred_forest, normalize=True))
#Classification report, to check some more details

svc_class = classification_report(y_test, pred_svc)

neigh_class = classification_report(y_test, pred_neigh)

forest_class = classification_report(y_test, pred_forest)



print('----svc_class----')

print(svc_class)

print('----neigh_class----')

print(neigh_class)

print('----forest_class----')

print(forest_class)
#finalise with a Precision Recall Curve to visualise efficacy

models = [pred_svc, pred_neigh, pred_forest]



fig = plt.figure(figsize=(12,9))

ax1 = fig.add_subplot(1,2,1)

ax1.set_xlim([-.05,1.05])

ax1.set_ylim([-.05,1.05])

ax1.set_xlabel('Recall')

ax1.set_ylabel('Precision')

ax1.set_title('PR Curve')



ax2 = fig.add_subplot(1,2,2)

ax2.set_xlim([-.05,1.05])

ax2.set_ylim([-.05,1.05])

ax2.set_xlabel('False Positive Rate')

ax2.set_ylabel('True Positive Rate')

ax2.set_title('ROC Curve')



for i, j in zip(models, ['pred_svc', 'pred_neigh', 'pred_forest']):

    precision, recall, thresholds = precision_recall_curve(y_test, i)

    tpr, fpr, thresholds = roc_curve(y_test, i)

    ax1.plot(recall, precision, label=j)

    ax2.plot(tpr, fpr, label=j)

ax1.legend(loc='lower left')

ax2.legend(loc='lower left')



plt.show()