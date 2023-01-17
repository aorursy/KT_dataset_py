# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
advertising_dataset = pd.read_csv('../input/advertising/advertising.csv')
advertising_dataset.shape
advertising_dataset.head()
advertising_dataset.info()
sns.countplot(advertising_dataset['Clicked on Ad'], data=advertising_dataset)
sns.distplot(advertising_dataset['Age'][advertising_dataset['Clicked on Ad']==1], color='green')

sns.distplot(advertising_dataset['Age'][advertising_dataset['Clicked on Ad']==0], color='red')

plt.legend("0", "1")

plt.show()
sns.distplot(advertising_dataset['Daily Time Spent on Site'][advertising_dataset['Clicked on Ad']==1], color='green')

sns.distplot(advertising_dataset['Daily Time Spent on Site'][advertising_dataset['Clicked on Ad']==0], color='red')

plt.show()
sns.countplot(advertising_dataset['Male'][advertising_dataset['Clicked on Ad']==1])
sns.countplot(advertising_dataset['Male'][advertising_dataset['Clicked on Ad']==0])
x = advertising_dataset['Area Income']

y = advertising_dataset['Clicked on Ad']

plt.scatter(x, y)
sns.distplot(advertising_dataset['Daily Internet Usage'][advertising_dataset['Clicked on Ad']==1], color='green')

sns.distplot(advertising_dataset['Daily Internet Usage'][advertising_dataset['Clicked on Ad']==0], color='red')
sns.pairplot(advertising_dataset, hue = 'Clicked on Ad', vars = ['Daily Time Spent on Site', 'Age' ,'Area Income', 'Daily Internet Usage'],

             palette = 'husl')
advertising_dataset.duplicated().sum()
advertising_dataset.describe()
advertising_dataset['Timestamp'] = pd.to_datetime(advertising_dataset['Timestamp']) 

advertising_dataset['Month'] = advertising_dataset['Timestamp'].dt.month 

advertising_dataset['Day'] = advertising_dataset['Timestamp'].dt.day     

advertising_dataset['Hour'] = advertising_dataset['Timestamp'].dt.hour   

advertising_dataset["Weekday"] = advertising_dataset['Timestamp'].dt.dayofweek 
advertising_dataset = advertising_dataset.drop(['Timestamp'], axis=1)

advertising_dataset.head()
plt.figure(figsize=(12,5))

sns.heatmap(advertising_dataset.corr(), cmap='RdYlGn', annot=True)

plt.show()
advertising_dataset.drop(['Male', 'Month', 'Day', 'Hour', 'Weekday'], axis=1)
cat_features = advertising_dataset.select_dtypes(include=['object']).columns

cat_features
advertising_dataset = pd.get_dummies(advertising_dataset)

advertising_dataset.shape
features = advertising_dataset.drop(['Clicked on Ad'], axis=1)

label = advertising_dataset['Clicked on Ad']

features.shape, label.shape
from sklearn.model_selection import train_test_split

feature_train, feature_test, label_train, label_test = train_test_split(features, label, random_state = 7, test_size=0.3)

feature_train.shape, feature_test.shape, label_train.shape, label_test.shape
#Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]



#Number of features to consider in every split

max_features = ['auto', 'sqrt']



#Maximum number of levels in a tree

max_depth = [int(x) for x in np.linspace(start=5, stop=30, num=6)]



#Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]



#Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]



#Random Grid

random_grid = {'n_estimators' : n_estimators,

              'max_features' : max_features,

              'max_depth' : max_depth,

              'min_samples_split' : min_samples_split,

              'min_samples_leaf' : min_samples_leaf}
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

random_forest = RandomForestClassifier()

random_forest_model = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid, scoring='accuracy',

                                        cv=5, n_jobs=1, n_iter=10, verbose=2)

random_forest_model.fit(feature_train, label_train)
random_forest_model.best_params_
label_pred = random_forest_model.predict(feature_test)

label_pred
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
confusion_matrix(label_test, label_pred)
plot_confusion_matrix(random_forest_model, feature_test, label_test)
accuracy_score(label_train, random_forest_model.predict(feature_train))
accuracy_score(label_test, label_pred)
recall_score(label_test, label_pred)
precision_score(label_test, label_pred)
f1_score(label_test, label_pred)
plt.style.use('seaborn')



fpr, tpr, thresholds = roc_curve(label_test, random_forest_model.predict_proba(feature_test)[:,1], pos_label=1)



random_probs = [0 for i in range(len(label_test))]

p_fpr, p_tpr, _ = roc_curve(label_test, random_probs, pos_label=1)



plt.plot(fpr, tpr, linestyle='--',color='orange', label='Random Forest')

plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')



plt.title('Random Forest ROC Curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')



plt.legend(loc='best')

plt.savefig('ROC',dpi=300)



plt.show()
auc = roc_auc_score(label_test, random_forest_model.predict_proba(feature_test)[:,1])

auc