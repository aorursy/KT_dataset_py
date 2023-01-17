# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heart = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

heart.head()
heart.info()
import matplotlib.pyplot as plt

import seaborn as sns
# Histogram



sns.distplot(heart['age'], bins = 15)

plt.title('Distribution of Age')

plt.show()
# Count how many people have the disease and how many people don't.



ax = plt.subplot()

sns.countplot(heart['target'])

ax.set_xticks([0,1])

ax.set_xticklabels(['No Disease', 'Disease'])

plt.show()
# How many men and women have heart disease?



ax = plt.subplot()

sns.countplot(x = heart['target'], hue = 'sex', data = heart)

ax.set_xticklabels(['No Disease', 'Disease'])

plt.legend(['Female', 'Male'])

plt.title('Heart Disease Based on Sex')

plt.show()
# Correlation Matrix



corr_matrix = heart.corr()

plt.figure(figsize = (16,8))

sns.heatmap(corr_matrix, annot = True)

plt.show()
# Let's investigate the distibution of age based on heart disease.



plt.figure(figsize = (12,8))

sns.distplot(heart.age[heart['target'] == 0], label = 'No Disease', bins = 20)

sns.distplot(heart.age[heart['target'] == 1], label = 'Disease', bins = 20)

plt.legend()

plt.title('Age Distribution Based on Disease')

plt.show()
plt.figure(figsize = (10,6))

plt.scatter(x = heart.age[heart['target'] == 0], y = heart.thalach[heart['target'] == 0], c = 'red')

plt.scatter(x = heart.age[heart['target'] == 1], y = heart.thalach[heart['target'] == 1], c = 'blue')

plt.legend(['No Disease', 'Disease'])

plt.xlabel('Age')

plt.ylabel('Maximum Heart Rate')

plt.show()
# Import necessary libraries



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc
# Create dummy variables.



a = pd.get_dummies(heart['sex'], prefix = 'sex')

b = pd.get_dummies(heart['cp'], prefix = 'cp')

c = pd.get_dummies(heart['restecg'], prefix = 'restecg')

d = pd.get_dummies(heart['fbs'], prefix = 'fbs')

e = pd.get_dummies(heart['exang'], prefix = 'exang')

f = pd.get_dummies(heart['slope'], prefix = 'slope')

g = pd.get_dummies(heart['ca'], prefix = 'ca')

h = pd.get_dummies(heart['thal'], prefix = 'thal')



dummies = [heart, a, b, c, d, e, f, g, h]

heart = pd.concat(dummies, axis = 1)

heart = heart.drop(columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

heart.head()
# Split up data by features and target variables



X = heart.drop(['target'], axis = 1).values

y = heart['target']

print(X.shape)

print(y.shape)
# Scale feature columns



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
# Split data into train and test sets.



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(logreg.score(X_test, y_test))

print(logreg.score(X_train, y_train))
# Pick hyperparameters that result in the best accuracy rate.



from sklearn.svm import SVC



largest = {'value':0, 'gamma':1, 'C':1}

for gamma in range(1,7):

    for C in range(1,7):

        classifier = SVC(kernel = 'linear', C = C, gamma = gamma)

        classifier.fit(X_train, y_train)

        score = classifier.score(X_test, y_test)

        if (score > largest['value']):

            largest['value'] = score

            largest['gamma'] = gamma

            largest['C'] = C



print(largest)

print(classifier.score(X_train, y_train))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 25, bootstrap = True, max_features = 'sqrt')

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(rf.score(X_train, y_train))

print(rf.score(X_test, y_test))
from sklearn.neighbors import KNeighborsClassifier



train_accuracies = []

test_accuracies = []

for k in range(1,11):

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, y_train)

    train_accuracies.append(knn.score(X_train, y_train))

    test_accuracies.append(knn.score(X_test, y_test))

    

# Plotting the results.



k_list = range(1,11)

plt.plot(k_list, test_accuracies)

plt.xlabel('k')

plt.ylabel('Validation Accuracy')

plt.title('Accuracy Scores')

plt.show()



print('Accuracy score for KNN: ', round(max(train_accuracies) * 100), '%.')

print('Accuracy score for KNN: ', round(max(test_accuracies) * 100), '%.')
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

gnb.fit(X_train, y_train)

print(gnb.score(X_train, y_train))

print(gnb.score(X_test, y_test))
accuracy_scores = {'Model':['Logistic Regression', 'SVM', 'Random Forest', 'KNN', 'Naive Bayes'], 

                   'Training Score':[87.6,87.19,99.58,100,46.69],

                   'Test Score':[86.88,83.60,83.6,87,47.54]}

accuracy_scores_df = pd.DataFrame(accuracy_scores)

accuracy_scores_df
# Let's compare train set accuracy score.



sns.barplot(x = 'Model', y = 'Training Score', data = accuracy_scores_df)

plt.title('Training Accuracy Rate per Model')

plt.show()
# Let's compare test set accuracy score.



sns.barplot(x = 'Model', y = 'Test Score', data = accuracy_scores_df)

plt.title('Test Accuracy Rate per Model')

plt.show()