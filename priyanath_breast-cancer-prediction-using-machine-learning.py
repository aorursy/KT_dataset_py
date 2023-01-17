import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import io

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy.stats import norm

from scipy import stats

from sklearn.metrics import accuracy_score
df = pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
df.head()
df.columns
df.info()
df.describe()
f,ax=plt.subplots(1,2,figsize=(18,8))

df['diagnosis'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('diagnosis')

ax[0].set_ylabel('')

sns.countplot('diagnosis',data=df,ax=ax[1])

ax[1].set_title('diagnosis')

plt.show()





# 1 is for malignant, 0 is for benign
def diagnostic_plots(df, variable):## defining a function to plot histogram and Q-Q plot

    plt.figure(figsize = (15,6))

    plt.subplot(1,2,1)

    sns.distplot(df[variable], fit=norm);

    plt.subplot(1,2,2)

    stats.probplot(df[variable], dist = 'norm', plot = plt)

    plt.show()
diagnostic_plots(df, 'mean_radius')
#applying log transformation

df['mean_radius'] = np.log(df['mean_radius'] + 1)# +1 is added in case there is any 0 input to it which would create issue in taking log

diagnostic_plots(df, 'mean_radius')
diagnostic_plots(df, 'mean_texture')
#applying log transformation

df['mean_texture'] = np.log(df['mean_texture'] + 1)# +1 is added in case there is any 0 input to it which would create issue in taking log

diagnostic_plots(df, 'mean_texture')
diagnostic_plots(df, 'mean_perimeter')
#applying log transformation

df['mean_perimeter'] = np.log(df['mean_perimeter'] + 1)# +1 is added in case there is any 0 input to it which would create issue in taking log

diagnostic_plots(df, 'mean_perimeter')
diagnostic_plots(df, 'mean_area')
#applying log transformation

df['mean_area'] = np.log(df['mean_area'] + 1)# +1 is added in case there is any 0 input to it which would create issue in taking log

diagnostic_plots(df, 'mean_area')
diagnostic_plots(df, 'mean_smoothness')
#applying log transformation

df['mean_smoothness'] = np.log(df['mean_smoothness'] + 1)# +1 is added in case there is any 0 input to it which would create issue in taking log

diagnostic_plots(df, 'mean_smoothness')
#correlation matrix

corrmat = df.corr()

f, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(corrmat, vmax=.8, square=True, annot = True);
#pairplot

sns.set()

cols = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']

sns.pairplot(df[cols], size = 2.5)

plt.show();
y = df['diagnosis']

X = df.drop(['diagnosis'], axis = True)

y.head(3)
#splitting dataset into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

accu_reg = accuracy_score(y_test, Y_pred)

print("Accuracy score using Random Forest:", accu_reg*100)
#support vector machine

from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

accu_svc = accuracy_score(y_test, Y_pred)

print("Accuracy score using Random Forest:", accu_svc*100)
#knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

accu_knn = accuracy_score(y_test, Y_pred)

print("Accuracy score using Random Forest:", accu_knn*100)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = accuracy_score(y_test, Y_pred)

print("Accuracy score using Random Forest:", acc_gaussian*100)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = accuracy_score(y_test, Y_pred)

print("Accuracy score using Random Forest:", acc_decision_tree*100)
# Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = accuracy_score(y_test, Y_pred)

print("Accuracy score using Random Forest:", acc_random_forest*100)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 

              'Decision Tree'],

    'Score': [accu_svc, accu_knn, accu_reg, 

              acc_random_forest, acc_gaussian, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)