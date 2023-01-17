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
train_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')

test_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')



train_df.shape, test_df.shape
train_df.columns
train_df.info()
train_df.describe()
train_df.head()
#histogram



import seaborn as sns



sns.distplot(train_df['Cover_Type'])
#skewness and kurtosis

print("Skewness: %f" % train_df['Cover_Type'].skew())

print("Kurtosis: %f" % train_df['Cover_Type'].kurt())
#histogram and normal probability plot

from scipy import stats

from scipy.stats import norm

sns.distplot(train_df['Cover_Type'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_df['Cover_Type'], plot=plt)
train_df[['Elevation', 'Cover_Type']].groupby(['Elevation'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Aspect', 'Cover_Type']].groupby(['Aspect'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Slope', 'Cover_Type']].groupby(['Slope'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Horizontal_Distance_To_Hydrology', 'Cover_Type']].groupby(['Horizontal_Distance_To_Hydrology'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Vertical_Distance_To_Hydrology', 'Cover_Type']].groupby(['Vertical_Distance_To_Hydrology'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Horizontal_Distance_To_Roadways', 'Cover_Type']].groupby(['Horizontal_Distance_To_Roadways'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Hillshade_9am', 'Cover_Type']].groupby(['Hillshade_9am'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Hillshade_Noon', 'Cover_Type']].groupby(['Hillshade_Noon'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Hillshade_3pm', 'Cover_Type']].groupby(['Hillshade_3pm'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
train_df[['Horizontal_Distance_To_Fire_Points', 'Cover_Type']].groupby(['Horizontal_Distance_To_Fire_Points'], as_index = False).mean().sort_values(by = 'Cover_Type', ascending = False)
#scatterplot

import matplotlib.pyplot as plt

sns.set()

cols = ['Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',\

        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

sns.pairplot(train_df[cols], size = 2.5)

plt.show();



X1 = train_df.drop(['Cover_Type'],axis = 1)

y1 = train_df['Cover_Type']
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X1,y1)
#Using Feature Importance for Feature Selection and selection top 30 important features out of the 55 variables.

print(model.feature_importances_)

feat_imp = pd.Series(model.feature_importances_, index = X1.columns)

feat_imp.nlargest(30).plot(kind = 'barh')

plt.show()
X_train = train_df[['Id','Elevation', 'Horizontal_Distance_To_Roadways', 'Wilderness_Area4', 'Horizontal_Distance_To_Fire_Points', \

                     'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Hillshade_9am', 'Aspect', 'Hillshade_Noon', 'Hillshade_3pm',\

 'Slope', 'Soil_Type10', 'Soil_Type38', 'Soil_Type3', 'Soil_Type39', 'Soil_Type4', 'Soil_Type40', 'Soil_Type30', 'Soil_Type17', 'Soil_Type2',\

'Soil_Type13', 'Soil_Type22', 'Wilderness_Area1', 'Wilderness_Area3', 'Soil_Type23', 'Soil_Type29', 'Soil_Type12', 'Soil_Type32', 'Soil_Type33']].copy()

Y_train = y1

X_test = test_df[['Id','Elevation', 'Horizontal_Distance_To_Roadways', 'Wilderness_Area4', 'Horizontal_Distance_To_Fire_Points', \

                     'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Hillshade_9am', 'Aspect', 'Hillshade_Noon', 'Hillshade_3pm',\

 'Slope', 'Soil_Type10', 'Soil_Type38', 'Soil_Type3', 'Soil_Type39', 'Soil_Type4', 'Soil_Type40', 'Soil_Type30', 'Soil_Type17', 'Soil_Type2',\

'Soil_Type13', 'Soil_Type22', 'Wilderness_Area1', 'Wilderness_Area3', 'Soil_Type23', 'Soil_Type29', 'Soil_Type12', 'Soil_Type32', 'Soil_Type33']]

                 

X_train.shape, Y_train.shape, X_test.shape
#Standardized

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print(scaler.fit(X_train))

#MinMaxScaled

from sklearn.preprocessing import MinMaxScaler

scaler1 = MinMaxScaler()

print(scaler1.fit(X_train))
#logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

accu_reg = round(logreg.score(X_train, Y_train) * 100, 2)

accu_reg
# Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
#knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

accu_knn = round(knn.score(X_train, Y_train) *100, 2)

accu_knn
#Perceptron

from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC

from sklearn.svm import SVC, LinearSVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Decision Tree



from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [ accu_knn, accu_reg, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "Id": test_df["Id"],

        "Cover_Type": Y_pred

    })



submission.head(20)
submission.to_csv('Submission.csv', index=False)