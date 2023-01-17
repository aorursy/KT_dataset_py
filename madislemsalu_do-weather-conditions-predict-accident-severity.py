# Libraries 

import pandas as pd

import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

#from sklearn import preprocessing

import seaborn as sns

#from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt



pd.set_option('display.max_columns', 70) # Since we're dealing with moderately sized dataframe,

pd.set_option('display.max_rows', 13)# max 13 columns and rows will be shown
df=pd.read_csv("../input/Kaagle_Upload.csv", sep=",", decimal=",", engine='python') # Read the data from a csv

df=df.dropna() # The dataset is huge, therefore, dropping any rows with missing values is fine

df.head()

df.isnull().sum().sum()



# First I select variables based on prefrence, then for df2 I add weather related conditions of:

#'road_surface_conditions','light_conditions','weather_conditions'

#Feel free to mix these variables up

df2 = df[['special_conditions_at_site','pedestrian_movement','road_surface_conditions','light_conditions','weather_conditions','age_of_vehicle','sex_of_driver','age_of_driver','junction_location', 'junction_detail','junction_control','did_police_officer_attend_scene_of_accident','accident_severity','day_of_week']]

df1 = df[['special_conditions_at_site','pedestrian_movement','age_of_vehicle','sex_of_driver','age_of_driver','junction_location','junction_detail','junction_control','did_police_officer_attend_scene_of_accident','day_of_week','accident_severity']]





df1.replace(-1, np.nan, inplace=True) # -1 should be imputed to NaN to be recognized as missing in the next row

df1=df1.dropna() # I drop all the rows with missing data once again

df1.shape



df2.replace(-1, np.nan, inplace=True) # Same as previously 

df2=df2.dropna()

df2.shape
# Here I took a subset of features from the previous cell. This is so I could narrow it more down / 

# This can be considered redundant, but was mostly part of the workflow when looking at different variables

df1 = df1[['special_conditions_at_site','pedestrian_movement','age_of_vehicle','sex_of_driver','age_of_driver','junction_location','junction_detail',

           'junction_control','day_of_week',

           'accident_severity']]



df2 = df2[['special_conditions_at_site','pedestrian_movement','road_surface_conditions','light_conditions','weather_conditions','age_of_vehicle','sex_of_driver','age_of_driver',

          'junction_location', 'junction_detail','junction_control',

          'accident_severity','day_of_week']]



df1.shape

df2.shape


import matplotlib.pyplot as plt

corrmat = df2.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)



#ax = sns.pairplot(df, size)

plt.show()


#cols2 = ['junction_detail','light_conditions','weather_conditions','casualty_type','day_of_week','junction_control','road_surface_conditions','casualty_severity']



k = 6 #number of variables for heatmap

cols = corrmat.nlargest(k, 'accident_severity')['accident_severity'].index

cm = np.corrcoef(df2[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
df2.head()
# cols_with = ['junction_location','junction_detail','light_conditions','weather_conditions','day_of_week','junction_control','road_surface_conditions']

# cols_without = ['junction_location','junction_detail','day_of_week','junction_control']

# import seaborn as sns

# def one_hot(df, cols):

#     for each in cols:

#         dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)

#         df = pd.concat([df, dummies],axis=1)

#     df = df.drop(cols, axis=1)

#     return df  %%!

# df2 = one_hot(df2,cols_with)

# df1 = one_hot(df1,cols_without)

from scipy.stats import norm

from scipy import stats

#histogram and normal probability plot

sns.distplot(df1['age_of_driver'], fit=norm);

fig = plt.figure()

res = stats.probplot(df1['age_of_driver'], plot=plt)

plt.show()

df2['age_of_driver'] = np.log1p(df2['age_of_driver']) 

df2['age_of_vehicle'] = np.log1p(df2['age_of_vehicle'])# standardise the feature



df1['age_of_driver'] = np.log1p(df1['age_of_driver']) 

df1['age_of_vehicle'] = np.log1p(df1['age_of_vehicle'])#

sns.distplot(df1['age_of_driver'], fit=norm);

fig = plt.figure()

res = stats.probplot(df1['age_of_driver'], plot=plt)

plt.show()
df2
df1= df1[:15000] #keep 1500 to decrease running times

df2= df2[:15000] #keep 15000



Y = df2.accident_severity.values

Y1 = df1.accident_severity.values

Y
cols = df2.shape[1]

X = df2.loc[:, df2.columns != 'accident_severity']

X1 = df1.loc[:, df1.columns != 'accident_severity']

X.columns;
X.shape

X1.shape
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



X_train1, X_test1,Y_train1,Y_test1 = train_test_split(X1, Y1, test_size=0.33, random_state=99)

#Without weather

svc = SVC()

svc.fit(X_train1, Y_train1)

Y_pred = svc.predict(X_test1)

acc_svc1 = round(svc.score(X_test1, Y_test1) * 100, 2)

acc_svc1



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train1, Y_train1)

Y_pred = knn.predict(X_test1)

acc_knn1 = round(knn.score(X_test1, Y_test1) * 100, 2)

acc_knn1





# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train1, Y_train1)

Y_pred = logreg.predict(X_test1)

acc_log1 = round(logreg.score(X_train1, Y_train1) * 100, 2)

acc_log1





# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train1, Y_train1)

Y_pred = gaussian.predict(X_test1)

acc_gaussian1 = round(gaussian.score(X_test1, Y_test1) * 100, 2)

acc_gaussian1



# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train1, Y_train1)

Y_pred = perceptron.predict(X_test1)

acc_perceptron1 = round(perceptron.score(X_test1, Y_test1) * 100, 2)

acc_perceptron1



# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train1, Y_train1)

Y_pred = linear_svc.predict(X_test1)

acc_linear_svc1 = round(linear_svc.score(X_test1, Y_test1) * 100, 2)

acc_linear_svc1



# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train1, Y_train1)

Y_pred = sgd.predict(X_test1)

acc_sgd1 = round(sgd.score(X_test1, Y_test1) * 100, 2)

acc_sgd1



# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train1, Y_train1)

Y_pred = decision_tree.predict(X_test1)

acc_decision_tree1 = round(decision_tree.score(X_test1, Y_test1) * 100, 2)

acc_decision_tree1



# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train1, Y_train1)

Y_pred = random_forest.predict(X_test1)

random_forest.score(X_train1, Y_train1)

acc_random_forest1 = round(random_forest.score(X_test1, Y_test1) * 100, 2)

acc_random_forest1

# Same with weather related data

# Support Vector Machines

X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.33, random_state=99)

#with weather condition



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_test, Y_test) * 100, 2)

acc_svc

#KNN



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_test, Y_test) * 100, 2)

acc_knn



# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log



# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)

acc_gaussian



# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_test, Y_test) * 100, 2)

acc_perceptron



# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)

acc_linear_svc



# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_test, Y_test) * 100, 2)

acc_sgd



# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)

acc_decision_tree



# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)

acc_random_forest
print("Machine Learning algorithm scores without weather related conditions")

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc1, acc_knn1, acc_log1, 

              acc_random_forest1, acc_gaussian1, acc_perceptron1, 

              acc_sgd1, acc_linear_svc1, acc_decision_tree1]})

models.sort_values(by='Score', ascending=False)

print("Machine Learning algorithm scores with weather related conditions")

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)



# Confusion matrix with random forest

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

x,y = df1.loc[:,df1.columns != 'accident_severity'], df1.loc[:,'accident_severity']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

rf = RandomForestClassifier(random_state = 4)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print('Confusion matrix: \n',cm)

print('Classification report: \n',classification_report(y_test,y_pred))

y_test.value_counts()
# Confusion matrix with random forest

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

x,y = df2.loc[:,df2.columns != 'accident_severity'], df2.loc[:,'accident_severity']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

rf = RandomForestClassifier(random_state = 4)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print('Confusion matrix: \n',cm)

print('Classification report: \n',classification_report(y_test,y_pred))

y_test.value_counts()
sns.heatmap(cm,annot=True,fmt="d") 

plt.show()