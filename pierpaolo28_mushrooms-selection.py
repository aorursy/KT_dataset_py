import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure

from sklearn.utils import shuffle

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import time

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

pd.options.display.max_columns = None

df.head()
percent_missing = df.isnull().sum() * 100 / len(df)

missing_values = pd.DataFrame({'percent_missing': percent_missing})

missing_values.sort_values(by ='percent_missing' , ascending=False)
sns.set(style="ticks")

f = sns.countplot(x="class", data=df, palette="bwr")

plt.show()
df['class'].value_counts()
df.shape
X = df.drop(['class'], axis = 1)

Y = df['class']
X = pd.get_dummies(X, prefix_sep='_')

X.head()
len(X.columns)
Y = LabelEncoder().fit_transform(Y)

#np.set_printoptions(threshold=np.inf)

Y
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier



X2 = StandardScaler().fit_transform(X)



X_Train, X_Test, Y_Train, Y_Test = train_test_split(X2, Y, test_size = 0.30, random_state = 101)
start = time.process_time()

trainedmodel = LogisticRegression().fit(X_Train,Y_Train)

print(time.process_time() - start)

predictions =trainedmodel.predict(X_Test)

print(confusion_matrix(Y_Test,predictions))

print(classification_report(Y_Test,predictions))
start = time.process_time()

trainedsvm = svm.LinearSVC().fit(X_Train, Y_Train)

print(time.process_time() - start)

predictionsvm = trainedsvm.predict(X_Test)

print(confusion_matrix(Y_Test,predictionsvm))

print(classification_report(Y_Test,predictionsvm))
start = time.process_time()

trainedtree = tree.DecisionTreeClassifier().fit(X_Train, Y_Train)

print(time.process_time() - start)

predictionstree = trainedtree.predict(X_Test)

print(confusion_matrix(Y_Test,predictionstree))

print(classification_report(Y_Test,predictionstree))
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz



data = export_graphviz(trainedtree,out_file=None,feature_names= X.columns,

                       class_names=['edible', 'poisonous'],  

                       filled=True, rounded=True,  

                       max_depth=2,

                       special_characters=True)

graph = graphviz.Source(data)

graph
start = time.process_time()

trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)

print(time.process_time() - start)

predictionforest = trainedforest.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))
figure(num=None, figsize=(20, 22), dpi=80, facecolor='w', edgecolor='k')



feat_importances = pd.Series(trainedforest.feature_importances_, index= X.columns)

feat_importances.nlargest(19).plot(kind='barh')
X_Reduced = X[['odor_n','odor_f', 'gill-size_n','gill-size_b']]

X_Reduced = StandardScaler().fit_transform(X_Reduced)

X_Train2, X_Test2, Y_Train2, Y_Test2 = train_test_split(X_Reduced, Y, test_size = 0.30, random_state = 101)
start = time.process_time()

trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train2,Y_Train2)

print(time.process_time() - start)

predictionforest = trainedforest.predict(X_Test2)

print(confusion_matrix(Y_Test2,predictionforest))

print(classification_report(Y_Test2,predictionforest))
from sklearn.feature_selection import RFE



model = RandomForestClassifier(n_estimators=700)

rfe = RFE(model, 4)

start = time.process_time()

RFE_X_Train = rfe.fit_transform(X_Train,Y_Train)

RFE_X_Test = rfe.transform(X_Test)

rfe = rfe.fit(RFE_X_Train,Y_Train)

print(time.process_time() - start)

print("Overall Accuracy using RFE: ", rfe.score(RFE_X_Test,Y_Test))
model = RandomForestClassifier(n_estimators=700)

rfe = RFE(model, 4)

RFE_X_Train = rfe.fit_transform(X_Train,Y_Train)

model.fit(RFE_X_Train,Y_Train) 

print("Number of Features: ", rfe.n_features_)

print("Selected Features: ")

colcheck = pd.Series(rfe.support_,index = list(X.columns))

colcheck[colcheck == True].index
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel



model = ExtraTreesClassifier()

start = time.process_time()

model = model.fit(X_Train,Y_Train)

model = SelectFromModel(model, prefit=True)

print(time.process_time() - start)

Selected_X = model.transform(X_Train)

Selected_X.shape
start = time.process_time()

trainedforest = RandomForestClassifier(n_estimators=700).fit(Selected_X, Y_Train)

print(time.process_time() - start)

Selected_X_Test = model.transform(X_Test)

predictionforest = trainedforest.predict(Selected_X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

importances = trainedforest.feature_importances_

std = np.std([tree.feature_importances_ for tree in trainedforest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(Selected_X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(Selected_X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(Selected_X.shape[1]), indices)

plt.xlim([-1, Selected_X.shape[1]])

plt.show()
Numeric_df = pd.DataFrame(X)

Numeric_df['Y'] = Y

Numeric_df.head()
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')



corr= Numeric_df.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)



# Selecting only correlated features

corr_y = abs(corr["Y"])

highest_corr = corr_y[corr_y >0.5]

highest_corr.sort_values(ascending=True)
X_Reduced2 = X[['bruises_f' , 'bruises_t' , 'gill-color_b' , 'gill-size_b' , 'gill-size_n' , 'ring-type_p' , 'stalk-surface-below-ring_k' , 'stalk-surface-above-ring_k' , 

                'odor_f', 'odor_n']]

X_Reduced2 = StandardScaler().fit_transform(X_Reduced2)

X_Train3, X_Test3, Y_Train3, Y_Test3 = train_test_split(X_Reduced2, Y, test_size = 0.30, random_state = 101)
start = time.process_time()

trainedsvm = svm.LinearSVC().fit(X_Train3, Y_Train3)

print(time.process_time() - start)

predictionsvm = trainedsvm.predict(X_Test3)

print(confusion_matrix(Y_Test3,predictionsvm))

print(classification_report(Y_Test3,predictionsvm))
min_max_scaler = preprocessing.MinMaxScaler()

Scaled_X = min_max_scaler.fit_transform(X2)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



X_new = SelectKBest(chi2, k=2).fit_transform(Scaled_X, Y)

X_Train3, X_Test3, Y_Train3, Y_Test3 = train_test_split(X_new, Y, test_size = 0.30, random_state = 101)

start = time.process_time()

trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train3,Y_Train3)

print(time.process_time() - start)

predictionforest = trainedforest.predict(X_Test3)

print(confusion_matrix(Y_Test3,predictionforest))

print(classification_report(Y_Test3,predictionforest))
from sklearn.linear_model import LassoCV



regr = LassoCV(cv=5, random_state=101)

regr.fit(X_Train,Y_Train)

print("LassoCV Best Alpha Scored: ", regr.alpha_)

print("LassoCV Model Accuracy: ", regr.score(X_Test, Y_Test))

model_coef = pd.Series(regr.coef_, index = list(X.columns[:-1]))

print("Variables Eliminated: ", str(sum(model_coef == 0)))

print("Variables Kept: ", str(sum(model_coef != 0))) 
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')



top_coef = model_coef.sort_values()

top_coef[top_coef != 0].plot(kind = "barh")

plt.title("Most Important Features Identified using Lasso (!0)")