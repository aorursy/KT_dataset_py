# Libraries for Data Analysis

import numpy as np

import pandas as pd

# Libraries for Visualization

import matplotlib.pyplot as plt

import seaborn as sns
# Loading the dataset from datasets module of sklearn package

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.keys())
# Creating features dataframe

df_feature = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Creating target dataframe

df_target = pd.DataFrame(cancer.target, columns=['cancer'])

# Concatenating dataframe

df = pd.concat([df_feature, df_target], axis=1)
df.head()
df.info()
df.describe()
df.iloc[:,:-1].hist(figsize=(20,15), edgecolor='black')

plt.show()
sns.set_style('whitegrid')

sns.countplot(df.cancer)
plt.figure(figsize=(10,8))

x_axis = df.iloc[:,:-1].corrwith(df.cancer).values

y_axis = df.iloc[:,:-1].corrwith(df.cancer).index

plt.barh(y_axis, x_axis)

plt.title('correlation with target(cancer)', fontsize=20)
# Fixing the problem of imbalanced data by stratifying the data.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('cancer',axis=1), df.cancer, stratify=df.cancer, random_state=66)
# Scaling the data for KNN model

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
# Finding the best value for n_neighbors parameter

from sklearn.neighbors import KNeighborsClassifier

train_accuracy = []

test_accuracy = []

for i in range(1,11):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train_scaled, y_train)

    train_accuracy.append(knn.score(X_train_scaled, y_train))

    test_accuracy.append(knn.score(X_test_scaled, y_test))



plt.figure(figsize=(10,5))

plt.plot(range(1,11), train_accuracy, label='train_accuracy')

plt.plot(range(1,11), test_accuracy, label='test_accuracy')

plt.legend()

plt.xlabel('n_neighbors')

plt.ylabel('accuracy')



score = pd.DataFrame({'n_neighbors':range(1,11),'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('n_neighbors')

score.transpose()
# Training the final model

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train_scaled,y_train)

knn_score = knn.score(X_test_scaled, y_test)
from sklearn.linear_model import LogisticRegression

train_accuracy = []

test_accuracy = []

for i in [0.001,0.01, 0.1, 1, 100]:

    logreg = LogisticRegression(C=i).fit(X_train, y_train)

    train_accuracy.append(logreg.score(X_train, y_train))

    test_accuracy.append(logreg.score(X_test, y_test))



score = pd.DataFrame({'C':[0.001,0.01, 0.1, 1, 100], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('C').transpose()

score
# Building the best model

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

logreg_score = logreg.score(X_test, y_test)
from sklearn.svm import LinearSVC

train_accuracy = []

test_accuracy = []

for i in [0.001, 0.01, 0.1, 1, 100]:

    lsvc = LinearSVC(C=i).fit(X_train, y_train)

    train_accuracy.append(lsvc.score(X_train, y_train))

    test_accuracy.append(lsvc.score(X_test, y_test))

    

pd.DataFrame({'C':[0.001,0.01,0.1,1,100], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('C').transpose()
# Building the best model

lsvc = LinearSVC(C=0.001).fit(X_train, y_train)

lsvc_score = lsvc.score(X_test, y_test)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(X_train, y_train)

gnb_score = gnb.score(X_test,y_test)

print('train_accuracy: {}'.format(gnb.score(X_train, y_train)))

print('test_accuracy: {}'.format(gnb_score))
from sklearn.tree import DecisionTreeClassifier

train_accuracy = []

test_accuracy = []

for i in [1,2,3,10,100]:

    tree = DecisionTreeClassifier(max_depth=i).fit(X_train, y_train)

    train_accuracy.append(tree.score(X_train, y_train))

    test_accuracy.append(tree.score(X_test, y_test))

    

pd.DataFrame({'max_depth':[1,2,3,10,100], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('max_depth').transpose()
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='tree_limited.dot', feature_names = X_train.columns,

                class_names = cancer.target_names,

                rounded = True, proportion = False, precision = 2, filled = True)



!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600



from IPython.display import Image

Image(filename = 'tree_limited.png')
plt.figure(figsize=(10,8))

plt.barh(X_train.columns, tree.feature_importances_)
# Building our final model

tree = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)

tree_score = tree.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

train_accuracy = []

test_accuracy = []

for i in [5, 20, 50, 75, 100]:

    forest = RandomForestClassifier(n_estimators=i, random_state=43).fit(X_train, y_train)

    train_accuracy.append(forest.score(X_train, y_train))

    test_accuracy.append(forest.score(X_test, y_test))

    

pd.DataFrame({'n_estimator':[5,20,50,75,100], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('n_estimator').transpose()
plt.figure(figsize=(10,8))

plt.barh(X_train.columns, forest.feature_importances_)
# Building our final model

forest = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

forest_score = forest.score(X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier

train_accuracy = []

test_accuracy = []

for i in [0.001,0.01,0.1,1]:

    boost = GradientBoostingClassifier(learning_rate=i).fit(X_train, y_train)

    train_accuracy.append(boost.score(X_train, y_train))

    test_accuracy.append(boost.score(X_test, y_test))

    

pd.DataFrame({'learning_rate':[0.001,0.01,0.1,1], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('learning_rate').transpose()
# Building the best model

boost = GradientBoostingClassifier(learning_rate= 0.1).fit(X_train, y_train)

boost_score = boost.score(X_test, y_test)

plt.figure(figsize=(10,8))

plt.barh(X_train.columns, boost.feature_importances_)
from sklearn.svm import SVC

train_accuracy = []

test_accuracy = []

for i in [1, 10 ,100 ,1000]:

    svc = SVC(C=i).fit(X_train, y_train)

    train_accuracy.append(svc.score(X_train, y_train))

    test_accuracy.append(svc.score(X_test, y_test))

    

pd.DataFrame({'C':[1,10,100,1000], 'train_accuracy':train_accuracy,'test_accuracy':test_accuracy}).set_index('C').transpose()
# Building the best model

svc = SVC(C=1000).fit(X_train, y_train)

svc_score = svc.score(X_test, y_test)
from sklearn.neural_network import MLPClassifier

train_accuracy = []

test_accuracy = []

for i in [[10],[10,10], [20,20]]:

    mlp = MLPClassifier(activation='tanh', random_state=0, hidden_layer_sizes=i).fit(X_train, y_train)

    train_accuracy.append(mlp.score(X_train, y_train))

    test_accuracy.append(mlp.score(X_test, y_test))

    

pd.DataFrame({'hidden_layers':['10','10,10','20,20'], 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy}).set_index('hidden_layers').transpose()
# Building the best model

mlp = MLPClassifier(activation='tanh', random_state=0, hidden_layer_sizes=[20,20]).fit(X_train, y_train)

mlp_score = mlp.score(X_test, y_test)
scores = pd.DataFrame({'model':['LogisticRegression', 'LinearSVM', 'DecisionTree','RandomForest','GradientBoosting',

                      'KernelSVM','NeuralNetwork','NaiveBayes'], 'accuracy':[logreg_score, lsvc_score, tree_score, forest_score,

                                                               boost_score, svc_score, mlp_score, gnb_score]})

scores.sort_values(by ='accuracy', ascending=False)