# uncheck this cell to install the needed modules

# !pip install pandas as pd #data structure

# !pip install numpy as np #numerical computing

# !pip install matplotlib.pyplot as plt #matlab based plotting

# !pip install seaborn as sns #more pretty visulzation

# !pip install warnings #warning messages eliminating

!pip install dython #data analysis tools for python 3.x

# !pip install math #mathematical functions

# !pip install catboost #gradient boosting

# !pip install tensorflow #keras backend

!pip install scipy #math operations

# !pip install graphviz #decision tree visualzations

!pip install pydotplus #convert graphviz viz from svg to png

import warnings

warnings.filterwarnings('ignore')
#import modules

import pandas as pd #data structure

import numpy as np #numerical computing

import matplotlib.pyplot as plt #matlab based plotting

import seaborn as sns #more pretty visulzation

import dython #data analysis tools for python 3.x

import catboost #gradient boosting

import tensorflow #keras backend

import scipy.stats as ss #math operations

import pydotplus

import graphviz #decision tree visualzations

#configurations#

# %autosave 60

%matplotlib inline

# %config InlineBackend.figure_format ='retina'

import datetime

print('Last update on the nootebook was: \n', datetime.datetime.now())
# if you read data from local path

# df = pd.read_csv('/content/mushrooms.csv')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

print('First 5 rows of all columns: \n\n',df.head().T)

print('\nTotal number of columns: \n',df.shape[1])

print('\nTotal number of rows: \n',df.shape[0])
print(df.describe().T)
import math #mathematical functions

from collections import Counter

def conditional_entropy(x,y):

    # entropy of x given y

    y_counter = Counter(y)

    xy_counter = Counter(list(zip(x,y)))

    total_occurrences = sum(y_counter.values())

    entropy = 0

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        entropy += p_xy * math.log(p_y/p_xy)

    return entropy



def theil_u(x,y):

    s_xy = conditional_entropy(x,y)

    x_counter = Counter(x)

    total_occurrences = sum(x_counter.values())

    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))

    s_x = ss.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) / s_x   

#correlation viz

theilu = pd.DataFrame(index=['class'])

columns = df.columns

for j in range(0,len(columns)):

    u = theil_u(df['class'].tolist(),df[columns[j]].tolist())

    theilu.loc[:,columns[j]] = u

theilu.fillna(value=np.nan,inplace=True)

plt.figure(figsize=(20,1))

sns.heatmap(theilu,annot=True,fmt='.2f')

plt.show()
#viz per class

fig, ax = plt.subplots(1,3, figsize=(15,5))

sns.countplot(x="cap-shape", hue='class', data=df, ax=ax[0])

sns.countplot(x="cap-surface", hue='class', data=df, ax=ax[1])

sns.countplot(x="cap-color", hue='class', data=df, ax=ax[2])

fig, ax = plt.subplots(1,2, figsize=(15,5))

sns.countplot(x="bruises", hue='class', data=df, ax=ax[0])

sns.countplot(x="odor", hue='class', data=df, ax=ax[1])

fig, ax = plt.subplots(1,4, figsize=(20,5))

sns.countplot(x="gill-attachment", hue='class', data=df, ax=ax[0])

sns.countplot(x="gill-spacing", hue='class', data=df, ax=ax[1])

sns.countplot(x="gill-size", hue='class', data=df, ax=ax[2])

sns.countplot(x="gill-color", hue='class', data=df, ax=ax[3])

fig, ax = plt.subplots(2,3, figsize=(20,10))

sns.countplot(x="stalk-shape", hue='class', data=df, ax=ax[0,0])

sns.countplot(x="stalk-root", hue='class', data=df, ax=ax[0,1])

sns.countplot(x="stalk-surface-above-ring", hue='class', data=df, ax=ax[0,2])

sns.countplot(x="stalk-surface-below-ring", hue='class', data=df, ax=ax[1,0])

sns.countplot(x="stalk-color-above-ring", hue='class', data=df, ax=ax[1,1])

sns.countplot(x="stalk-color-below-ring", hue='class', data=df, ax=ax[1,2])

fig, ax = plt.subplots(2,2, figsize=(15,10))

sns.countplot(x="veil-type", hue='class', data=df, ax=ax[0,0])

sns.countplot(x="veil-color", hue='class', data=df, ax=ax[0,1])

sns.countplot(x="ring-number", hue='class', data=df, ax=ax[1,0])

sns.countplot(x="ring-type", hue='class', data=df, ax=ax[1,1])

fig, ax = plt.subplots(1,3, figsize=(20,5))

sns.countplot(x="spore-print-color", hue='class', data=df, ax=ax[0])

sns.countplot(x="population", hue='class', data=df, ax=ax[1])

sns.countplot(x="habitat", hue='class', data=df, ax=ax[2])

fig.tight_layout()

fig.show()
# exclude any Na's which is represented by '?'

df = df[df['stalk-root'] != '?']

# drop column veil-type becaue of 1 only unique observation

df = df.drop(['veil-type'],axis=1)

print('Unique columns from all data are: \n\n',np.unique(df.columns))

print('\nUnique values from all columns: \n',np.unique(df.values))

print('\nTotal number of new columns: \n',df.shape[1])

print('\nTotal number of new rows: \n',df.shape[0])

# How many Na's count per column

# df.isnull().sum().sort_values(ascending=False)

print('\nCheck if we have na value in any column:\n',df.isnull().any())
print(df.describe().T)
#one hot label encoding

features = df.iloc[:,1:]

features = pd.get_dummies(features)

target = df.iloc[:,0].replace({'p': 0, 'e': 1})

print('First 5 rows of new encoded feature columns:\n',features.head())

print('First 5 rows of new encoded target class of mushroom poisonous = 0 edible = 1:\n',target.head())

X = features.values

y = target.values
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



X_train, X_test, y_train, y_test = train_test_split(X,y,

                                                    test_size = 0.3,

                                                    random_state=29)

target_names = ['poisonous', 'edible']

print ('X_train Shape:', X_train.shape)

print ('X_test Shape:', X_test.shape)

print ('y_train Shape:', y_train.shape)

print ('y_test Shape:', y_test.shape)
#calling kmeans classifier from sklearn

from sklearn.cluster import KMeans

# setting the classifier parameters

k_means=KMeans(n_clusters=2)

#Fitting kmesnd to training set

k_means.fit(X_train, y_train)

#Predicting values on test set

k_means_predict = k_means.predict(X_test)

#report the results

print("\nKmeans confusion matrix: \n",confusion_matrix(y_test, k_means_predict))

print("\nKmeans Classifier report: \n",classification_report(y_test,k_means_predict,target_names=target_names))

#testing

# print("\nAccuraccy score of the model is:\n",accuracy_score(k_means_predict, y_test)*100)
#calling the svm classifier from sklearn

from sklearn.svm import SVC

# setting the classifier parameters

svm = SVC(kernel= 'sigmoid',gamma='scale',probability=True)

#Fitting SVM to training set

svm.fit(X_train, y_train)

#Predicting values on test set

svm_predict = svm.predict(X_test)

#report the results

print("\nSVM confusion matrix: \n",confusion_matrix(y_test, svm_predict))

print("\nSVM classification report: \n",classification_report(y_test,svm_predict,target_names=target_names))
#calling the decision tree classifier from sklearn and graphiz for visuals

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier, export_graphviz

# setting the classifier parameters

dtree = tree.DecisionTreeClassifier(max_depth=3)

#Fitting decision tree to training set

dtree.fit(X_train, y_train)

#Predicting values on test set

dtree_predict = dtree.predict(X_test)

#report the results

print("\nDecision tree confusion matrix: \n",confusion_matrix(y_test, dtree_predict))

print("\nDecision tree classification report: \n",classification_report(y_test,dtree_predict,target_names=target_names))

#test

# print(accuracy_score(y_test,dtree_predict)) #raw_score

dtree_viz = export_graphviz(dtree, out_file=None, 

                         feature_names=features.columns,  

                         filled=True, rounded=True,  

                         special_characters=True,

                         impurity=True,proportion=True,

                         rotate=True,node_ids=True,

                         class_names=['Poisonous','Edible'])  

import pydotplus #convert graphviz viz from svg to png

# Draw graph

graph = pydotplus.graph_from_dot_data(dtree_viz)  



from IPython.display import Image  

# Show graph as png since it default output it as svg

Image(graph.create_png())
from catboost import CatBoostClassifier, Pool

#catboost model, use_best_model params will make the model prevent overfitting

catboost = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True)

catboost.fit(X_train,y_train,use_best_model=True,eval_set=(X_test,y_test),verbose=False)

catboost_predict = catboost.predict(X_test)

#report the results

print("Catboost confusion matrix: \n",confusion_matrix(y_test, catboost_predict))

print("Catboost classification report: \n",classification_report(y_test,catboost_predict,target_names=target_names))
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(50,),

                    activation='logistic',

                    max_iter=10,

                    solver='adam',

                    verbose=True)

mlp.fit(X_train, y_train)

mlp_predict = mlp.predict(X_test)

print("\nMLP confusion matrix: \n",confusion_matrix(y_test, mlp_predict))

print("\nMLP classification report: \n",classification_report(y_test,mlp_predict))
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(50, input_shape=(97,)),

                          keras.layers.Dense(2, activation='sigmoid')])

# model.summary()

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['acc'])

model.fit(X_train, y_train,epochs=5, verbose=1)

keras_pred = model.predict_classes(X_test)

# keras_pred = np.argmax(keras_pred, axis=1)

print('\nKeras confusion matrix:\n',confusion_matrix(keras_pred, y_test))

print('\nKeras classification Report:\n',classification_report(keras_pred, y_test,target_names=target_names))