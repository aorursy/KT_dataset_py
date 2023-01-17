# For explore data

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



# For escalation of values

from scipy import stats



# For machine learning modeles

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.ensemble import  AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, chi2

from sklearn import preprocessing



# For the validation of models

from sklearn.metrics import accuracy_score, precision_score, recall_score

from numpy import mean

df = pd.read_csv('../input/weatherAUS.csv')

df.head()
# The Shape of dataset

df.shape
# Removing "RISK_MM"

df = df.drop(columns='RISK_MM')

df.shape
df.info()
df = df.drop(columns=['Location','Date', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1)

df.info()
# Shape vizualization

df.shape
df = df.dropna(how='any')

df.shape
# Ploting blockspots

sns.boxplot(x=df['MinTemp'])

plt.show()
sns.boxplot(x=df['MaxTemp'])

plt.show()
sns.boxplot(x=df['WindGustSpeed'])

plt.show()
sns.boxplot(x=df['Rainfall'])

plt.show()
sns.boxplot(x=df['WindSpeed9am'])

plt.show()
sns.boxplot(x=df['Humidity9am'])

plt.show()
# Appling a escalation in numerics datas using "get_numeric_data".

z = np.abs(stats.zscore(df._get_numeric_data()))

# Print a table with z-scores

print(z)

# Removing outliers

df= df[(z < 3).all(axis=1)]

# Looking the new shape of dataframe

print(df.shape)
df.info()
len(df.WindGustDir.value_counts())
# List of features that will be changed

winds = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

    

# Doing the transformation with "get_dummies"

df = pd.get_dummies(df, columns=winds)



# Cheking the new shape

df.shape
# Removing one collumns of each group

df = df.drop(['WindGustDir_WSW', 'WindDir3pm_SSW', 'WindDir9am_NNE'], axis =1)

df.shape
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)

df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
df.RainToday.value_counts()
df.RainTomorrow.value_counts()
# Doing the escalation using "MinMaxScale" model

scaler = preprocessing.MinMaxScaler()

# Training the model

scaler.fit(df)

# Changing data 

df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

# Returning the data frama after the escalation

df.head()
# Splinting the data in features (X) and labels (y)

X = df.loc[:,df.columns!='RainTomorrow']

y = df[['RainTomorrow']]

# Using função SelectKBest and determining the parameters numbers of features, K = 58

selector = SelectKBest(chi2, k=58)

# Traning

selector.fit(X, y)

# Returning scores

scores = selector.scores_

# Creating a list for features names

lista = df.columns

lista = [x for x in lista if x != 'RainTomorrow']

# Creationg a dictionaty with the features name list and scores  

unsorted_pairs = zip(lista, scores)

sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))

k_best_features = dict(sorted_pairs[:58])
# Ploting the graphic area

plt.figure(figsize=(20,7),facecolor = 'w',edgecolor = 'w')

# Ploting the bar graphic

p = plt.bar(range(len(k_best_features)), list(k_best_features.values()), align='center')

plt.xticks(range(len(k_best_features)), list(k_best_features.keys()))

# Editing the names

plt.xticks(rotation='90')

plt.title('K best features scores')

plt.xlabel('Features')

plt.ylabel('Score')

plt.show()
# Creating a list of features names with score above 71 points

K_values = []

for key in k_best_features:

    if float(k_best_features[key]) >= float(0.01 * k_best_features['RainToday']):

        K_values.append(key)
df_predi = df[K_values + ['RainTomorrow']]

X = df[K_values]

y = df['RainTomorrow']
# Cirando uma lista para contagens de features de K_values, no caso 31

n_features_list = list(range(2,len(K_values)+1))
# Creating list for each model to append reseults

# For Logistic Regression

accuracy_LR=[]

# For Decision Tree

accuracy_dt=[]

# For Kmeans

accuracy_Kmeans=[]



# Creating a loop for the number of features unsing the list of names features

for n in n_features_list:  

    

    # Splinting the values for the training and test sets with "train_test_split"

    # We will leave 20% of the data for test and the rest for training.

    features_train, features_test, labels_train, labels_test = train_test_split(df[K_values[:n]], y, test_size=0.2, random_state=42)



    # Applying Logistic Regression model

    l_clf = LogisticRegression()

    # Training

    l_clf.fit(features_train, labels_train)

    # Doing the prediction

    prediction_lr = l_clf.predict(features_test)

    # Append the values of accuracy in a list

    accuracy_LR.append(accuracy_score(labels_test, prediction_lr))

    

    # The steps are the same for others models

    

    # For Decision Tree

    dt_clf = DecisionTreeClassifier(random_state=0)

    dt_clf.fit(features_train, labels_train)

    prediction_dt = dt_clf.predict(features_test)

    accuracy_dt.append(accuracy_score(labels_test, prediction_dt))

    

    # For Kmeans

    k_clf = KMeans(n_clusters=2)

    k_clf.fit(features_train, labels_train)

    prediction_k = k_clf.predict(features_test)

    accuracy_Kmeans.append(accuracy_score(labels_test, prediction_k))

# Ploting the graphic area

plt.figure(figsize=(9,6),facecolor = 'w',edgecolor = 'w')



# Ploting the graphic about accuracy x number of features for each model

# Losgistic Regression

line1 = plt.plot(n_features_list, accuracy_LR, 'b', label='LR')

# Decision Tree

line2 = plt.plot(n_features_list, accuracy_dt, 'r', label= 'dt')

# Kmeans

line3 = plt.plot(n_features_list, accuracy_Kmeans, 'g', label= 'Kmean')



# Editing the names 

plt.legend(('Logistic Regression', 'Decision tree', 'Kmean'), loc = 'best')

# Editing labels

plt.title('accuracy x features')

plt.ylabel('accuracy score')

plt.xlabel('n features')

# Editing grids

plt.xticks(n_features_list)

plt.grid(b='true',which='both', axis='both')

plt.show()
# Splinting data to test and traing with 3 features

X = df[K_values[:3]]

y = df['RainTomorrow']

features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating a list with parameters 

parameters = {'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), 'C':[0.01, 0.1, 10, 10**5,10**10,10**15,10**20],'tol':[10**-20,10**-15,10**-10,10**-5,0.01, 0.1, 10]}

# Applying the model

l_clf = LogisticRegression()

clf = GridSearchCV(l_clf, parameters)

clf.fit(features_train, labels_train)

# Outout of parameters 

best_l_clf = clf.best_estimator_

clf.best_estimator_
# Creating a list with parameters 

parameters = { 'criterion': ('gini', 'entropy'), 'min_samples_leaf' : range(1, 5), 'max_depth' : range(1, 5), 'class_weight': ['balanced'] }

# Applying the model

dt_clf = DecisionTreeClassifier(random_state=0)

clf = GridSearchCV(dt_clf, parameters)

# Outout of parameters 

clf.fit(features_train, labels_train)

clf.best_estimator_
# Creating a list with parameters 

parameters = {'algorithm':('auto', 'full', 'elkan'), 'tol':[10**-20,10**-15,10**-10,10**-5,0.01, 0.1, 10], 'n_init': [10,25,50,75,100,200], 'algorithm': ('auto', 'full', 'elkan')}

# Applying the model

k_clf = KMeans(n_clusters=2)

clf = GridSearchCV(k_clf, parameters)

# Outout of parameters 

clf.fit(features_train, labels_train)

clf.best_estimator_
# Logistic Regression

l_clf = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='saga', tol=0.1,

          verbose=0, warm_start=False)
# Decision Tree com AdaBoost

dt_clf = AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=3,

            max_features=None, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=2, min_samples_split=2,

            min_weight_fraction_leaf=0.0, presort=False, random_state=0,

            splitter='best'), n_estimators=50, learning_rate=.8)
# Kmean

k_clf =  KMeans(algorithm='elkan', copy_x=True, init='k-means++', max_iter=300,

    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',

    random_state=None, tol=1e-20, verbose=0)
def avaliacao_clf(clf, features, labels, n_iters=1000):

    print (clf)

    

    # Creating list for outputs

    accuracy = []

    precision = []

    recall = []

    first = True

    

    # Creating a loop to thousand interactions

    for tentativa in range(n_iters):

        

        # Splinting data to test and traing

        features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.3)



        # Applying the model

        clf.fit(features_train, labels_train)

        predictions = clf.predict(features_test)

        # Appending accuracy

        accuracy.append(accuracy_score(labels_test, predictions))

        # Appending precision

        precision.append(precision_score(labels_test, predictions))

        # Appending recall

        recall.append(recall_score(labels_test, predictions))



    # Taking the average of metrics for evaluating and implementing the results



    print ("precision: {}".format(mean(precision)))

    print ("recall:    {}".format(mean(recall)))

    print ("accuracy:    {}".format(mean(accuracy)))

    

    return mean(precision), mean(recall), mean(accuracy)
avaliacao_clf(l_clf, X, y)
avaliacao_clf(dt_clf, X, y)
avaliacao_clf(k_clf, X, y)