# Importing the libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Importing the dataset



dataset = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
# Viewing the first five rows of dataset



dataset.head()
# We can see all the columns have categorical value.

# We have 22 features (independent variables) and a dependent variable (class).



# We will continue with data preprocessing but lets get some insights of the dataset first.
# Visualising the number of mushrooms that fall in each class - p = poisonous, e=edible

plt.style.use('dark_background')

plt.rcParams['figure.figsize']=8,8 

s = sns.countplot(x = "class", data = dataset)

for p in s.patches:

    s.annotate(format(p.get_height(), '.1f'), 

               (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

plt.show()
# In the given dataset we have 3916 poisonous mushrooms and 4208 edible mushrooms
features = dataset.columns

print(features)
f, axes = plt.subplots(22,1, figsize=(15,150), sharey = True) 

k = 1

for i in range(0,22):

    s = sns.countplot(x = features[k], data = dataset, ax=axes[i], palette = 'GnBu')

    axes[i].set_xlabel(features[k], fontsize=20)

    axes[i].set_ylabel("Count", fontsize=20)

    axes[i].tick_params(labelsize=15)

    k = k+1

    for p in s.patches:

        s.annotate(format(p.get_height(), '.1f'), 

        (p.get_x() + p.get_width() / 2., p.get_height()), 

        ha = 'center', va = 'center', 

        xytext = (0, 9), 

        fontsize = 15,

        textcoords = 'offset points')
# From above graph we can see how many mushrooms belong to each category in each feature
f, axes = plt.subplots(22,1, figsize=(15,150), sharey = True) 

k = 1

for i in range(0,22):

    s = sns.countplot(x = features[k], data = dataset, hue = 'class', ax=axes[i], palette = 'CMRmap')

    axes[i].set_xlabel(features[k], fontsize=20)

    axes[i].set_ylabel("Count", fontsize=20)

    axes[i].tick_params(labelsize=15)

    axes[i].legend(loc=2, prop={'size': 20})

    k = k+1

    for p in s.patches:

        s.annotate(format(p.get_height(), '.1f'), 

        (p.get_x() + p.get_width() / 2., p.get_height()), 

        ha = 'center', va = 'center', 

        xytext = (0, 9), 

        fontsize = 15,

        textcoords = 'offset points')
# From above graph we can see how many mushrooms belong to each category in each feature and among those how many are edible and how many are 

# poisonous mushrooms.
df1 = dataset[dataset['class'] == 'p']

df2 = dataset[dataset['class'] == 'e']

print(df1)

print(df2)
# Creating independent and dependent variables

x = dataset.iloc[:,1:].values

y = dataset.iloc[:,0].values
print(x)
print(len(x[0]))
print(y)
# Finding missing values



dataset.isna().sum()
dataset.info()
# Categories in each feature x

column_list = dataset.columns.values.tolist()

#print(column_list)

for column_name in column_list:

    print(column_name)

    print(dataset[column_name].unique())
# Label encoding y - dependent variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
print(y)
# One hot encoding independent variable x



from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

x = onehotencoder.fit_transform(x).toarray()
print(x[0])
# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 3)

x_train = pca.fit_transform(x_train)

x_test = pca.transform(x_test)
# Training the Logistic Regression Model on the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)
# Predicting the test set

y_pred = classifier.predict(x_test)
# Making the confusion matrix and calculating accuracy score

acscore = []

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

acscore.append(ac)

print(cm)

print(ac)
# Training the Naive Bayes Classification model on the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)
# Predicting the test set

y_pred = classifier.predict(x_test)
# Making the confusion matrix and calculating the accuarcy score

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

acscore.append(ac)

print(cm)

print(ac)
# Training the RBF Kernel SVC on the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state=0)

classifier.fit(x_train, y_train)
# predicting test set

y_pred = classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

acscore.append(ac)

print(cm)

print(ac)
# Calculating the optimum number of neighbors

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

list1 = []

for neighbors in range(3,10,1):

    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    list1.append(accuracy_score(y_test,y_pred))

plt.plot(list(range(3,10,1)), list1)

plt.show()
# Training the K Nearest Neighbor Classification on the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

classifier.fit(x_train, y_train)
# Predicting the test set

y_pred = classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

acscore.append(ac)

print(cm)

print(ac)
# Training the Decision Tree Classification on the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)

classifier.fit(x_train, y_train)
# Predicting the test set 

y_pred = classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

acscore.append(ac)

print(cm)

print(ac)
# Training the XGBoost Classification on the Training set

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(x_train,y_train)
# Predicting the test set

y_pred = classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

acscore.append(ac)

print(cm)

print(ac)
# Finding the optimum number of n_estimators

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

list1 = []

for estimators in range(10,150):

    classifier = RandomForestClassifier(n_estimators = estimators, random_state=0, criterion='entropy')

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    list1.append(accuracy_score(y_test,y_pred))

#print(mylist)

plt.plot(list(range(10,150)), list1)

plt.show()
# Training the Random Forest Classification on the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0, n_estimators = 100)

classifier.fit(x_train, y_train)
# Predicting the test set

y_pred = classifier.predict(x_test)
# Making the confusion matrix and accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

ac = accuracy_score(y_test, y_pred)

acscore.append(ac)

print(cm)

print(ac)
# Printing accuracy score of all the classification models we have applied 

print(acscore)
models = ['LogisticRegression','NaiveBayes','KernelSVM','KNearestNeighbors','DecisionTree','XGBoost','RandomForest']
# Visualising the accuracy score of each classification model

plt.rcParams['figure.figsize']=15,8 

plt.style.use('dark_background')

ax = sns.barplot(x=models, y=acscore, palette = "rocket", saturation =1.5)

plt.xlabel("Classifier Models", fontsize = 20 )

plt.ylabel("% of Accuracy", fontsize = 20)

plt.title("Accuracy of different Classifier Models", fontsize = 20)

plt.xticks(fontsize = 13, horizontalalignment = 'center', rotation = 0)

plt.yticks(fontsize = 13)

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')

plt.show()
# So among all classification model Random Forest Classification has highest accuracy score = 99.75%.