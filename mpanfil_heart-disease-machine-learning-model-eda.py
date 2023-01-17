from __future__ import division

import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.set_style('darkgrid')

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('../input/heart-disease-uci/heart.csv')



print('Number of features: %s' %data.shape[1])

print('Number of examples: %s' %data.shape[0])
data.head()
data.info()
data.describe()
data.isnull().sum()
# Evaluation 1 - age distribution



min_age = data['age'].min()

max_age = data['age'].max()

mean_age = round(data['age'].mean(),1)



print('Min age: %s' %min_age)

print('Max age: %s' %max_age)

print('Mean age: %s' %mean_age)
# Evaluation 2 - target percentage



# no disease = 0

# disease = 1



no_disease = len(data[data['target'] ==0])

with_disease = len(data[data['target'] ==1])



print('Percentage of people without disease: {:.2f} %' .format(no_disease/len(data['target'])*100))

print('Percentage of people with disease: {:.2f} %' .format(with_disease/len(data['target'])*100))
# Evaluation 3 - gender counts



# female = 0

# male = 1



female = len(data[data['sex'] ==0])

male = len(data[data['sex'] ==1])



print('Percentage of female: {:.2f} %' .format(female/len(data['sex'])*100))

print('Percentage of male: {:.2f} %' .format(male/len(data['sex'])*100))
# Evaluation 4 - mean value of target



mean_target = round(data['target'].mean(), 2)



print('Mean value of target: %s' %mean_target)



mean_target_df = pd.DataFrame(data.groupby('target').mean()).reset_index()

mean_target_df
# Visualisation 1 - heatmap of dataset



plt.figure(figsize=(15,8))

cbar_kws = { 'ticks' : [-1, -0.5, 0, 0.5, 1], 'orientation': 'horizontal'}

sns.heatmap(data.corr(), cmap='PuBu', linewidths=0.1, annot=True, vmax=1, vmin=-1, cbar_kws=cbar_kws)
# Visualisation 2 -  distribution of age



plt.figure(figsize=(15,8))

sns.distplot(data['age'], hist=True, bins=30, color='grey')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.title('Distribution of age', fontsize=15)
# Visualisation 3 -  gender



plt.figure(figsize=(15,8))

sns.countplot(data['sex'], palette='PuBu')

plt.xlabel('Gender')

plt.ylabel('Count')

plt.title('Gender', fontsize=15)
# Visualisation 4 - count of females and males vs age



plt.figure(figsize=(15,8))

sns.countplot(data['age'], hue=data['sex'], palette='PuBu', saturation=0.8)

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Gender count', fontsize=15)

plt.legend(loc='upper right', fontsize=15, labels=['Female', 'Male'])
# Visualisation 5 - count of target



plt.figure(figsize=(15,8))

sns.countplot(data['target'], palette='PuBu')

plt.xlabel('Target')

plt.ylabel('Count')

plt.title('Target count', fontsize=15)
# Visualisation 6 - target in age



plt.figure(figsize=(15,8))

sns.countplot(data['age'], hue=data['target'], palette='PuBu', saturation=0.8)

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Target count', fontsize=15)

plt.legend(loc='upper right', fontsize=15, labels=['No disease', 'Disease'])
# Visualisation 7 - target in genders



plt.figure(figsize=(15,8))

sns.countplot(data['sex'], hue=data['target'], palette='PuBu', saturation=0.8)

plt.xlabel('Sex')

plt.ylabel('Count')

plt.title('Target count in genders', fontsize=15)

plt.legend(loc='upper right', fontsize=15, labels=['No disease', 'Disease'])
# Visualisation 8 - crosstabs for features and target



names = ['Chest Pain Type', 'Slope', 'FBS - (Fasting Blood Sugar)', 'Resting electrocardiographic results',

        'Exercise induced angina', 'Number of major vessels', 'Thal']



for col in data[['cp', 'slope','fbs', 'restecg', 'exang', 'ca', 'thal']]:

    plt.figure(figsize=(15,8))

    sns.countplot(data[col], hue=data.target, palette='PuBu')

    plt.title(col)

    plt.legend(loc='upper right', fontsize=15, labels=['No disease', 'Disease'])
data.head()
# Dummy values for categorical features - cp, slope, thal



dummy_cp = pd.get_dummies(data.cp, prefix='cp')

dummy_slope = pd.get_dummies(data.slope, prefix='slope')

dummy_thal = pd.get_dummies(data.thal, prefix='thal')
# Merging dummies with daataframe and dropping this columns



data = pd.concat([data,dummy_cp, dummy_slope, dummy_thal], axis=1)

data.drop(['cp', 'slope', 'thal'], axis=1, inplace=True)
data.head()
# Preparing train and test data for machine learning algorithms



X_data = data.drop('target', axis=1)

y = data.target
# Normalizing data

X = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
log = LogisticRegression()

log.fit(X_train, y_train)

log_pred = log.predict(X_test)
log_conf = confusion_matrix(y_test, log_pred)

log_class = classification_report(y_test, log_pred)
log_acc_train = log.score(X_train, y_train)*100

log_acc_test = log.score(X_test, y_test)*100



print("Train Accuracy {:.2f}%".format(log_acc_train))

print("Test Accuracy {:.2f}%".format(log_acc_test))
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_conf = confusion_matrix(y_test, knn_pred)

knn_class = classification_report(y_test, knn_pred)
knn_acc_train = knn.score(X_train, y_train)*100

knn_acc_test = knn.score(X_test, y_test)*100



print("Train Accuracy {:.2f}%".format(knn_acc_train))

print("Test Accuracy {:.2f}%".format(knn_acc_test))
# Best value for KNN



knn_score_list = []

for i in range(1,20):

    knn_2 = KNeighborsClassifier(n_neighbors = i)

    knn_2.fit(X_train, y_train)

    knn_score_list.append(knn_2.score(X_test, y_test))

 



plt.figure(figsize=(15,8))

plt.plot(range(1,20), knn_score_list)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K values")

plt.ylabel("Score")



knn_acc_2_max = max(knn_score_list)*100

print("Maximum KNN Score is {:.2f}%".format(knn_acc_2_max))
svm = SVC(random_state=1)

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_conf = confusion_matrix(y_test, svm_pred)

svm_class = classification_report(y_test, svm_pred)
svm_acc_train = svm.score(X_train, y_train)*100

svm_acc_test = svm.score(X_test, y_test)*100



print("Train Accuracy {:.2f}%".format(svm_acc_train))

print("Test Accuracy {:.2f}%".format(svm_acc_test))
nb = GaussianNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
nb_conf = confusion_matrix(y_test, nb_pred)

nb_class = classification_report(y_test, nb_pred)
nb_acc_train = nb.score(X_train, y_train)*100

nb_acc_test = nb.score(X_test, y_test)*100



print("Train Accuracy {:.2f}%".format(nb_acc_train))

print("Test Accuracy {:.2f}%".format(nb_acc_test))
d_tree = DecisionTreeClassifier()

d_tree.fit(X_train, y_train)

dtree_pred = d_tree.predict(X_test)
dtree_conf = confusion_matrix(y_test, dtree_pred)

dtree_class = classification_report(y_test, dtree_pred)
dtree_acc_train = d_tree.score(X_train, y_train)*100

dtree_acc_test = d_tree.score(X_test, y_test)*100



print("Train Accuracy {:.2f}%".format(dtree_acc_train))

print("Test Accuracy {:.2f}%".format(dtree_acc_test))
rtree = RandomForestClassifier()

rtree.fit(X_train, y_train)

rtree_pred = rtree.predict(X_test)
rtree_conf = confusion_matrix(y_test, rtree_pred)

rtree_class = classification_report(y_test, rtree_pred)
rtree_acc_train = rtree.score(X_train, y_train)*100

rtree_acc_test = rtree.score(X_test, y_test)*100



print("Train Accuracy {:.2f}%".format(rtree_acc_train))

print("Test Accuracy {:.2f}%".format(rtree_acc_test))
# confusion martix all



conf_all = [log_conf, knn_conf, svm_conf, nb_conf, dtree_conf, rtree_conf]

class_all = [log_class, knn_class, svm_class, nb_class, dtree_class, rtree_class]

class_names = ['Logistic regression', 'K Nearest Neighbours', 'Support Vector Machine', 'Naive Bayes'

              , 'Decision tree', 'Random forest']
plt.figure(figsize=(15,8))

plt.suptitle("Confusion Matrixes",fontsize=22)



plt.subplot(2,3,1)

plt.title('Logistic Regression Confusion Matrix', fontsize=15)

sns.heatmap(conf_all[0], annot=True, cbar=False, annot_kws={'size':15}, cmap='PuBu')



plt.subplot(2,3,2)

plt.title('K Nearest Neighbours Confusion Matrix', fontsize=15)

sns.heatmap(conf_all[1], annot=True, cbar=False, annot_kws={'size':15}, cmap='PuBu')



plt.subplot(2,3,3)

plt.title('Support Vector Machine Confusion Matrix', fontsize=15)

sns.heatmap(conf_all[2], annot=True, cbar=False, annot_kws={'size':15}, cmap='PuBu')



plt.subplot(2,3,4)

plt.title('Naive Bayes Confusion Matrix', fontsize=15)

sns.heatmap(conf_all[3], annot=True, cbar=False, annot_kws={'size':15}, cmap='PuBu')



plt.subplot(2,3,5)

plt.title('Decission tree Confusion Matrix', fontsize=15)

sns.heatmap(conf_all[4], annot=True, cbar=False, annot_kws={'size':15}, cmap='PuBu')



plt.subplot(2,3,6)

plt.title('Random forestrs Confusion Matrix', fontsize=15)

sns.heatmap(conf_all[5], annot=True, cbar=False, annot_kws={'size':15}, cmap='PuBu')
for name, cls in zip(class_names, class_all):

    print('Classification report for %s' %name)

    print(cls)

    print('\n')