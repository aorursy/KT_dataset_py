import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# wordcloud

from wordcloud import WordCloud, STOPWORDS



# machine learning

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn import preprocessing



%matplotlib inline



plt.style.use('seaborn-dark-palette')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%time data = pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2004.csv')
missing = (data.isnull().sum())
type(missing[missing>0])
data
data.describe()
data.info()
data.shape
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    missing_data['Types'] = types

    return(np.transpose(missing_data))
%%time

missing_data(data)
data['Winner'].value_counts()


plt.figure(figsize=(10,6))

sns.countplot('Winner', data=data, palette='Set3')

plt.xticks(rotation=90)

plt.title('Winner Count',fontsize=20)

plt.ylabel('Count',fontsize=16)

plt.xlabel('Winner?',fontsize=16)
def build_wordcloud(df, title):

    wordcloud = WordCloud(

        background_color='black', 

        stopwords=set(STOPWORDS), 

        max_words=100, 

        max_font_size=40, 

        random_state=666

    ).generate(str(df))



    fig = plt.figure(1, figsize=(15,15))

    plt.axis('off')

    fig.suptitle(title, fontsize=16)

    fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
build_wordcloud(data['Candidate'], 'Prevalent words in Name for all dataset')
temp = data['Party'].value_counts().head(20)
plt.figure(figsize=(10,6))

sns.barplot(x=temp.index, y=temp.values, palette='Set3')

plt.xticks(rotation=90)

plt.title('Number of Seats Contested by PARTIES (TOP 20)',fontsize=20)

plt.ylabel('Number of Seats',fontsize=16)

plt.xlabel('Political Parties',fontsize=16)
data = data.fillna(0)
data.drop(['Candidate'], axis=1, inplace=True)
data
data["Party"] = data["Party"].astype("category")

data = pd.get_dummies(data, columns = ["Party"],prefix="Party")
data["Education"] = data["Education"].astype("category")

data = pd.get_dummies(data, columns = ["Education"],prefix="Education")
data["Constituency"] = data["Constituency"].astype("category")

data = pd.get_dummies(data, columns = ["Constituency"],prefix="Constituency")
data['Gender'] = data['Gender'].map({'M':1, 'F':0})
X = data.copy().drop('Winner', axis=1)

y = data['Winner']
X_scaled = preprocessing.scale(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3)

X_train.shape, Y_train.shape, X_test.shape
# k-nearest neighbor

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

knn_Y_pred = knn.predict(X_test)

knn_accuracy = knn.score(X_test, Y_test)

knn_accuracy
# creating confusion matrix heatmap



conf_mat = confusion_matrix(Y_test, knn_Y_pred)

fig = plt.figure(figsize=(10,7))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in

                conf_mat.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_mat, annot=labels, annot_kws={"size": 16}, fmt='')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# getting precision, recall and f1-score via classification report



print(classification_report(Y_test, knn_Y_pred))
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

decision_tree_Y_pred = decision_tree.predict(X_test)

decision_tree_accuracy = decision_tree.score(X_test, Y_test)

decision_tree_accuracy
# creating confusion matrix heatmap



conf_mat = confusion_matrix(Y_test, decision_tree_Y_pred)

fig = plt.figure(figsize=(10,7))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in

                conf_mat.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_mat, annot=labels, annot_kws={"size": 16}, fmt='')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# getting precision, recall and f1-score via classification report



print(classification_report(Y_test, decision_tree_Y_pred))
# Support Vector Machine

svc = SVC()

svc.fit(X_train, Y_train)

svm_Y_pred = svc.predict(X_test)

svc_accuracy = svc.score(X_test, Y_test)

svc_accuracy
# creating confusion matrix heatmap



conf_mat = confusion_matrix(Y_test, svm_Y_pred)

fig = plt.figure(figsize=(10,7))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in

                conf_mat.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_mat, annot=labels, annot_kws={"size": 16}, fmt='')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# getting precision, recall and f1-score via classification report



print(classification_report(Y_test, svm_Y_pred))
# Random Forest



random_forest = RandomForestClassifier(n_estimators=1000)

random_forest.fit(X_train, Y_train)

random_forest_Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

random_forest_accuracy = random_forest.score(X_test, Y_test)

random_forest_accuracy
# creating confusion matrix heatmap



conf_mat = confusion_matrix(Y_test, random_forest_Y_pred)

fig = plt.figure(figsize=(10,7))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in

                conf_mat.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_mat, annot=labels, annot_kws={"size": 16}, fmt='')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()


# getting precision, recall and f1-score via classification report



print(classification_report(Y_test, random_forest_Y_pred))
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

gaussian_Y_pred = gaussian.predict(X_test)

gaussian_accuracy = gaussian.score(X_test, Y_test)

gaussian_accuracy
# creating confusion matrix heatmap



conf_mat = confusion_matrix(Y_test, gaussian_Y_pred)

fig = plt.figure(figsize=(10,7))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in

                conf_mat.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_mat, annot=labels, annot_kws={"size": 16}, fmt='')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# getting precision, recall and f1-score via classification report



print(classification_report(Y_test, gaussian_Y_pred))
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

sgd_Y_pred = sgd.predict(X_test)

sgd_accuracy = sgd.score(X_test, Y_test)

sgd_accuracy
# creating confusion matrix heatmap



conf_mat = confusion_matrix(Y_test, sgd_Y_pred)

fig = plt.figure(figsize=(10,7))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in

                conf_mat.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_mat, annot=labels, annot_kws={"size": 16}, fmt='')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# getting precision, recall and f1-score via classification report



print(classification_report(Y_test, sgd_Y_pred))
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

linear_svc_Y_pred = linear_svc.predict(X_test)

linear_svc_accuracy = linear_svc.score(X_test, Y_test)

linear_svc_accuracy
# creating confusion matrix heatmap



conf_mat = confusion_matrix(Y_test, linear_svc_Y_pred)

fig = plt.figure(figsize=(10,7))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in

                conf_mat.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(conf_mat, annot=labels, annot_kws={"size": 16}, fmt='')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
# getting precision, recall and f1-score via classification report



print(classification_report(Y_test, linear_svc_Y_pred))
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Linear SVC', 'Decision Tree','Random Forest', 'Stochastic Gradient Descent', 'Gaussian Naive Bayes'],

    'Score': [svc_accuracy, knn_accuracy, linear_svc_accuracy, decision_tree_accuracy, random_forest_accuracy, sgd_accuracy, gaussian_accuracy]})

models.sort_values(by='Score', ascending=False)