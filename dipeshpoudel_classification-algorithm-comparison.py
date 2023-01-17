import numpy as np

import pandas as pd

import itertools

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

from  sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

plt.style.use('seaborn')

warnings.filterwarnings('ignore')
file_name = '../input/heart.csv'
data_df = pd.read_csv(file_name)
data_df.head()
pd.DataFrame(data_df.isna().sum(),columns=['null_count'])
for col in data_df.columns:

    fig, ax = plt.subplots(figsize=(20, 10))

    sns.distplot(data_df[col])

    plt.show()
data_df['target'].value_counts().plot(kind='bar')

plt.title("Target Frequency")

plt.xlabel("Target")

plt.ylabel("Count")

plt.show()
data_df.dtypes
dtype_map={"sex":"category","cp":"category","fbs":"category","restecg":"category","exang":"category",

           "slope":"category","ca":"category","thal":"category","target":"category"}
data_df = data_df.astype(dtype_map)
sns.pairplot(data_df)
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = data_df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
X = data_df.drop(columns=['target'])

y = data_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.25)
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
# prepare models

models = []

models.append(("LogisticRegression",LogisticRegression()))

models.append(("SVC",SVC()))

models.append(("LinearSVC",LinearSVC()))

models.append(("KNeighbors",KNeighborsClassifier()))

models.append(("DecisionTree",DecisionTreeClassifier()))

models.append(("RandomForest",RandomForestClassifier()))

rf2 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0, max_features=None)

models.append(("RandomForest2",rf2))

models.append(("MLPClassifier",MLPClassifier()))

# evaluate each model in turn

results = []

names = []

seed=0

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    cross_val_result = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print("Print the Corss Validation Result {}".format(name))

    print(cross_val_result)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    plot_confusion_matrix(cm=cm, classes=[0,1])

    acc_score = accuracy_score(y_test,y_pred)

    print("Accuracy Score of {} is {}".format(name,acc_score))