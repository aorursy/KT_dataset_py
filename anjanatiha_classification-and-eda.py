# System

import os

import sys



# Numerical

import numpy as np

from numpy import median

import pandas as pd





# NLP

import re

from string import ascii_letters





# Tools

import itertools



# Machine Learning - Preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler





# Machine Learning - Model Selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV





# Machine Learning - Models

from sklearn import svm

from sklearn.svm import SVC

from sklearn.svm import LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomTreesEmbedding, RandomForestClassifier, VotingClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB 

from sklearn.neighbors import KDTree, KNeighborsClassifier, NearestNeighbors

from sklearn.neural_network import BernoulliRBM, MLPClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.mixture import GaussianMixture





# Machine Learning - Evaluation

from sklearn import metrics 

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

from sklearn.utils import class_weight





# Plot

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns



print(os.listdir("../input"))

# import train 7 test data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submission = pd.read_csv("../input/gender_submission.csv")
train.head()
train.describe()
train.info()
df = train

columns = df.columns
target = "Survived"

y_label = "Count"





nrows = 6

ncols = 22



labelsize = ncols*0.8

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.set(style="whitegrid")

sns.set_context("notebook")



ax = sns.countplot(x=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(target + " Count", fontsize=titlesize)

ax.set_xlabel(target, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)
col = "Sex"

target = "Survived"

y_label = "Count"





nrows = 6

ncols = 22



labelsize = ncols*0.8

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.set(style="whitegrid")

sns.set_context("notebook")



ax = sns.countplot(x=col, hue=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+target +" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)
col = "Age"

target = "Survived"

y_label = "Count"



nrows = 8

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



ax = sns.countplot(x=col, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize, rotation=90)

plt.show()



ax = sns.countplot(x=col, hue=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize, rotation=90)

plt.show()
col = "Pclass"

target = "Survived"

y_label = "Count"



nrows = 6

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



plt.subplot(121)

ax = sns.countplot(x=col, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)



plt.subplot(122)

ax = sns.countplot(x=col, hue=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Count", fontsize=titlesize)

ax.set_xlabel(target, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()



ax = sns.lineplot(x=col, y=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Line Plot", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()
col = "SibSp"

target = "Survived"

y_label = "Count"





nrows = 6

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



plt.subplot(121)

ax = sns.countplot(x=col, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)



plt.subplot(122)

ax = sns.countplot(x=col, hue=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()





ax = sns.lineplot(x=col, y=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Line Plot", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()
col = "Parch"

target = "Survived"

y_label = "Count"



nrows = 6

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



plt.subplot(121)

ax = sns.countplot(x=col, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)



plt.subplot(122)

ax = sns.countplot(x=col, hue=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()



ax = sns.lineplot(x=col, y=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Line Plot", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()
col = "Embarked"

target = "Survived"

y_label = "Count"







nrows = 6

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



plt.subplot(121)

ax = sns.countplot(x=col, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(target, fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)



plt.subplot(122)

ax = sns.countplot(x=col, hue=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target, fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()



ax = sns.lineplot(x=col, y=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Line Plot", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()
col1 = "Age"

col2 = "Fare"

col3 = "Name"



target = "Survived"





nrows = 12

ncols = 22



labelsize = ncols*.5

fontsize = labelsize*1.5

titlesize = labelsize*2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



plt.subplots_adjust(hspace = 0.4)



df[col1] = df[col1].fillna(df[col1].median())

plt.subplot(221)

ax = sns.distplot(df[col1])

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col1+" Distribution", fontsize=titlesize)

ax.set_xlabel(col1, fontsize=fontsize)

ax.set_ylabel("", fontsize=fontsize)

ax.tick_params(labelsize=labelsize)



plt.subplot(222)

ax = sns.distplot(df[col2])

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col2+" Distribution", fontsize=titlesize)

ax.set_xlabel(col2, fontsize=fontsize)

ax.set_ylabel("", fontsize=fontsize)

ax.tick_params(labelsize=labelsize)



plt.subplot(223)

ax = sns.distplot(df[col3].apply(lambda x:len(x)))

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col3+" Length"+" Distribution", fontsize=titlesize)

ax.set_xlabel(col3, fontsize=fontsize)

ax.set_ylabel("", fontsize=fontsize)

ax.tick_params(labelsize=labelsize)
col1 = "Name"

col2 = "Title"

target = "Survived"

y_label = "Count"





nrows = 6

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



df[col2] = df[col1].apply(lambda x:x.split(" ")[1])

ax = sns.countplot(x=col2, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col2+" Count", fontsize=titlesize)

ax.set_xlabel(col1, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize, rotation=90)

plt.show()



ax = sns.countplot(x=col2, hue=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col2+" vs. "+ target+" Count", fontsize=titlesize)

ax.set_xlabel(col2, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize, rotation=90)

plt.show()
col1 = "Name"

col2 = "Name Length"

target = "Survived"

y_label = "Count"





nrows = 6

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



df[col2] = df[col1].apply(lambda x:len(x))

ax = sns.countplot(x=col2, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col2+" Count", fontsize=titlesize)

ax.set_xlabel(col2, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()



ax = sns.countplot(x=col2, hue=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col2+" vs. "+ target+" Count", fontsize=titlesize)

ax.set_xlabel(col2, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()





ax = sns.lineplot(x=col2, y=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col2+" vs. "+ target+" Line Plot", fontsize=titlesize)

ax.set_xlabel(col2, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()



ax = sns.regplot(x=col2, y=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col2+" vs. "+ target+" Line Plot", fontsize=titlesize)

ax.set_xlabel(col2, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()

col = "Fare"

target = "Survived"

y_label = "Count"





nrows = 6

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.6

titlesize = labelsize*2.2



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



ax = sns.distplot(df[col])

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize, rotation=90)

plt.show()





ax = sns.lineplot(x=col2, y=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col2+" vs. "+ target+" Line Plot", fontsize=titlesize)

ax.set_xlabel(col2, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)

plt.show()



ax = sns.regplot(x=col, y=target, data=df)

ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title(col+" vs. "+ target+" Count", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize, rotation=90)

plt.show()
def transform(x):

    if x==0: return 0

    else: return 1

    

def prep(train, title=None, ticket=None):

    m_embarked = {"S": 0, "Q": 1, "C": 2}



    train["NameLen"] = train["Name"].apply(lambda x: len(x))

    train["NameCount"] = train["Name"].apply(lambda x:len(x.split(" ")))

    train["Title"] = train["Name"].apply(lambda x:x.split(" ")[1])

    

    if not title:

        title = {}

    for item, value in enumerate(set(train["Title"])):

        title[value] = item



    train["Title"] = train["Title"].map(title)





    train['FamilySize'] = train ['SibSp'] + train['Parch'] + 1



    train['IsAlone'] = 1 

    train['IsAlone'].loc[train['FamilySize'] > 1] = 0 



    

    train["Sex"] = train["Sex"].apply(lambda x: 0 if x=="male" else 1)

    

    train['Age'].fillna(train['Age'].median(), inplace = True)

    train['Age'] = train['Age'].apply(lambda x: int(x/5))

    

#     train['AgeBin'] = pd.cut(train['Age'].astype(int), 5)

    

#     if age_bin==None:

#         age_bin = {}

#         for item, value in enumerate(set(train['AgeBin'])):

#             age_bin[value] = item

    

       

#     train['AgeBin'] = train['AgeBin'].map(age_bin)

    

    train['SibSp'] = train['SibSp'].apply(lambda x: transform(x))

    train['Parch'] = train['Parch'].apply(lambda x: transform(x))

    

    train['Fare'].fillna(train['Fare'].median(), inplace = True)

#     train['FareBin'] = pd.qcut(train['Fare'], 4)

    

#     if not fare_bin:

#         fare_bin = {}

#         for item, value in enumerate(set(train['FareBin']), 1):

#             fare_bin[value] = item

#     train['FareBin'] = train['FareBin'].map(fare_bin)

    

    emb = train['Embarked'].mode()[0]

    train['Embarked'] = train['Embarked'].apply(lambda x: x if x in m_embarked else emb)

    train['Embarked'] = train['Embarked'].map(m_embarked)

    

    

    train["TicketType"] = train["Ticket"].apply(lambda x: x.split(" ")[0])

    train["TicketType"] = train["TicketType"].apply(lambda x: 0 if x.isdigit() else re.sub(r"[^a-zA-Z]", "", x.split("/")[0]))

    

    if not ticket:

        ticket = {}

    for item, value in enumerate(set(train["TicketType"])):

        ticket[value] = item

    train["TicketType"] = train["TicketType"].map(ticket)

    

    

    if title and ticket:

        return train, title, ticket

    else:

        return train

train, title, ticket = prep(train)

test, title, ticket = prep(test, title=title, ticket=ticket)
all_columns = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','FamilySize', 'IsAlone']

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'NameLen', 'NameCount', 'Title', 'FamilySize', 'IsAlone', 'TicketType']

drop_cols = ["PassengerId", "Survived", "Name", "Ticket", "Cabin"]

target_col = 'Survived'



# categorical_data = train.select_dtypes("object_")

# numerical_data = train.select_dtypes("number")

cols = [columns+[target_col]][0]

d = train[cols]

sns.pairplot(d, kind='reg', diag_kind='kde', dropna=True)
# g = sns.PairGrid(

#     d, 

#     diag_sharey=True, 

#     height=2.5, 

#     aspect=1, 

#     despine=True, 

#     dropna=False)

# g = g.map(plt.scatter)

# g.map_diag(plt.hist)

# g.map_offdiag(plt.scatter);
sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



nrows = 22

ncols = 22



cmap = sns.diverging_palette(h_neg=220, h_pos=10, s=75, l=50, sep=10, n=len(columns), center='light', as_cmap=True)



sns.clustermap(d.corr(),  

               figsize=(ncols, nrows), 

               center=0,

               cmap=cmap,

               linewidths=.75)



plt.show()
nrows = 22

ncols = 22



labelsize = ncols*.7

fontsize = labelsize*1.5

titlesize = labelsize*2.5



sns.reset_defaults()

sns.set(style="whitegrid")

sns.set_context("notebook")



d =train[all_columns]



# Compute the correlation matrix

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

fig, ax = plt.subplots(figsize=(ncols, nrows))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(h_neg=220, h_pos=10, s=75, l=50, sep=10, n=len(columns), center='light', as_cmap=True)



ax = sns.heatmap(

    corr,

    cmap=cmap,

    center=0,

    robust=True,

    annot=True,

    linewidths=0.5,

    linecolor='white',

    cbar=True,

    cbar_kws={"shrink": .5},

    square=True,

    mask=mask)



ax.figure.set_size_inches(ncols, nrows)

ax.axes.set_title("Feature Correleation", fontsize=titlesize)

ax.set_xlabel(col, fontsize=fontsize)

ax.set_ylabel(y_label, fontsize=fontsize)

ax.tick_params(labelsize=labelsize)
def print_performance(model, X_test, y_test, class_names):

    preds = model.predict(X_test)



    # accuracy_score = metrics.accuracy_score(y_test, preds)

    # auc = metrics.auc(y_test, preds)

    # average_precision_score = metrics.average_precision_score(y_test, preds)

    classification_report = metrics.classification_report(y_test, preds)

    # cohen_kappa_score = metrics.cohen_kappa_score(y_test, preds)

    confusion_matrix = metrics.confusion_matrix(y_test, preds)

    f1_score_ = metrics.f1_score(y_test, preds, average="weighted")

    # precision_recall_curve = metrics.precision_recall_curve(y_test, preds)

    # precision_score = metrics.precision_score(y_test, preds, average="weighted")

    # recall_score = metrics.recall_score(y_test, preds, average="weighted")

    # roc_auc_score = metrics.roc_auc_score(y_test, preds, average="weighted")

    # roc_curve = metrics.roc_curve(y_test, preds)

    

    print("-"*55)

    print("Performance")

    print("-"*55)

    print("{} : {:.4f} ".format("Accuracy Score                  ", accuracy_score))

    # print("{} : {:.4f} ".format("AUC                             ", auc))

    # print("{} : {:.4f} ".format("Average Precision Score         ", average_precision_score))

#     print("{} : {:.4f} ".format("Classification Report           ", classification_report))

#     print("{} : {:.4f} ".format("Confusion Matrix                ", confusion_matrix))

    print("{} : {:.4f} ".format("F1 Score                        ", f1_score_))

    # print("{} : {:.4f} ".format("Precision Recall Curve          ", precision_recall_curve))

    print("{} : {:.4f} ".format("Precision Score                 ", precision_score))

    print("{} : {:.4f} ".format("Recall Score                    ", recall_score))

    # print("{} : {:.4f} ".format("Roc Auc Score                   ", roc_auc_score))

    # print("{} : {:.4f} ".format("Roc Curve                       ", roc_curve))

    print(classification_report)

    

    print("-"*55)

    print("\n\n")

    



    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.subplot(121)

    plot_confusion_matrix(confusion_matrix, classes=class_names, title='Confusion matrix, without normalization')



    # Plot normalized confusion matrix

    plt.subplot(122)

    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')



    plt.show()

    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    

    

def print_performance_grid(clf):

    # print("*"*100)

    # print("{}{}{}".format("*"*40,"Performance", "*"*40))

    print("{}".format("Performance"))

    print("*"*100)

    print("Score            : {}".format(clf.score(X, y)))

    print("Best Estimator   : {}".format(clf.best_estimator_))

    print("Best Score       : {}".format(clf.best_score_))

    print("Best Params      : {}".format(clf.best_params_))

    print("Best Index       : {}".format(clf.best_index_))

    print("Scorer           : {}".format(clf.scorer_))

    print("Refit Time       : {}".format(clf.refit_time_))

    # print("CV Results       : {}".format(clf.cv_results_))



    params = clf.get_params()

    best_estimator = clf.best_estimator_

    cv_results = clf.cv_results_

    

    return params, best_estimator, cv_results
train_features = train[columns]

test_features = test[columns]





target = train[target_col]
train_features.head()
train_features = train_features.values

test_features = test_features.values



target = target.values
# train_features = train_features.fillna(0)



# test_features = test_features.fillna(0)



scaler = StandardScaler()



train_features = scaler.fit_transform(train_features)

test_features = scaler.transform(test_features)
train_features[0], test_features[0]
X = train_features

y = target
# # SVC

# estimator = svm.SVC(class_weight='balanced')



# kernel = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')

# degree = np.arange(1, 10, 1)

# C = np.array([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1), np.arange(1, 10, 1)]).flatten()

# gamma = np.array([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1), np.arange(1, 10, 1)]).flatten()





# param_grid = {

#     'kernel': ['linear'], 

#     'C': C,

#     'gamma': gamma

# }



# # RandomForestClassifier

# # estimator = RandomForestClassifier(n_estimators=100, class_weight="balanced")



# # param_grid = {"n_estimators": np.arange(1, 100, 1)}





# cv = 3

# verbose = 1



# grid_clf = GridSearchCV(estimator=estimator,param_grid=param_grid, n_jobs=-1, cv=cv, verbose=verbose)



# grid_clf.fit(X, y)
# params, best_estimator, cv_results = print_performance_grid(grid_clf)
clf = RandomForestClassifier(n_estimators=1000, class_weight="balanced")

# clf = SVC(C=0.04, gamma=0.01, kernel='linear', class_weight="balanced")



cv = 3

verbose = 1

scores = cross_val_score(clf, X, y, cv=cv, verbose=verbose)



print(scores)
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")

clf = clf.fit(X, y)

preds = clf.predict(test_features)
submission_csv = "submission.csv"



df_test = pd.DataFrame({'PassengerId':submission['PassengerId'], 'Survived':preds})



df_test.to_csv(submission_csv, index=False)

df_test.head()