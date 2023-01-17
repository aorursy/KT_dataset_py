import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv('../input/mushrooms.csv')
df.shape
df.head()
df.columns
df.isnull().sum()
df.duplicated().sum()
df.nunique()
df['class'].unique()
df.describe()
df['class'].value_counts()
%matplotlib inline

items=pd.DataFrame(df['class'].value_counts())

items.plot(kind='bar', figsize=(4,6), width=0.3, color=[('#63d363', '#d36363')], legend=False)

plt.title("Number of Edible and Poisonous Mushrooms in this Dataset", fontsize="15")

plt.xlabel("Edible or Poisonous", fontsize="12")

plt.ylabel("Number of Mushrooms", fontsize="12")

plt.xticks(np.arange(2),("Edible", "Poisonous"), rotation=0)

plt.grid()   

plt.show()
df['cap-color'].value_counts()
caps=pd.DataFrame(df['cap-color'].value_counts())

caps.plot(kind='bar', figsize=(8,8), width=0.8, color=[('#bf7050', '#A9A9A9', '#d36363', '#f3f6c3', '#DCDCDC', '#bfa850', '#f9d7f7', '#D2691E', '#63d363', '#7050bf')], legend=False)

plt.xlabel("Cap Color",fontsize=12)

plt.ylabel('Number of Mushrooms',fontsize=12)

plt.title('Mushroom Cap Color Types in the Dataset', fontsize=15)

plt.xticks(np.arange(10),('Brown', 'Gray','Red','Yellow','White','Buff','Pink','Cinnamon', 'Green','Purple'))

plt.grid()       

plt.show() 
df['cap-shape'].value_counts()
capsh=pd.DataFrame(df['cap-shape'].value_counts())

capsh.plot(kind='bar', figsize=(8,8), width=0.5, color=[('#A9A9A9')], legend=False)

plt.xlabel("Cap Shape",fontsize=12)

plt.ylabel('Number of Mushrooms',fontsize=12)

plt.title('Mushroom Cap Types in the Dataset', fontsize=15)

plt.xticks(np.arange(6),('Convex', 'Flat','Knobbed','Bell','Sunken','Conical'))

plt.grid()       

plt.show() 
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
# Encodes labels from 0 to n_classes-1

labelEncoder = preprocessing.LabelEncoder()

for col in df.columns:

    df[col] = labelEncoder.fit_transform(df[col])
df.head()
df['class'].value_counts()
# 75% train, 25% test

train, test = train_test_split(df, test_size = 0.25) 

y_train = train['class']

X_train = train[[x for x in train.columns if 'class' not in x]]

y_test = test['class']

X_test = test[[x for x in test.columns if 'class' not in x]]



from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize the training and test data 

vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
from sklearn.naive_bayes import MultinomialNB

# Creating a MultinomialNB classifier and fit the model

cl = MultinomialNB()

cl.fit(X_train, y_train)
y_pred=cl.predict(X_test)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score



print("Accuracy score: ", accuracy_score(y_test, y_pred))

print("Recall score: ", recall_score(y_test, y_pred, average = 'weighted'))

print("Precision score: ", precision_score(y_test, y_pred, average = 'weighted'))

print("F1 score: ", f1_score(y_test, y_pred, average = 'weighted'))
from sklearn import model_selection

from sklearn.model_selection import cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=7)

scoring = 'accuracy'

results = model_selection.cross_val_score(cl, X_train, y_train, cv=kfold, scoring=scoring)

print("Cross validation average accuracy with 10-folds: %f" % (results.mean()))
from sklearn.metrics import confusion_matrix

import itertools
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

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
cm = confusion_matrix(y_test, y_pred)

plt.figure()

plot_confusion_matrix(cm, classes=['p','e'], title='Confusion matrix, without normalization')
plt.figure()

plot_confusion_matrix(cm, classes=['p','e'], normalize=True, title='Confusion matrix, with normalization')
from sklearn import svm
clf = svm.SVC(gamma='auto')

clf.fit(X_train, y_train) 
y_pred=clf.predict(X_test)
print("Accuracy score: ", accuracy_score(y_test, y_pred))

print("Recall score: ", recall_score(y_test, y_pred, average = 'weighted'))

print("Precision score: ", precision_score(y_test, y_pred, average = 'weighted'))

print("F1 score: ", f1_score(y_test, y_pred, average = 'weighted'))
from sklearn import model_selection

from sklearn.model_selection import cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=7)

scoring = 'accuracy'

results = model_selection.cross_val_score(clf, X_train, y_train, cv=kfold, scoring=scoring)

print("Cross validation average accuracy with 10-folds: %.3f" % (results.mean()))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()

plot_confusion_matrix(cm, classes=['p','e'], title='Confusion matrix, without normalization')
plt.figure()

plot_confusion_matrix(cm, classes=['p','e'], normalize=True, title='Confusion matrix, with normalization')
from sklearn import linear_model, datasets

logreg = linear_model.LogisticRegression(solver='lbfgs',max_iter=2000)
logreg.fit(X_train, y_train)
y_pred=logreg.predict(X_test)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score



print("Accuracy score: ", accuracy_score(y_test, y_pred))

print("Recall score: ", recall_score(y_test, y_pred, average = 'weighted'))

print("Precision score: ", precision_score(y_test, y_pred, average = 'weighted'))

print("F1 score: ", f1_score(y_test, y_pred, average = 'weighted'))
from sklearn import model_selection

from sklearn.model_selection import cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=7)

scoring = 'accuracy'

results = model_selection.cross_val_score(logreg, X_train, y_train, cv=kfold, scoring=scoring)

print("Cross validation average accuracy with 10-folds: %.3f" % (results.mean()))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure()

plot_confusion_matrix(cm, classes=['p','e'], title='Confusion matrix, without normalization')
plt.figure()

plot_confusion_matrix(cm, classes=['p','e'], normalize=True, title='Confusion matrix, with normalization')