import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
spam_df=pd.read_csv("/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv")

spam_df.head()
X_text=spam_df["text"]

type(X_text)

X_text.head()
y_label=spam_df["label"]

y_label.head()
# Visialization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

# Barplot describes the count of the class labels

plt.figure(figsize = (12, 6))

sns.countplot(data = spam_df, x = 'label');
y_label[y_label.isnull()].count()
X_text_train, X_text_test, y_label_train, y_label_test = train_test_split(X_text,

                                                      y_label, 

                                                    test_size=0.33,

                                                    random_state=53)

(X_text_train.shape),(X_text_test.shape),(y_label_train.shape),(y_label_test.shape)
tfIdfVecorizer=TfidfVectorizer(stop_words='english')

tfIdfVecorizer
count_train=tfIdfVecorizer.fit_transform(X_text_train)

count_train
tfIdfVecorizer.get_feature_names()[0:10]
len(tfIdfVecorizer.get_stop_words())
tfIdfVecorizer.get_stop_words()
count_test=tfIdfVecorizer.transform(X_text_test)

count_test
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model
model.fit(count_train,y_label_train)

label_pred = model.predict(count_test)
score=accuracy_score(y_label_test,label_pred)

score
con=confusion_matrix(y_label_test,label_pred)

con
navie_classifier=MultinomialNB()

##Fit the classifier to the training data

navie_classifier.fit(count_train,y_label_train)
## predict the data

label_pred=navie_classifier.predict(count_test)

label_pred
score=accuracy_score(y_label_test,label_pred)

score
con=confusion_matrix(y_label_test,label_pred)

con
print(classification_report(y_label_test,label_pred))

cmat = confusion_matrix(y_label_test, label_pred)

plt.figure(figsize = (6, 6))

sns.heatmap(cmat, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);
from joblib import dump, load

dump(model, 'model.joblib')

dump(tfIdfVecorizer, 'tfIdfVecorizer.joblib')
model = load('model.joblib')

vect = load('tfIdfVecorizer.joblib')
count_test=vect.transform(X_text_test)

label_pred=model.predict(count_test)

score=accuracy_score(y_label_test,label_pred)

score
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier()

]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(count_train,y_label_train)

    name = clf.__class__.__name__

    train_predictions = clf.predict(count_test)

    acc = accuracy_score(y_label_test, train_predictions)

   

       

    log_entry = pd.DataFrame([[name, acc*100]], columns=log_cols)

    log = log.append(log_entry)
sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()
input=spam_df.iloc[3:4,:]

input
decisionTreeModel = load('model.joblib')

tfIdfVecorizer = load('tfIdfVecorizer.joblib')

count_test=tfIdfVecorizer.transform(input['text'])

# count_test=tfIdfVecorizer.transform(['I am input'])

## predict the data

label_pred=decisionTreeModel.predict(count_test)

if label_pred[0]=="spam":

  print("This is a spam email")

else:

  print("This is a ham email")