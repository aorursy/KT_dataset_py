import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')
data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
data.head()
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace=True)

data.rename(columns={'v1':'Label', 'v2':'Message',}, inplace=True)
data.head()
sns.countplot('Label', data = data)
data.groupby('Label').describe()
X = data['Message']

y = data['Label']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
X_train = tfidf_vect.fit_transform(X_train)

X_test = tfidf_vect.transform(X_test)
len(tfidf_vect.get_feature_names())
# tfidf_vect.get_feature_names()
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier







lr = LogisticRegression()

nb = MultinomialNB()

knc = KNeighborsClassifier()

svc = SVC(gamma = 'auto')

dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier(n_estimators=100)

gbc = GradientBoostingClassifier()

abc = AdaBoostClassifier()







models = {'Logistic Regression':lr, 'Naive Bayes classifier':nb, 'k-nearest neighbors':knc, 

          'Support Vector Machine':svc, 'Decision Tree Classifier':dtc, 

          'Random Forest Classifier':rfc, 'Gradient Boosting Classifier':gbc, 'AdaBoost Classifier':abc}
def eval_model(model):

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    test_accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['ham', 'spam'], index=['ham','spam'])

    

    return test_accuracy, conf_matrix
test_accuracies = []

confusion_matrices = []

for name, model in models.items():

    test_acc, conf_matrix = eval_model(model) 

    test_accuracies.append(test_acc)

    confusion_matrices.append(conf_matrix)

    print(f'{name} ---> Test accuracy - {test_acc*100:.2f}%')
results = pd.DataFrame(test_accuracies, index=list(models.keys()), columns=['test_acc'])

results
plt.figure(figsize=(10, 6))

sns.barplot(x ='test_acc', y=results.index, data=results)

plt.xlim(0.85, 1.0)

plt.title('Performance comparision')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)



tfidf_vect = TfidfVectorizer(stop_words='english')



X_train = tfidf_vect.fit_transform(X_train)

X_test = tfidf_vect.transform(X_test)
len(tfidf_vect.get_feature_names())
lr = LogisticRegression()

nb = MultinomialNB()

knc = KNeighborsClassifier()

svc = SVC(gamma = 'auto')

dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier(n_estimators=100)

gbc = GradientBoostingClassifier()

abc = AdaBoostClassifier()







models = {'Logistic Regression':lr, 'Naive Bayes classifier':nb, 'k-nearest neighbors':knc, 

          'Support Vector Machine':svc, 'Decision Tree Classifier':dtc, 

          'Random Forest Classifier':rfc, 'Gradient Boosting Classifier':gbc, 'AdaBoost Classifier':abc}
test_accuracies_no_stopwords = []

confusion_matrices_no_stopwords = []

for name, model in models.items():

    test_acc, conf_matrix = eval_model(model) 

    test_accuracies_no_stopwords.append(test_acc)

    confusion_matrices_no_stopwords.append(conf_matrix)

    print(f'{name} ---> Test accuracy - {test_acc*100:.2f}%')
results['test_acc_without_stopwords'] = pd.Series(test_accuracies_no_stopwords, index=list(models.keys()))

results
def plot_confusion_matrices(models, confusion_matrices):

    fig, axs = plt.subplots(2,4, figsize=(10,5)) 



    m = 0

    for i, ax_r in enumerate(axs):

        for j, ax in enumerate(ax_r):

            sns.heatmap(confusion_matrices[m], annot=True, cbar=False, cmap='Blues', fmt='g', ax = ax)

            ax.set_xlabel('Predicted label')

            ax.set_ylabel('True label')

            ax.set_title(f'{list(models.keys())[m]}', fontsize=12, fontweight='bold')

            m += 1



    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                        wspace=0.35)

    plt.tight_layout()

    plt.show()

    
plot_confusion_matrices(models, confusion_matrices)
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('tfidf_vect', TfidfVectorizer()),

    ('classifier', RandomForestClassifier(n_estimators=100))

])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
test_acc, conf_matrix = eval_model(pipeline) 



print('Test accuracy - ',test_acc)

print('Confusion matrix - \n', conf_matrix)
print('Classification Report \n', classification_report(y_test, pipeline.predict(X_test)))
messages = ['Thank you for subscribing! You will be notified when you win your 1 Million Dollar prize money! Please call our customer service representative on 0800012345 for further details ',

          'Hi, hope you are doing well. Please call me as soon as possible!']
pipeline.predict(messages)