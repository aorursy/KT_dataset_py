import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv("../input/indian-liver-patient-records/indian_liver_patient.csv")
data.isnull().sum()
data = data.dropna()

data.isnull().any()
fig, ax = plt.subplots(figsize=(7,7))

sns.heatmap(abs(data.corr()), annot=True, square=True, cbar=False, ax=ax, linewidths=0.25);
data = data.drop(columns= ['Direct_Bilirubin', 'Alamine_Aminotransferase', 'Total_Protiens'])
data['Dataset'] = data['Dataset'].replace(1,0)

data['Dataset'] = data['Dataset'].replace(2,1)
print('How many people to have disease:', '\n', data.groupby('Gender')[['Dataset']].sum(), '\n')

print('How many people participated in the study:', '\n', data.groupby('Gender')[['Dataset']].count())
print('Percentage of people with the disease depending on gender:')

data.groupby('Gender')[['Dataset']].sum()/ data.groupby('Gender')[['Dataset']].count()
age=pd.cut(data['Age'], [0,18,91])

print('Distribution of the disease by gender and age')

data.pivot_table('Dataset', ['Gender', age])
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
X = data[['Age', 'Gender', 'Total_Bilirubin','Alkaline_Phosphotase','Aspartate_Aminotransferase','Albumin','Albumin_and_Globulin_Ratio']]

y = pd.Series(data['Dataset'])
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X['Gender'] = labelencoder.fit_transform(X['Gender'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(x_train)

X_test = scaler.transform(x_test)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from numpy import mean
ABC = AdaBoostClassifier()
def get_models():

        models = dict()

        models['10'] = AdaBoostClassifier(n_estimators=10)

        models['30'] = AdaBoostClassifier(n_estimators=30)

        models['50'] = AdaBoostClassifier(n_estimators=50)

        models['75'] = AdaBoostClassifier(n_estimators=75)

        models['100'] = AdaBoostClassifier(n_estimators=100)

        models['125'] = AdaBoostClassifier(n_estimators=125)

        models['150'] = AdaBoostClassifier(n_estimators=150)

        return models

 

def evaluate_model(model):

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

        return scores



models = get_models()



results, names = list(), list()

for name, model in models.items():

        scores = evaluate_model(model)

        results.append(scores)

        names.append(name)

        print('>%s %.3f' % (name, mean(scores)))



plt.boxplot(results, labels=names, showmeans=True)

plt.title('n_estimators')

plt.show()    
def get_models():

        models = dict()

        for i in range(1,15):

            models[str(i)] = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=i))

        return models

 

def evaluate_model(model):

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

        return scores

 

models = get_models()



results, names = list(), list()

for name, model in models.items():

        scores = evaluate_model(model)

        results.append(scores)

        names.append(name)

        print('>%s %.3f' % (name, mean(scores)))



plt.boxplot(results, labels=names, showmeans=True)

plt.title('max_depth')

plt.show()
def get_models():

        models = dict()

        for i in range(1, 21, 1):

            per = i/10

            key = '%.3f' % per

            models[key] = AdaBoostClassifier(learning_rate=per)

        return models

 

def evaluate_model(model):

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

        return scores

 

models = get_models()



results, names = list(), list()

for name, model in models.items():

        scores = evaluate_model(model)

        results.append(scores)

        names.append(name)

        print('>%s %.3f' % (name, mean(scores)))



plt.boxplot(results, labels=names, showmeans=True)

plt.xticks(rotation=45)

plt.title('Learning rate')

plt.show()
ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),

                         n_estimators=125,

                        learning_rate = 0.6,

                        random_state=42)



ABC.fit(X_train, y_train)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(ABC, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

('Accuracy: %.3f' % (mean(n_scores)))
labels = ABC.predict(X_test)

matrix = confusion_matrix(y_test, labels)

sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');
logit_roc_auc = roc_auc_score(y_test, labels)

fpr, tpr, thresholds = roc_curve(y_test, ABC.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
print(classification_report(y_test, labels))