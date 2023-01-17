import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

import warnings



#Setting the properties to personal preference

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

plt.rcParams["figure.figsize"] = (8,4)

warnings.filterwarnings("ignore")    # No major warnings came out on first run. So, I am ignoring the "deprecation" warnings instead of showing them the first time to keep the code clean



print(os.listdir("../input"))

df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.columns
df.drop('Serial No.', inplace=True, axis=1)

df.rename({'Chance of Admit ': 'Chance of Admit', 'LOR ':'LOR'}, axis=1, inplace=True)
df.tail()
df.info()
df.describe()
for rating in sorted(df['University Rating'].unique()):

    print("For University Rating: ", rating, "\n")

    print(df[df['University Rating']==rating].describe(), 2*"\n")
for rating in sorted(df['University Rating'].unique()):

    sns.jointplot(data=df[df['University Rating']==rating], x = 'GRE Score', y = 'Chance of Admit')

    print("Jointplot for the University Rating: ", rating)

    plt.show()
for rating in sorted(df['University Rating'].unique()):

    sns.distplot(df[df['University Rating']==rating]['GRE Score'], hist=False)

plt.show()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(), annot=True)
sns.pairplot(df)
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

y = df['Chance of Admit']

X = df[features]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_train_label = [1 if each > 0.8 else 0 for each in y_train]

y_test_label  = [1 if each > 0.8 else 0 for each in y_test]
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC  

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report





# Create a pipeline

pipe = Pipeline([('classifier', LogisticRegression())])



# Create space of candidate learning algorithms and their hyperparameters

search_space = [{'classifier': [LogisticRegression()]},

        {'classifier': [SVC()]},

        {'classifier': [KNeighborsClassifier(n_neighbors=5)]}]



# Create grid search 

clf = GridSearchCV(pipe, search_space)



# Fit grid search

best_model = clf.fit(X_train, y_train_label)

# View best model

print(best_model.best_estimator_.get_params()['classifier'], "\n")

print("Accuracy of our best model is", clf.score(X_test, y_test_label)*100, "%", "\n")

print("Classification Report:", "\n", classification_report(y_test_label, best_model.predict(X_test)))
lg = LogisticRegression()



lg.fit(X_train, y_train_label)

predictions = lg.predict(X_test)



from sklearn.metrics import classification_report

print(classification_report(y_test_label, predictions))
svmmodel = SVC()

svmmodel.fit(X_train,y_train_label)

y_pred_svm = svmmodel.predict(X_test)



print(classification_report(y_test_label, y_pred_svm))
knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train,y_train_label)

y_pred_knn = knn.predict(X_test)



print(classification_report(y_test_label, y_pred_knn))