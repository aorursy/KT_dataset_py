import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sb

import matplotlib.pyplot as plt

from scipy import stats

%matplotlib inline

print(os.listdir("../input"))
heart = pd.read_csv('../input/heart.csv')

print("Shape of this data {} \n{} is no of row and {} is no of column."

      .format(heart.shape, heart.shape[0], heart.shape[1]))

heart.head()
heart.info()
#Check null value in this data 

print(list(heart.isnull().any()))
#counting target variable

sb.countplot(x='target', data=heart)

plt.xlabel("Target ({} for Non-disease, {} for disease)".format(0, 1))

plt.show()

heart.target.value_counts()
len_No_Disease, len_Disease = (len(heart[heart.target == 0]), len(heart[heart.target == 1]))

len_heart = heart.shape[0]

print("Patients which havn't heart disease: {:.2f}%".format(len_No_Disease/(len_heart)*100))

print("Patients which have heart disease: {:.2f}%".format(len_Disease/(len_heart)*100))
#Counting about sex

sb.countplot(x='sex', data=heart)

plt.xlabel("Sex - {} for female and {} for male".format(0, 1))

plt.show()
heart.hist(figsize=(18, 12))

plt.show()
col = ['age', 'sex', 'cp', 'fbs', 'restecg', 'trestbps',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']
plt.style.use('ggplot')

for item in col:

    pd.crosstab(heart[item], heart.target).plot(kind='bar', figsize=(15, 7))

    plt.title("{} with target".format(str(item)))

    plt.legend(["non disease", "disease"])

    plt.ylabel("Frequency")

plt.show()
plt.style.use('ggplot')

left_col = ['chol', 'trestbps', 'thalach']

for col in left_col:

    plt.figure(figsize=(12, 7))

    plt.title("{} with target".format(col))

    sb.boxplot(x=heart.target, y=heart[col])

plt.show()
plt.subplots(figsize=(20,7))

sb.heatmap(heart.corr(), annot=True, cmap='coolwarm')

plt.show()
f, axes = plt.subplots(4,4, figsize=(20, 15))

sb.distplot( heart["age"], ax=axes[0,0])

sb.distplot( heart["sex"], ax=axes[0,1])

sb.distplot( heart["cp"], ax=axes[0,2])

sb.distplot( heart["trestbps"], ax=axes[0,3])

sb.distplot( heart["chol"], ax=axes[1,0])

sb.distplot( heart["fbs"], ax=axes[1,1])

sb.distplot( heart["restecg"], ax=axes[1,2])

sb.distplot( heart["thalach"], ax=axes[1,3])

sb.distplot( heart["exang"], ax=axes[2,0])

sb.distplot( heart["oldpeak"], ax=axes[2,1])

sb.distplot( heart["slope"], ax=axes[2,2])

sb.distplot( heart["ca"], ax=axes[2,3])

sb.distplot( heart["thal"], ax=axes[3,0])

sb.distplot( heart["target"], ax=axes[3,1])

plt.show()
cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak','target']

sb.set_style('whitegrid')

sb.pairplot(heart[cols], height=3, hue='target')

plt.show()
y = heart.target

X = heart.drop(['target'], axis=1).values
# Split data into train and test formate

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
algo = {'Logistic Regression': LogisticRegression(solver='liblinear'), 

        'Decision Tree':DecisionTreeClassifier(), 

        'Random Forest':RandomForestClassifier(n_estimators=10, random_state=0), 

        'SVM':SVC(gamma=0.01, kernel='linear'),

        'Gradient Boosting' :GradientBoostingClassifier(max_features=1, learning_rate=0.05)

       }

predict_value = {}

plt.figure(figsize=(10, 5))

for k, v in algo.items():

    model = v

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    predict_value[k] = model.score(X_test, y_test)*100

    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100))

    print('AUC-ROC Curve of ' + k + ' is {0:.2f}\n'.format(roc_auc*100))

    

    # plot the roc curve for the model

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.plot(false_positive_rate, true_positive_rate, marker='*', label=k)

    plt.legend(loc='lower right')

    

plt.show()
plt.style.use('ggplot')

plt.figure(figsize=(12, 7))

sb.barplot(x=list(predict_value.keys()), y=list(predict_value.values()))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy")

plt.xlabel("Modals")

plt.show()