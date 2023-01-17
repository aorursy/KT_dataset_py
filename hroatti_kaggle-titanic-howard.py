# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_train.describe()
df_test.describe()
df_train.head()
df_test.head()
df_train.drop_duplicates(inplace=True)

df_test.drop_duplicates(inplace=True)



df_train.drop(labels=['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

df_test.drop(labels=['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)



df_train.dropna(subset=['Embarked'], inplace=True)#Exists value nan

df_test.dropna(subset=['Embarked'], inplace=True)



df_train.fillna(value={'Age': df_train['Age'].median()}, inplace=True)

df_test.fillna(value={'Age': df_test['Age'].median()}, inplace=True)



df_test.fillna(value={'Fare': df_test['Fare'].median()}, inplace=True)
df_train.head()
df_test.head()
df_train['Sex'] = pd.Categorical(df_train['Sex'])

df_test['Sex'] = pd.Categorical(df_test['Sex'])



df_train['Pclass'] = pd.Categorical(df_train['Pclass'])

df_test['Pclass'] = pd.Categorical(df_test['Pclass'])



df_train['SibSp'] = pd.Categorical(df_train['SibSp'])

df_test['SibSp'] = pd.Categorical(df_test['SibSp'])



df_train['Embarked'] = pd.Categorical(df_train['Embarked'])

df_test['Embarked'] = pd.Categorical(df_test['Embarked'])
df_train['PassengerId'] = pd.to_numeric(df_train['PassengerId'])

df_test['PassengerId'] = pd.to_numeric(df_test['PassengerId'])



df_train['Age'] = pd.to_numeric(df_train['Age'])

df_test['Age'] = pd.to_numeric(df_test['Age'])



df_train['Parch'] = pd.to_numeric(df_train['Parch'])

df_test['Parch'] = pd.to_numeric(df_test['Parch'])



df_train['Fare'] = pd.to_numeric(df_train['Fare'])

df_test['Fare'] = pd.to_numeric(df_test['Fare'])
columns_categorical = ['Sex', 'Pclass', 'SibSp', 'Embarked']

columns_numerical = ['PassengerId', 'Age', 'Parch', 'Fare']
df_train.head()
df_test.head()
from sklearn.preprocessing import LabelEncoder

enc_sex = LabelEncoder().fit(df_train['Sex'])

df_train['Sex'] = enc_sex.transform(df_train['Sex'])

df_test['Sex'] = enc_sex.transform(df_test['Sex'])



enc_emb = LabelEncoder().fit(df_train['Embarked'])

df_train['Embarked'] = enc_emb.fit_transform(df_train['Embarked'])

df_test['Embarked'] = enc_emb.fit_transform(df_test['Embarked'])
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



classifier = RandomForestClassifier(n_jobs=6, criterion='entropy', bootstrap=False)

X = df_train[['Sex', 'Pclass', 'SibSp', 'Embarked', 'PassengerId', 'Age', 'Parch', 'Fare']]

y = df_train['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



classifier.fit(X_train, y_train)



predicoes = classifier.predict(X_test)



np.mean(predicoes == y_test)
from sklearn import metrics

print(metrics.classification_report(y_test, predicoes, target_names=['No', 'Yes']))
from sklearn.metrics import plot_confusion_matrix



# Plot non-normalized confusion matrix

titles_options = [("Confusion matrix, without normalization", None),

                  ("Normalized confusion matrix - Precision", 'pred'),

                 ("Normalized confusion matrix - Recall", 'true')]

for title, normalize in titles_options:

    disp = plot_confusion_matrix(classifier, X_test, y_test,

                                 display_labels=['No', 'Yes'],

                                 cmap=plt.cm.Blues,

                                 normalize=normalize)

    disp.ax_.set_title(title)



    #print(title)

    #print(disp.confusion_matrix)



plt.show()
def curva_roc(modelo, X_test, y_test, y_pred):

       

    from sklearn.metrics import roc_curve

    from sklearn.metrics import roc_auc_score   



    probs = modelo.predict_proba(X_test)

    probs = probs[:, 1]



    auc = roc_auc_score(y_test, y_pred)

    print('AUC: %.3f' % auc)

   

    fpr, tpr, thresholds = roc_curve(y_test, probs)

    

    plt.plot([0, 1], [0, 1], linestyle='--')

   

    plt.plot(fpr, tpr, marker='.')

    

    plt.show()
curva_roc(classifier, X_test, y_test, predicoes)
print("PassengerId,Survived")

y_pred_final = classifier.predict(df_test)

for i, j in zip(df_test['PassengerId'], y_pred_final):

    print(str(i) + "," + str(j))
submission=pd.read_csv('/kaggle/input/titanic/test.csv')

submission['Survived']=y_pred_final

submission= submission.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'], axis=1)

submission.to_csv('submission.csv',index=False)