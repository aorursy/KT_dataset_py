#22p21c0162_Sornpraram



import pandas as pd 



from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

train_df
df = train_df.drop(['Name','PassengerId','Ticket','Cabin'], axis = 1)
for i in ['Survived', 'Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']:

    df[i] = df[i].fillna(df[i].mode()[0])
df
df = pd.get_dummies(df, columns=['Sex', 'Embarked'],drop_first = False)

df
y_train_df = df[['Survived']]

X_train_df = df.copy().drop('Survived', axis=1)



print(X_train_df.shape, y_train_df.shape)
nf=5

dt_model = DecisionTreeClassifier(max_leaf_nodes=5)

nb_model = GaussianNB()

mlp_model = MLPClassifier(random_state=42)
kf = KFold(n_splits=nf, random_state=42, shuffle=True)

dt_kfacc = 0

nb_kfacc = 0

mlp_kfacc = 0

for train_index, test_index in kf.split(X_train_df):

    X_train = X_train_df.iloc[train_index]

    X_test = X_train_df.iloc[test_index]

    y_train = y_train_df.iloc[train_index]

    y_test = y_train_df.iloc[test_index]

    

    dt_model.fit(X_train, y_train)

    nb_model.fit(X_train, y_train)

    mlp_model.fit(X_train, y_train)

    

    dt_predict = dt_model.predict(X_test)

    nb_predict = nb_model.predict(X_test)

    mlp_predict = mlp_model.predict(X_test)

    

    dt_kfacc += accuracy_score(y_test, dt_predict)

    nb_kfacc += accuracy_score(y_test, nb_predict)

    mlp_kfacc += accuracy_score(y_test, mlp_predict)



dt_kfacc = dt_kfacc/nf

nb_kfacc = nb_kfacc/nf

mlp_kfacc = mlp_kfacc/nf

    

print("Decision Tree K-fold validation score: ", dt_kfacc)

print("Naive Bayes K-fold validation score: ", nb_kfacc)

print("Multilayer Perceptron K-fold validation score: ", mlp_kfacc)
dt_predict = dt_model.predict(X_test)

nb_predict = nb_model.predict(X_test)

mlp_predict = mlp_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_predict)

dt_recall = recall_score(y_test, dt_predict)

dt_precision = precision_score(y_test, dt_predict)

dt_f1 = f1_score(y_test, dt_predict)

print('Decision Tree accuracy:',dt_acc)

print('Decision Tree recall:',dt_recall)

print('Decision Tree precision:',dt_precision)

print('Decision Tree f1:',dt_f1)
nb_acc = accuracy_score(y_test, nb_predict)

nb_recall = recall_score(y_test, nb_predict)

nb_precision = precision_score(y_test, nb_predict)

nb_f1 = f1_score(y_test, nb_predict)

print('Naive Bayes accuracy:',nb_acc)

print('Naive Bayes recall:',nb_recall)

print('Naive Bayes precision:',nb_precision)

print('Naive Bayes f1:',nb_f1)
mlp_acc = accuracy_score(y_test, mlp_predict)

mlp_recall = recall_score(y_test, mlp_predict)

mlp_precision = precision_score(y_test, mlp_predict)

mlp_f1 = f1_score(y_test, mlp_predict)

print('Multilayer Perceptron accuracy:',mlp_acc)

print('Multilayer Perceptron recall:',mlp_recall)

print('Multilayer Perceptron precision:',mlp_precision)

print('Multilayer Perceptron f1:',mlp_f1)