import pandas as pd

import numpy as np

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold

import seaborn as sns

import matplotlib.pyplot as plt
df_original = pd.read_csv('/kaggle/input/titanic/train.csv')

df = df_original.copy()
df.head(10)
df.info()
df.describe()
n_row, n_col = df.shape



print("Number of row: ",n_row)

print("Number of column: ",n_col)
fig = plt.subplots(figsize = (12,5))

sns.countplot(x = 'Survived', hue = 'Pclass', data = df_original)
fig = plt.subplots(figsize = (12,5))

sns.countplot(x = 'Survived', hue = 'Sex', data = df_original)
fig = plt.subplots(figsize = (12,5))

sns.countplot(x = 'Survived', hue = 'Embarked', data = df_original)
df.Age.quantile(.99)
plt.figure(figsize = (16,12))

temp = df[df.Age < 65.87]

sns.violinplot(x = 'Survived', y = 'Age', data = temp)
df.Fare.quantile(.99)
plt.figure(figsize = (16,12))

temp = df[df.Fare < 249.006]

sns.violinplot(x = 'Survived', y = 'Fare', data = temp)
corrMatrix = df.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
df.isna().sum()
df.Age.plot.hist()
np.random.seed(0)

nan_rows = df['Age'].isna()

random_age = np.random.choice(df['Age'][~nan_rows], replace=True, size=sum(nan_rows))
df.loc[nan_rows,'Age'] = random_age
df.Age.plot.hist()
df.Embarked.value_counts()
df.Embarked.mode()[0]
df.Embarked.fillna(df.Embarked.mode()[0], inplace=True)
df = df.drop(['Ticket','Cabin','PassengerId','Name'],axis=1)
mean = df['Age'].mean()

std = df['Age'].std()

   

# Any value higher than upper limit or below lower limit is an outlier

upper_limit = mean + 3*std

lower_limit = mean - 3*std

upper_limit, lower_limit
outlier_rows = (df['Age'] > upper_limit) | (df['Age'] < lower_limit)  

df['Age'][outlier_rows]
df.loc[outlier_rows, 'Age'] = df['Age'][~outlier_rows].mean()
condition = (df['Age']>60) & (df['Sex'] == 'male')

condition_2 = (df['Age']>60) & (df['Sex'] == 'female')

df['ElderMale'] = np.where(condition, 1, 0)

df['ElderFemale'] = np.where(condition_2, 1, 0)

df.head()
dummy = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, dummy], axis=1)

df = df.drop(['Embarked'], axis=1)

df.head(10)
dummy = pd.get_dummies(df['Pclass'], prefix='Pclass')
df = pd.concat([df, dummy], axis=1)

df = df.drop(['Pclass'], axis=1)

df.head(10)
dummy = pd.get_dummies(df['Sex'], prefix='Sex')
df = pd.concat([df, dummy], axis=1)

df = df.drop(['Sex'], axis=1)

df.head(10)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(df['Age'].values.reshape(-1,1))
df['Age'] = scaler.transform(df['Age'].values.reshape(-1,1))

df['Age'].describe()
scaler.fit(df['Fare'].values.reshape(-1,1))

df['Fare'] = scaler.transform(df['Fare'].values.reshape(-1,1))

df['Fare'].describe()
test_size = int(df.shape[0]*0.1)
train_df = df.iloc[:-test_size].copy()

test_df = df.iloc[-test_size:].copy()
X = train_df.loc[:, df.columns != 'Survived']

Y = train_df.loc[:, 'Survived']
X.head(10)
Y.head(10)
# K = 5

K = 5

kf = KFold(n_splits=5,random_state=2020,shuffle=True)
K_Fold_list = []

for train_index,test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    K_Fold_list.append([[X_train,y_train],[X_test,y_test]])
K_Fold_list[0][0][0]
K_Fold_list[0][0][1]
from sklearn.metrics import recall_score

from sklearn.metrics import precision_score



def evaluate(y_true,y_pred,label=1):

    precision = precision_score(y_true, y_pred, pos_label=label)

    recall = recall_score(y_true, y_pred, pos_label=label)

    f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}
def show_result(clf, K_Fold_list):

    c = 1

    f1_list_class_survived = []

    f1_list_class_not_survived = []

    model_list = []

    for fold in K_Fold_list:

        print("#"*50)

        print("Fold #{}".format(c))

        print("#"*50)

        clf = clf.fit(fold[0][0], fold[0][1])

        model_list.append(clf)

        y_pred = clf.predict(fold[1][0]) # Predict

        y_true = fold[1][1]



        print("Class survived")

        metrics = evaluate(y_pred,y_true,label=1) # Lable 1 positive

        print("Precision survived:",metrics['precision'])

        print("Recall survived:",metrics['recall'])

        print("F1 survived:",metrics['f1'])

        f1_list_class_survived.append(metrics['f1'])



        print("")

        print("Class not survived")

        metrics = evaluate(y_pred,y_true,label=0) # Lable 0 positive

        print("Precision not survived:",metrics['precision'])

        print("Recall not survived:",metrics['recall'])

        print("F1 not survived:",metrics['f1'])

        f1_list_class_not_survived.append(metrics['f1'])

        print("#"*50)

        print("")

        c+=1



    avg_f1_class_survived = sum(f1_list_class_survived)/len(f1_list_class_survived)

    avg_f1_class_not_survived = sum(f1_list_class_not_survived)/len(f1_list_class_not_survived)



    print("#"*50)

    print("Summary")

    print("#"*50)

    print("Average F1 survived:",avg_f1_class_survived)

    print("Average F1 not survived:",avg_f1_class_not_survived)

    return {"Average_F1_survived":avg_f1_class_survived,

            "Average_F1_not_survived":avg_f1_class_not_survived

           },model_list

            
clf_tree = tree.DecisionTreeClassifier()

clf_tree = clf_tree.fit(K_Fold_list[0][0][0], K_Fold_list[0][0][1])
import graphviz 

dot_data = tree.export_graphviz(clf_tree,out_file=None,

                        feature_names = K_Fold_list[0][0][0].columns.to_list(),

                        class_names = ['Survived','No Survived'],

                        filled=True, rounded=True,special_characters=True)  

graph = graphviz.Source(dot_data) 

graph
clf_tree = tree.DecisionTreeClassifier()

f1_DT_avg,model_DT = show_result(clf_tree,K_Fold_list)
gnb = GaussianNB()

f1_NB_avg,model_NB = show_result(gnb,K_Fold_list)
clf_NN = MLPClassifier(random_state=2020, max_iter=5000, hidden_layer_sizes = 7)

f1_NN_avg,model_NN = show_result(clf_NN,K_Fold_list)
df_Average_F1 = pd.DataFrame([f1_DT_avg,f1_NB_avg,f1_NN_avg])

df_Average_F1['Name'] = ['Decision Tree', ' Naive Bayes', 'Neural Network']
ax = sns.barplot(x = 'Name', y = 'Average_F1_survived', data = df_Average_F1)

ax.set(xlabel='Name', ylabel='Average F1')

ax.set_title("Average F1 Predict Survived")
ax = sns.barplot(x = 'Name', y = 'Average_F1_not_survived', data = df_Average_F1)

ax.set(xlabel='Name', ylabel='Average F1')

ax.set_title("Average F1 Predict Not Survived")
def print_result(clfs, Xs):

    probs = np.zeros(shape=(Xs.shape[0], 2))



    for fold_id in range(len(clfs)):

        probs += clfs[fold_id].predict_proba(Xs.loc[:, df.columns != 'Survived']) / K

    preds = np.argmax(probs, axis=1)

    return preds

pred = print_result(model_DT, test_df)

y_true = test_df.loc[:, 'Survived']

evaluate(y_true,pred)
pred = print_result(model_NB, test_df)

y_true = test_df.loc[:, 'Survived']

evaluate(y_true,pred)
pred = print_result(model_NN, test_df)

y_true = test_df.loc[:, 'Survived']

evaluate(y_true,pred)