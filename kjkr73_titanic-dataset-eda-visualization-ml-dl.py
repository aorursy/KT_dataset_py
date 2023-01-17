import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report,auc,f1_score,precision_recall_curve,confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV,KFold

from sklearn.preprocessing import normalize

import tensorflow as tf

sns.set_style('darkgrid')

%matplotlib inline

import re

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, roc_curve

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset_train = pd.read_csv("/kaggle/input/titanic/train.csv")

dataset_train.head()
dataset_train.Ticket.dtype
dataset_test = pd.read_csv("/kaggle/input/titanic/test.csv")

dataset_test.head()
gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

gender.head()
dataset_train.describe()
dataset_train.isnull().sum()
dataset_test.describe()
dataset_test.isnull().sum()
gender.describe()
dataset_train['Age'] = dataset_train['Age'].fillna(dataset_train['Age'].mean())

mean_train = int(dataset_train['Age'].mean())

mean_test = int(dataset_test['Age'].mean())

#dataset_train['Age'] = dataset_train['Age'].fillna(np.random.choice([i for i in range(mean_train - 10, mean_train + 10)]))

dataset_train['Embarked'] = dataset_train['Embarked'].fillna(np.random.choice(['C','Q','S']))

dataset_test['Age'] = dataset_test['Age'].fillna(dataset_test['Age'].mean())

#dataset_test['Age'] = dataset_test['Age'].fillna(np.random.choice([i for i in range(mean_test - 10, mean_test + 10)]))

dataset_test['Fare'] = dataset_test['Fare'].fillna(dataset_test['Fare'].mean())
train_corr = dataset_train.corr()

plt.figure(figsize = (8,6))

sns.heatmap(train_corr)
test_corr = dataset_test.corr()

plt.figure(figsize = (8,6))

sns.heatmap(test_corr)
fig, axes = plt.subplots(2,4,figsize = (16,10), sharex = False, sharey = False)



sns.countplot(x = 'Sex', data = dataset_train, ax = axes[0,0],edgecolor = 'black')

sns.countplot(x = 'Survived', data = dataset_train, ax = axes[0,1],edgecolor = 'black')

sns.countplot(x = 'Parch', data = dataset_train, ax = axes[0,2],edgecolor = 'black')

sns.countplot(x = 'SibSp', data = dataset_train, ax = axes[0,3],edgecolor = 'black')

sns.distplot(dataset_train['Fare'],ax = axes[1,0])

sns.countplot(x = 'Embarked', data = dataset_train, ax = axes[1,1],edgecolor = 'black')

sns.distplot(dataset_train['Age'], ax = axes[1,2])  #dropped all the NaN values and plotted the remaining float value

sns.countplot(x = 'Pclass', data = dataset_train,ax = axes[1,3],edgecolor = 'black')

plt.show()
sns.pairplot(dataset_train, kind = 'scatter', hue = 'Sex',plot_kws=dict(s=80, edgecolor="white", linewidth=0.5))

plt.show()
#PIE chart visualization for Male-Female Percentage

plt.figure(figsize = (8,8))

plt.title('Percent Male-Female',fontsize = 30)

x_sex,y_sex = dataset_train.Sex.value_counts(normalize = True)*100

print(dataset_train.Sex.value_counts(normalize = True)*100)

wedges= plt.pie([x_sex,y_sex], labels = ['Male','Female'], colors = ['green','blue'], autopct='%.2f%%', 

        textprops = {'color':'Black','size':20}, wedgeprops={'linewidth':1, 'edgecolor':'black'})

plt.show()
#PIE chart visualization for Survived-Not Survived Percentage

plt.figure(figsize = (8,8))

plt.title('Percent Survived-Not Survived', fontsize = 30)

x_surv, y_surv = dataset_train.Survived.value_counts(normalize = True)*100

print(dataset_train.Survived.value_counts(normalize = True)*100)

plt.pie([x_surv,y_surv], labels = ['Survived','Not Survived'], colors = ['green','blue'],  autopct='%.2f%%', 

        textprops = {'color':'Black','size':20}, wedgeprops={'linewidth':1, 'edgecolor':'black'})

plt.show()
# BAR distributions of Survived and Pclass to look for std deviation

plt.figure(figsize = (10,7))

plt.hist(dataset_train['Fare'], bins = 20, histtype = 'bar', edgecolor = 'Black')

plt.title('Range of Fare value values VS Count', fontsize = 20)

plt.xlabel('Range(Fare)', fontsize = 15)

plt.ylabel('Count', fontsize = 15)

plt.show()
plt.figure(figsize = (10,7))

plt.hist(dataset_train['Age'], bins = 30, histtype = 'bar',edgecolor = 'Black')

plt.title('Range of Age values VS Count(No of persons)', fontsize = 20)

plt.xlabel('Range(AGE)', fontsize = 12)

plt.ylabel('Count', fontsize = 12)

plt.show()
def get_title(string):

    if '.' in string:

        return string.split(',')[1].split('.')[0].strip()

    else:

        return 'N.F'

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Dona', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

    

    

dataset_train['Title'] = dataset_train['Name'].apply(get_title)

temp_title = dataset_train.apply(replace_titles, axis = 1)

temp_title.value_counts()

data_sur = dataset_train[dataset_train['Survived'] == 1]

survived_title = data_sur.apply(replace_titles, axis = 1)

survived_title.value_counts()

temp_title.value_counts()
fig,axes = plt.subplots(1,2, figsize = (17,7))

a = axes[0].bar(temp_title.value_counts().index, height=temp_title.value_counts().values, color=['red','green','yellow','blue'],

       edgecolor = 'Black')

axes[0].set_xlabel('Title', fontdict={'fontsize':15})

axes[0].set_ylabel('Count', fontdict={'fontsize':15})

axes[0].set_title('Title(Survived and Not Survived) vs Count', fontdict={'fontsize':19})

axes[0].tick_params(labelsize = 13)

axes[0].set_yticks(range(0,800,100))

for ax in a.patches:

    axes[0].text(ax.get_x() + 0.25, ax.get_height() + 5, str(round(ax.get_height(), 2)), fontdict={'fontsize':14})

    

b = axes[1].bar(survived_title.value_counts().index, height=survived_title.value_counts().values, color=['red','green','yellow','blue'],

       edgecolor = 'Black')

axes[1].set_xlabel('Title', fontdict={'fontsize':15})

axes[1].set_ylabel('Count', fontdict={'fontsize':15})

axes[1].set_title('Title(Suvived) vs Count', fontdict={'fontsize':19})

axes[1].tick_params(labelsize = 13)

axes[1].set_yticks(range(0,300,20))

for ax in b.patches:

    axes[1].text(ax.get_x() + 0.25, ax.get_height() + 2, str(round(ax.get_height(), 2)), fontdict={'fontsize':14})

data_sur_fare = dataset_train.groupby('Sex').Fare.mean()

print(data_sur_fare)

ax_sur_fare = data_sur_fare.plot.bar(edgecolor = 'Black',fontsize = 12, figsize =(10,7), color = ['Red', 'Green'])

plt.title('Averge Fare for Males and Females', fontsize = 20)

plt.xlabel('Sex', fontdict = {'fontsize':17})

plt.ylabel('Avergae Fare', fontdict={'fontsize':17})

plt.yticks(range(0,70,10))

plt.xticks(rotation = 45)

plt.tick_params(labelsize = 14)

for i in ax_sur_fare.patches:

    ax_sur_fare.text(i.get_x()+0.19, i.get_height() + 0.4, str(round(i.get_height(),2)) , fontsize = 15 , color = 'Black')
plt.figure(figsize = (10,7))

sns.violinplot(dataset_train['Survived'], dataset_train['SibSp'], )

plt.tick_params(labelsize = 13)

plt.xlabel('Survived', fontdict={'fontsize':17})

plt.ylabel('SibSp', fontdict = {'fontsize':17})

plt.title('Survived VS SibSp', fontdict={'fontsize':20})

plt.show()
plt.figure(figsize = (13,8))

axes= sns.barplot(x = sorted(dataset_train['SibSp'].unique()), y = 'Age', data = dataset_train[['Age', 'SibSp']].groupby('SibSp').mean(),

                 linewidth = 1, edgecolor = 'black')

for ax in axes.patches:

    plt.text(ax.get_x() + 0.25, ax.get_height() + 0.3, str(round(ax.get_height(), 2)), fontsize = 15)

plt.xlabel('SibSp', fontdict = {'fontsize':15})

plt.ylabel('Average Age', fontdict = {'fontsize':15})

plt.yticks(range(0,45,5))

plt.tick_params(labelsize = 13)

plt.title('Average Age VS SibSp', fontdict = {'fontsize':20})

plt.show()
fare_cabin = dataset_train[['Cabin', 'Fare']].groupby(by = 'Cabin').mean()

fare_cabin = fare_cabin.sort_values(by = 'Fare', ascending = False)

fare_cabin = fare_cabin[:16][:]

fare_cabin = fare_cabin.reset_index()

fare_cabin = fare_cabin.sort_values(by = 'Cabin', ascending = True)

def label_encoder(string):

    num = re.findall('\d+', string)

    alpha = string[0]

    if len(num)>1:

        label = label = alpha + '(' + '-'.join(num) + ')'

    else:

        label = alpha + '-'.join(num)

    return label



fare_cabin['Cabin'] = fare_cabin['Cabin'].apply(label_encoder)
plt.figure(figsize = (20,8))

plt.plot(fare_cabin['Cabin'], fare_cabin['Fare'], color = 'Green')

plt.plot(fare_cabin['Cabin'], fare_cabin['Fare'], '*', color = 'red', markersize = 20)

plt.ylabel('Average Fare', fontdict = {'fontsize':17})

plt.xlabel('Cabin Number', fontdict= {'fontsize':17})

plt.xticks(rotation = 45)

plt.tick_params(labelsize = 13)

plt.title('Average fare for top 10 most popular Cabins according to data', fontdict={'fontsize':20})

plt.show()
dataset_train['Ticket'] = dataset_train['Ticket'].apply(lambda x: len(x))

dataset_train['Title'] = dataset_train['Name'].apply(get_title)

dataset_train['Title'] = dataset_train.apply(replace_titles, axis = 1)

drop_cols = ['PassengerId','Name','Cabin', 'Title']

encode_cols = ['Sex','Embarked', 'Title']

encode_after = pd.get_dummies(dataset_train[encode_cols])

fin_data = dataset_train.copy()

fin_data = fin_data.drop(drop_cols, axis = 1)

fin_data = pd.concat([fin_data, encode_after], axis = 1)

print(fin_data.columns)

fin_data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)
dataset_test['Title'] = dataset_test['Name'].apply(get_title)

dataset_test['Ticket'] = dataset_test['Ticket'].apply(lambda x: str(x))

dataset_test['Ticket'] = dataset_test['Ticket'].apply(lambda x: len(x))

dataset_test['Title'] = dataset_test.apply(replace_titles, axis = 1)

encode_cols_teset = pd.get_dummies(dataset_test[encode_cols])

fin_data_test = dataset_test.copy()

fin_data_test = fin_data_test.drop(['PassengerId','Name','Cabin', 'Embarked','Sex'],axis =1)

fin_data_test = pd.concat([fin_data_test, encode_cols_teset], axis = 1)

fin_data_test = fin_data_test[[ 'Pclass', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',

        'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q',

       'Embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr',

       'Title_Mrs']]
fin_data.head()
fin_data_test.head()
X_train = fin_data.values[:,1:]

Y_train = fin_data.values[:,0]

X_test = fin_data_test.values[:,:]

Y_test = gender.values[:,1:]
params = {'penalty':['l1','l2'], 'C':[0.01,0.1,1,10,100]}

lr = LogisticRegression(solver = 'liblinear')

grid = GridSearchCV(lr, param_grid=params, scoring ='f1', cv = 10, n_jobs=-1)

grid.fit(X_train, Y_train)

grid.best_params_
lr = LogisticRegression(C = 1, penalty='l2', solver='liblinear')

lr.fit(X_train, Y_train)

predict_lr = lr.predict(X_test)
print("Accuracy = {0}%".format(round(accuracy_score(Y_test, predict_lr)*100, 2)))

print(classification_report(Y_test, predict_lr))

print("Score = {0}".format(f1_score(Y_test, predict_lr)))

print(confusion_matrix(Y_test, predict_lr))
auc_score_lr = roc_auc_score(Y_test, predict_lr)

print('AUROC Score : {0}'.format(round(auc_score_lr, 4)))

fpr, tpr, threshold = roc_curve(Y_test, predict_lr)

plt.figure(figsize = (12,7))

plt.plot([0,1], [0,1], '--')

plt.plot(fpr, tpr, '-*', color = 'Orange', label = 'ROC curve area : {0}'.format(round(auc_score_lr, 4)))

plt.xlabel('False Postitive Rate', fontdict = {'fontsize':15})

plt.ylabel('True Postitive Rate', fontdict = {'fontsize':15})

plt.title('ROC-AUC Curve Logistic Regression', fontdict = {'fontsize':20})

plt.legend()

plt.show()
rf = RandomForestClassifier()

params_rf = {'n_estimators':list(range(1,20)), 'max_depth':list(range(1,10)), 'criterion':['gini', 'entropy']} #'max_features':['auto', 'log2', 'sqrt'],

            #'bootstrap':[True, False]

grid_rf = GridSearchCV(rf, param_grid=params_rf, cv = 5, scoring='accuracy', n_jobs = -1)

grid_rf.fit(X_train,Y_train)

print('HyperParameter optimization')

grid_rf.best_params_
rf = RandomForestClassifier(criterion = 'entropy', max_depth = 7, n_estimators = 16, bootstrap=False) #4,13

rf.fit(X_train, Y_train)

predict_rf = rf.predict(X_test)

print("Accuracy Score = {} %".format(rf.score(X_test, Y_test)*100))

print(classification_report(Y_test, predict_rf))

print('F1 Score = {}'.format(f1_score(Y_test, predict_rf)))
auc_score_rf = roc_auc_score(Y_test, predict_rf)

print('AUROC Score : {0}'.format(round(auc_score_rf, 4)))

fpr, tpr, threshold = roc_curve(Y_test, predict_rf)

plt.figure(figsize = (12,7))

plt.plot([0,1], [0,1], '--')

plt.plot(fpr, tpr, '-*', color = 'Orange', label = 'ROC curve area : {0}'.format(round(auc_score_rf, 4)))

plt.xlabel('False Postitive Rate', fontdict = {'fontsize':15})

plt.ylabel('True Postitive Rate', fontdict = {'fontsize':15})

plt.title('ROC-AUC Curve Random Forest', fontdict = {'fontsize':20})

plt.legend()

plt.show()
def model():

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(120, activation = tf.nn.relu, kernel_initializer = 'normal'))

    model.add(tf.keras.layers.Dense(120, activation = tf.nn.relu, kernel_initializer = 'normal'))

    model.add(tf.keras.layers.Dense(120, activation = tf.nn.relu, kernel_initializer = 'normal'))

    #model.add(tf.keras.layers.Dense(120, activation = tf.nn.relu, kernel_initializer = 'normal'))

    model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
model = model()

model.fit(X_train, Y_train, epochs = 500, verbose = 0)
val_loss, val_acc = model.evaluate(X_train, Y_train)
val_loss, val_acc = model.evaluate(X_test, Y_test)
predict_nn = model.predict([X_test])

predict_nn_fin = []

for i in range(len(predict_nn)):

    if predict_nn[i]>=0.7:

        predict_nn_fin.append(1)

    else:

        predict_nn_fin.append(0)
print("Accuracy = {0}%".format(round(accuracy_score(Y_test, predict_nn_fin)*100, 2)))

print(classification_report(Y_test, predict_nn_fin))

print("Score = {0}".format(f1_score(Y_test, predict_nn_fin)))

print(confusion_matrix(Y_test, predict_nn_fin))
auc_score_nn = roc_auc_score(Y_test, predict_nn_fin)

print('AUROC Score : {0}'.format(round(auc_score_nn, 4)))

fpr, tpr, threshold = roc_curve(Y_test, predict_nn_fin)

plt.figure(figsize = (12,7))

plt.plot([0,1], [0,1], '--')

plt.plot(fpr, tpr, '-*', color = 'Orange', label = 'ROC curve area : {0}'.format(round(auc_score_nn, 4)))

plt.xlabel('False Postitive Rate', fontdict = {'fontsize':15})

plt.ylabel('True Postitive Rate', fontdict = {'fontsize':15})

plt.title('ROC-AUC Curve Random Forest', fontdict = {'fontsize':20})

plt.legend()

plt.show()
'''submission = pd.DataFrame({

        "PassengerId": gender['PassengerId'],

        "Survived": predict_rf

    })



submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)



submission.to_csv("titanic1_submission.csv", index=False)'''