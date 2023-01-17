import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

#plt.style.use('fivethirtyeight')

%matplotlib inline

matplotlib.style.use('ggplot')

pd.options.display.max_rows = 999

pd.options.display.max_columns = 999

pd.options.mode.chained_assignment=None

from pandas.plotting import table

from subprocess import check_output

check_output(["ls", "../input"]).decode("utf8")
titanic_train = pd.read_csv("../input/train.csv", low_memory=False)

titanic_train.head()
titanic_test = pd.read_csv("../input/test.csv", low_memory=False)

titanic_test.head()
def data_overview(df):

    print('Data information:')

    df_info = df.info(verbose=False)

    df_describe = df.describe()

    df_missing = df.isnull().sum()[df.isnull().sum()>0]

    print('Data description : ')

    print(df_describe)

    print('Missing Data values:')

    print(df_missing)
data_overview(titanic_train)
data_overview(titanic_test)
titanic_train_test = titanic_train.append(titanic_test)

titanic_train_test.shape
titanic_train_test['Age'] = titanic_train_test['Age'].fillna(titanic_train['Age'].median())

titanic_train_test['Cabin']=titanic_train_test['Cabin'].fillna('missing')

titanic_train_test['Embarked']=titanic_train_test['Embarked'].fillna('missing')

titanic_train_test['Fare'] = titanic_train_test['Fare'].fillna(titanic_train_test['Fare'].median())
fig,ax=plt.subplots(figsize=(12,8))

titanic_train['Survived'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax,shadow=True);

ax.set_title('Survived');

ax.set_ylabel('');

plt.legend(['Dead','Survived']);
fig, ax = plt.subplots(figsize=(12,8))

sns.distplot(titanic_train_test['Fare'], hist=False, kde=True);

table(ax, np.round(titanic_train_test['Fare'].to_frame().describe()),loc='upper right', colWidths=[0.2, 0.2, 0.2])

ax.set_xlabel('Fare');

ax.set_ylabel('Density');

ax.set_title('Distribution of Titanic Fares');
titanic_Fare_range = [(0,10),(11,20), (21,30),(31,60),(61,100),(101,550)]

Fare_range = []

titanic_Fare_int = list(map(int,titanic_train_test.Fare.values))

for j in range(len(titanic_Fare_int)):

    for i in range(len(titanic_Fare_range)):

        if titanic_Fare_int[j] in range(titanic_Fare_range[i][0],titanic_Fare_range[i][1]):

            fare_range = titanic_Fare_range[i]

        else:

            pass

    Fare_range.append(fare_range)
titanic_train_test['Fare_range']=Fare_range
sns.factorplot(x="Fare_range",hue="Survived", data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)],

                                                                      kind="count",size=6,aspect=1.5, order=titanic_Fare_range);
fig,ax = plt.subplots(figsize=(18,8))

ax1 = plt.subplot(121)

titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)]['Fare_range'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1,shadow=True, 

                                                    legend=True);

ax1.set_title('Percentage distribution of fare range');

ax1.set_ylabel('');

#plt.legend(['0-8','9-15','16-30','31-60','61-100','101-600']);



survival_percent_fr = 100*(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Fare_range').sum()['Survived']/(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Fare_range').count()['Survived']))

#survival_percent_fr

ax2 = plt.subplot(122)

sns.barplot(survival_percent_fr.index.values, survival_percent_fr.values);

ax2.set_ylabel('survival percentage')

ax2.set_title('Percentage survived in each Fare range');
sns.factorplot(x="Sex",hue='Survived', data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)], kind="count",

                   palette="BuPu", size=6, aspect=1.5);
titanic_train_test['Sex'].value_counts()
survival_percent_sex = 100*(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Sex').sum()['Survived']/(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Sex').count()['Survived']))

survival_percent_sex
fix, ax = plt.subplots(figsize=(12,8))

sns.distplot(titanic_train_test.Age);

table(ax, np.round(titanic_train_test['Age'].to_frame().describe()),loc='upper right', colWidths=[0.2, 0.2, 0.2])

ax.set_xlabel('Age');

ax.set_ylabel('density');

ax.set_title('Distribution of Age');
titanic_Age_range = [(0,12),(13,20),(21,30),(31,40),(41,60),(61,80)]

Age_range = []

titanic_Age_int = list(map(int,titanic_train_test.Age.values))

for j in range(len(titanic_Age_int)):

    for i in range(len(titanic_Age_range)):

        if titanic_Age_int[j] in range(titanic_Age_range[i][0],titanic_Age_range[i][1]):

            age_range = titanic_Age_range[i]

        else:

            pass

    Age_range.append(age_range)
titanic_train_test['Age_range'] = Age_range
sns.factorplot(x="Age_range",hue="Survived", data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)], kind="count",size=6, 

               aspect=1.5, order=titanic_Age_range);
fig,ax = plt.subplots(figsize=(18,8))

ax1 = plt.subplot(121)

titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)]['Age_range'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1,shadow=True, 

                                                    legend=True);

ax1.set_title('Percentage distribution of age range');

ax1.set_ylabel('');

#plt.legend(['0-8','9-15','16-30','31-60','61-100','101-600']);



survival_percent_ag = 100*(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Age_range').sum()['Survived']/(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Age_range').count()['Survived']))

#survival_percent_fr

ax2 = plt.subplot(122)

sns.barplot(survival_percent_ag.index.values, survival_percent_ag.values);

ax2.set_ylabel('survival percentage')

ax2.set_title('Percentage survived in each Age range');

#survival_percent_ag = 100*(titanic_train.groupby('Age_range').sum()['Survived']/(titanic_train.groupby('Age_range').count()['Survived']))

#survival_percent_ag
titanic_train_test.Pclass.value_counts()
sns.factorplot(x="Pclass",hue="Survived", data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)], kind="count",size=6, 

               aspect=1.5);
survival_percent_pcl = 100*(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Pclass').sum()['Survived']/(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Pclass').count()['Survived']))

survival_percent_pcl
titanic_train_test.SibSp.value_counts()
titanic_Fam_size = ['single','two','three','many']

Fam_size = []

titanic_Fam_int = list(map(int,titanic_train_test.SibSp.values))

for j in range(len(titanic_Fam_int)):

    if titanic_Fam_int[j] ==0:

        fam_size = titanic_Fam_size[0]

    elif titanic_Fam_int[j]==1:

        fam_size=titanic_Fam_size[1]

    elif titanic_Fam_int[j] == 2:

        fam_size = titanic_Fam_size[2]

    elif titanic_Fam_int[j] >2 :

        fam_size = titanic_Fam_size[3]

    Fam_size.append(fam_size)
titanic_train_test['Fam_size'] = Fam_size
sns.factorplot(x="Fam_size",hue="Survived", data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)], kind="count",size=6, 

               aspect=1.5);
survival_percent_sibsp = 100*(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Fam_size').sum()['Survived']/(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Fam_size').count()['Survived']))

survival_percent_sibsp
titanic_train.Parch.value_counts()
ParChi = ['single','two','three','many']

Parch_size = []

titanic_parch_int = list(map(int,titanic_train_test.Parch.values))

for j in range(len(titanic_parch_int)):

    if titanic_parch_int[j] ==0:

        parch_size = ParChi[0]

    elif titanic_parch_int[j]==1:

        parch_size=ParChi[1]

    elif titanic_parch_int[j] == 2:

        parch_size = ParChi[2]

    elif titanic_parch_int[j] >2 :

        parch_size = ParChi[3]

    Parch_size.append(parch_size)
titanic_train_test['Parch_size'] = Parch_size
sns.factorplot(x="Parch_size",hue="Survived", data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)], kind="count",size=6, 

               aspect=1.5);
survival_percent_parch = 100*(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Parch_size').sum()['Survived']/(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Parch_size').count()['Survived']))

survival_percent_parch
titanic_train_test.Embarked.value_counts()
sns.factorplot(x="Embarked",hue="Survived", data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)], kind="count",size=6, 

               aspect=1.5);
survival_percent_embk = 100*(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Embarked').sum()['Survived']/(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Embarked').count()['Survived']))

survival_percent_embk
Title=[]

for i in range(len(titanic_train_test)):

    names = titanic_train_test.Name.values[i].replace('.',',').split(',')

    title = names[1]

    Title.append(title)    
titanic_train_test['PTitle'] = Title
titanic_train_test.PTitle.value_counts()
PasTitle = ['Mr','Miss','Mrs','Master','Dr or Rev','Other']

Pas_title = []

for j in range(len(titanic_train_test['PTitle'])):

    if titanic_train_test['PTitle'].values[j] ==' Mr':

        pas_title= PasTitle[0]

    elif titanic_train_test['PTitle'].values[j] ==' Miss':

        pas_title= PasTitle[1]

    elif titanic_train_test['PTitle'].values[j] ==' Mrs':

        pas_title= PasTitle[2]

    elif titanic_train_test['PTitle'].values[j] ==' Master':

        pas_title= PasTitle[3]

    elif (titanic_train_test['PTitle'].values[j] ==' Dr') or (titanic_train_test['PTitle'].values[j]==' Rev'):

        pas_title= PasTitle[4]

    else:

        pas_title= PasTitle[5]

    Pas_title.append(pas_title)
titanic_train_test['Pas_title'] = Pas_title
sns.factorplot(x="Pas_title",hue="Survived", data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)], kind="count",size=10, 

               aspect=1.5);
survival_percent_tit = 100*(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Pas_title').sum()['Survived']/(titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)].groupby('Pas_title').count()['Survived']))

survival_percent_tit
titanic_train_test.Cabin.value_counts().iloc[0:5]
titanic_decks = []

for i in range(len(titanic_train_test['Cabin'])):

    titanic_deck = titanic_train_test.Cabin.values[i][0]

    titanic_decks.append(titanic_deck)

titanic_train_test['Deck'] = titanic_decks
titanic_train_test.Deck.value_counts()
sns.factorplot(x="Deck",hue="Survived", data=titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)], kind="count",size=8, 

               aspect=1.5);
titanic_train_relevant = titanic_train_test[titanic_train_test['PassengerId']<=len(titanic_train)]

titanic_test_relevant = titanic_train_test[titanic_train_test['PassengerId']>len(titanic_train)]
titanic_train_relevant = titanic_train_relevant.drop(['Age','Cabin','Fare','Name','Parch','SibSp',

                                                     'Ticket','PTitle'], axis=1)
titanic_test_relevant = titanic_test_relevant.drop(['Age','Cabin','Fare','Name','Parch','SibSp',

                                                     'Ticket','PTitle','Survived'], axis=1)
titanic_train_test_relevant = titanic_train_relevant.append(titanic_test_relevant)

titanic_train_test_relevant = titanic_train_test_relevant.drop('Survived', axis=1)
titanic_train_test_relevant = titanic_train_test_relevant.set_index('PassengerId')

titanic_train_test_relevant.head()
titanic_train_test_relevant = pd.get_dummies(titanic_train_test_relevant, columns=list(titanic_train_test_relevant.columns.values))
titanic_train_test_relevant.shape
titanic_train_final=titanic_train_test_relevant.iloc[0:len(titanic_train)]

titanic_test_final=titanic_train_test_relevant.iloc[len(titanic_train):len(titanic_train_test_relevant)]
titanic_train_target = titanic_train_relevant['Survived'].astype(int)
titanic_train_final.head()
titanic_test_final.head()
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold

from sklearn import metrics

linreg = LinearRegression()

logreg = LogisticRegression()

rfclf = RandomForestClassifier()

svc = SVC()

linsvc=LinearSVC()

knn = KNeighborsClassifier()

gnb = GaussianNB()
X_train = titanic_train_final

y_train = titanic_train_target
def find_predicted_values(X_train, y_train,n, ml_algo):

    #n = number of cross validation folds

    #ml_algo = machine learning algorithms

    predictions = []

    kf = KFold(n_splits=n)

    

    for train, test in kf.split(X_train,y_train):

        train_data = X_train.iloc[train,:]

        target_data = y_train.iloc[train]

        ml_algo.fit(train_data,target_data)

        test_predict = ml_algo.predict(X_train.iloc[test,:])

        predictions.append(test_predict)

    predictions=np.concatenate(predictions)

    return predictions
linreg_predictions = find_predicted_values(X_train,y_train,4,linreg)

logreg_predictions = find_predicted_values(X_train,y_train,4,logreg)

rfclassifier_predictions = find_predicted_values(X_train,y_train,4,rfclf)

svc_predictions = find_predicted_values(X_train,y_train,4,svc)

linsvc_predictions = find_predicted_values(X_train,y_train,4,linsvc)

knn_predictions = find_predicted_values(X_train,y_train,4,knn)

gnb_predictions = find_predicted_values(X_train,y_train,4,gnb)
linreg_predictions[linreg_predictions > 0.5]=1

linreg_predictions[linreg_predictions <= 0.5]=0

accuracy_linreg = metrics.accuracy_score(y_train,linreg_predictions)

accuracy_linreg
accuracy_logreg = metrics.accuracy_score(y_train, logreg_predictions)

accuracy_logreg
accuracy_rfclf = metrics.accuracy_score(y_train,rfclassifier_predictions)

accuracy_rfclf
accuracy_svc = metrics.accuracy_score(y_train,svc_predictions)

accuracy_svc
accuracy_linsvc = metrics.accuracy_score(y_train,linsvc_predictions)

accuracy_linsvc
accuracy_knn = metrics.accuracy_score(y_train,knn_predictions)

accuracy_knn
accuracy_gnb = metrics.accuracy_score(y_train,gnb_predictions)

accuracy_gnb
N = 10  # number of folds varies from 1 to 10

accuracy_linreg_kfolds = []

accuracy_logreg_kfolds = []

accuracy_rfclf_kfolds = []

accuracy_svc_kfolds = []

accuracy_linsvc_kfolds = []

accuracy_knn_kfolds = []

accuracy_gnb_kfolds = []

for n in range(2,N+2):

    prediction_linreg_kfold = find_predicted_values(X_train,y_train,n,linreg)

    prediction_logreg_kfold = find_predicted_values(X_train,y_train,n,logreg)

    prediction_rfclf_kfold = find_predicted_values(X_train,y_train,n,rfclf)

    prediction_svc_kfold = find_predicted_values(X_train,y_train,n,svc)

    prediction_linsvc_kfold = find_predicted_values(X_train,y_train,n,linsvc)

    prediction_knn_kfold = find_predicted_values(X_train,y_train,n,knn)

    prediction_gnb_kfold = find_predicted_values(X_train,y_train,n,gnb)

    prediction_linreg_kfold[prediction_linreg_kfold > 0.5]=1

    prediction_linreg_kfold[prediction_linreg_kfold <= 0.5]=0

    

    accuracy_linreg_kfold = metrics.accuracy_score(y_train,prediction_linreg_kfold)

    accuracy_logreg_kfold = metrics.accuracy_score(y_train,prediction_logreg_kfold)

    accuracy_rfclf_kfold = metrics.accuracy_score(y_train,prediction_rfclf_kfold)

    accuracy_svc_kfold = metrics.accuracy_score(y_train,prediction_svc_kfold)

    accuracy_linsvc_kfold = metrics.accuracy_score(y_train,prediction_linsvc_kfold)

    accuracy_knn_kfold = metrics.accuracy_score(y_train,prediction_knn_kfold)

    accuracy_gnb_kfold = metrics.accuracy_score(y_train,prediction_gnb_kfold)

    

    accuracy_linreg_kfolds.append(accuracy_linreg_kfold)

    accuracy_logreg_kfolds.append(accuracy_logreg_kfold)

    accuracy_rfclf_kfolds.append(accuracy_rfclf_kfold)

    accuracy_svc_kfolds.append(accuracy_svc_kfold)

    accuracy_linsvc_kfolds.append(accuracy_linsvc_kfold)

    accuracy_knn_kfolds.append(accuracy_knn_kfold)

    accuracy_gnb_kfolds.append(accuracy_gnb_kfold)
fig, ax = plt.subplots(figsize=(12,12))

sns.set_style("darkgrid")

plt.plot(list(range(2,N+2)),accuracy_linreg_kfolds, label='Linear Regression');

plt.plot(list(range(2,N+2)),accuracy_logreg_kfolds, label='Logistic Regression');

plt.plot(list(range(2,N+2)),accuracy_rfclf_kfolds, label='Random Forest Classifier');

plt.plot(list(range(2,N+2)),accuracy_svc_kfolds, label='Support Vector');

plt.plot(list(range(2,N+2)),accuracy_linsvc_kfolds, label='Linear SVC');

plt.plot(list(range(2,N+2)),accuracy_knn_kfolds, label='K Nearest Neighbors');

plt.plot(list(range(2,N+2)),accuracy_gnb_kfolds, label='Gaussian Naive Bayes');

ax.set_xlabel('Number of cross validation folds')

ax.set_ylabel('accuracy')

handles, labels = ax.get_legend_handles_labels();

ax.legend(handles, labels);
accuracy_dict={'Linear Reg':accuracy_linreg_kfolds, 'Logistic Reg': accuracy_logreg_kfolds,

               'Random Forest':accuracy_rfclf_kfolds, 'Support Vector': accuracy_svc_kfolds,

              'Linear SVC': accuracy_linsvc_kfolds, 'KNN':accuracy_knn_kfolds, 'GNB':accuracy_gnb_kfolds}

accuracy_table = pd.DataFrame(accuracy_dict)

accuracy_table.index=range(2,N+2)

accuracy_table.index.name='CV Folds'

accuracy_table
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svc_submit = SVC()

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

clf = GridSearchCV(svc, parameters)

clf.fit(X_train,y_train)

submit_predict = svc.predict(titanic_test_final).astype(int)

submit_predict
submission = pd.DataFrame({

    "PassengerId": titanic_test_final.index,

    "Survived": submit_predict

    })

submission.to_csv('submission.csv',index=False)