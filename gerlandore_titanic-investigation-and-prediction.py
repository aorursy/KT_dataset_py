import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import metrics

from sklearn import ensemble

from sklearn import model_selection

from sklearn import preprocessing

from sklearn import svm

from sklearn import neighbors

from sklearn import naive_bayes

from sklearn import linear_model

import matplotlib.pyplot as plt



np.random.seed(1234) #For reproducible results
dataset = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

y = dataset['Survived'].values



dataset.head()
dataset.describe(include='all')
fig,ax = plt.subplots(2,1,figsize=(10,10)) 

ax[0].set_title('dataset missing count(%)')

ax[1].set_title('test set missing count(%)')



sns.barplot(x = dataset.columns, y = dataset.isna().sum()/dataset.shape[0]*100, ax = ax[0])

sns.barplot(x = test.columns, y = test.isna().sum()/test.shape[0]*100, ax = ax[1])
dataset['Cabin'].fillna('NotKnown',inplace=True)

dataset['Age'].fillna(dataset['Age'].mean(),inplace=True)

dataset['Embarked'].fillna('S',inplace=True)



test['Cabin'] = test['Cabin'].fillna('NotKnown')

test['Age'] = test['Age'].fillna(test['Age'].mean())

#We'll deal with Fare after.
dataset.describe(include='all')
d= pd.DataFrame([dataset.loc[dataset.Sex=='male'].shape[0],dataset.loc[dataset.Sex=='female'].shape[0]],columns=['number'],index=['male','female'])

sns.barplot(x=dataset['Sex'], y=dataset['Survived'])

d
age_survived_males = dataset.loc[dataset['Survived'] == 1].loc[dataset['Sex'] == 'male']['Age']

age_deceased_males = dataset.loc[dataset['Survived'] == 0].loc[dataset['Sex'] == 'male']['Age']



age_survived_females = dataset.loc[dataset['Survived'] == 1].loc[dataset['Sex'] == 'female']['Age']

age_deceased_females = dataset.loc[dataset['Survived'] == 0].loc[dataset['Sex'] == 'female']['Age']



female_age_dist = dataset.loc[dataset['Sex']=='female']['Age']

male_age_dist = dataset.loc[dataset['Sex']=='male']['Age']



fig, ax = plt.subplots(3,2,figsize=(30,15))

sns.distplot(age_survived_males, kde=True,rug=True, color='green', ax = ax[0][0])

ax[0][0].set_title('Survived Titanic Men')

sns.distplot(age_deceased_males, kde=True,rug=True, color='red', ax = ax[0][1])

ax[0][1].set_title('Survived Titanic Women')

sns.distplot(age_survived_females, kde=True,rug=True, color='grey', ax = ax[1][0])

ax[1][0].set_title('Deceased Titanic Men')

sns.distplot(age_deceased_females, kde=True,rug=True, color='grey', ax = ax[1][1])

ax[1][1].set_title('Deceased Titanic Women')

sns.distplot(male_age_dist, kde=True,rug=True, color='blue', ax = ax[2][0])

ax[2][0].set_title('Men Age Distribution')

sns.distplot(female_age_dist, kde=True,rug=True, color='pink', ax = ax[2][1])

ax[2][1].set_title('Women Age Distribution')



for i in range(2):

    for j in range(2):

        ax[i][j].axvspan(0,18, facecolor='orange', label='adolescents([0,18])', alpha=0.1)

        ax[i][j].axvspan(18,35, facecolor='cyan', label='young adults([19,35])', alpha=0.1)

        ax[i][j].axvspan(35,60, facecolor='purple', label='adults([35,60])', alpha=0.1)

        ax[i][j].axvspan(60,100, facecolor= 'brown', label='elders([60+])',alpha=0.1)

        ax[i][j].legend()

survived_class = dataset.loc[dataset['Survived']==1]['Pclass'] #Distribution of survived per class

deceased_class = dataset.loc[dataset['Survived']==0]['Pclass'] #Distribution of deceased per class



fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(survived_class,ax = ax[0],label='Survived')

sns.distplot(deceased_class,ax = ax[1],label='Decesed')

people_per_class = dataset.groupby('Pclass').count()['Survived']

surv_per_class = dataset.loc[dataset.Survived == 1].groupby('Pclass').count()['Survived']

ratios = surv_per_class/people_per_class

fig, ax = plt.subplots(1,3,figsize=(30,7))

ax[0].set_title('# of passenger for the Class')

sns.barplot(x = people_per_class.index, y = people_per_class, ax=ax[0])

ax[1].set_title('# of SURVIVED for the Class')

sns.barplot(x = surv_per_class.index, y = surv_per_class, ax=ax[1])

ax[2].set_title('survived/total per class')

sns.barplot(x = ratios.index, y = ratios, ax=ax[2])
print(np.unique([s.split(',')[1].split(' ')[1] for s in dataset['Name']]))

print(np.unique([s.split(',')[1].split(' ')[1] for s in test['Name']]))



dataset['Title'] = [s.split(',')[1].split(' ')[1] for s in dataset['Name']]

test['Title'] = [s.split(',')[1].split(' ')[1] for s in test['Name']]

dataset.head()
sns.barplot(data=dataset, x='Title', y='Survived')

plt.xticks(rotation=45);
dataset['Title'].replace(['Capt.'],'Board',inplace=True)

dataset['Title'].replace(['Don.','Dona.','Lady.','Jonkheer.','Rev.','Sir.','the'],'Hon',inplace=True)

dataset['Title'].replace(['Col.','Dr.','Major.','Master.'],'Professionals',inplace=True)

dataset['Title'].replace(['Miss.','Mlle.','Mme.','Mr.','Mrs.','Ms.'],'NoHon', inplace=True)



test['Title'].replace(['Capt.'],'Board',inplace=True)

test['Title'].replace(['Don.','Dona.','Lady.','Jonkheer.','Rev.','Sir.','the'],'Hon',inplace=True)

test['Title'].replace(['Col.','Dr.','Major.','Master.'],'Professionals',inplace=True)

test['Title'].replace(['Miss.','Mlle.','Mme.','Mr.','Mrs.','Ms.'],'NoHon', inplace=True)
sns.barplot(data=dataset, x='Title',y='Survived')
dataset['Parch-SibSp'] = dataset['Parch'] + dataset['SibSp']

test['Parch-SibSp'] = test['Parch'] + test['SibSp']



sns.barplot(data=dataset, x = 'Parch-SibSp', y='Survived')



dataset.drop(columns=['Parch','SibSp'],inplace = True)

test.drop(columns=['Parch','SibSp'],inplace=True)
dataset['Parch-SibSp'] = pd.cut(x = dataset['Parch-SibSp'], bins=[0,1,4,7,20],right=False,labels=['Alone','Small','Medium','Big'])

test['Parch-SibSp'] = pd.cut(x = test['Parch-SibSp'], bins=[0,1,4,7,20],right=False,labels=['Alone','Small','Medium','Big'])
sns.barplot(data=dataset,x='Parch-SibSp',y='Survived')
print(np.unique([x.split(' ')[0] if not(x.isdigit()) else 'OnlyNumber' for x in dataset['Ticket']]))

dataset['Ticket'] = [x.split(' ')[0] if not(x.isdigit()) else 'OnlyNumber' for x in dataset['Ticket']]

test['Ticket'] = [x.split(' ')[0] if not(x.isdigit()) else 'OnlyNumber' for x in test['Ticket']]
sns.boxplot(data = dataset, x = 'Fare')


print('before(train) -> '+ str(len(dataset.loc[dataset.Fare == 0.0]['Fare'])))

print('before(test) -> '+ str(len(test.loc[test.Fare == 0.0]['Fare'])))



dataset['Fare'].replace(0.0,np.nan,inplace=True)

test['Fare'].replace(0.0,np.nan,inplace=True)



##Depending on the class we fill the Fare

meanFirst = dataset.loc[dataset.Pclass==1]['Fare'].mean()

print('First Class mean is ' + str(meanFirst))

meanSecond = dataset.loc[dataset.Pclass==2]['Fare'].mean()

print('Second Class mean is ' + str(meanSecond))

meanThird = dataset.loc[dataset.Pclass==3]['Fare'].mean()

print('Third Class mean is ' + str(meanThird))

means=[meanFirst,meanSecond,meanThird]





fcd = dataset.loc[dataset.Pclass == 1].index

scd = dataset.loc[dataset.Pclass == 2].index

tcd = dataset.loc[dataset.Pclass == 3].index



fct = test.loc[test.Pclass == 1].index

sct = test.loc[test.Pclass == 2].index

tct = test.loc[test.Pclass == 3].index



dataset.loc[fcd,'Fare'] = dataset.loc[fcd,'Fare'].fillna(means[0])

dataset.loc[scd,'Fare'] = dataset.loc[scd,'Fare'].fillna(means[1])

dataset.loc[tcd,'Fare'] = dataset.loc[tcd,'Fare'].fillna(means[2])



test.loc[fct,'Fare'] = test.loc[fct,'Fare'].fillna(means[0])

test.loc[sct,'Fare'] = test.loc[sct,'Fare'].fillna(means[1])

test.loc[tct,'Fare'] = test.loc[tct,'Fare'].fillna(means[2])



# dataset['Fare'].fillna(dataset['Fare'].mean(),inplace=True)

# test['Fare'].fillna(dataset['Fare'].mean(),inplace=True)



print('after(train) -> '+ str(len(dataset.loc[dataset.Fare == 0]['Fare'])))

print('after(test) -> '+ str(len(test.loc[dataset.Fare == 0]['Fare'])))

print('after(train) null -> '+ str(dataset['Fare'].count() -891))

print('after(test) null -> '+ str(test['Fare'].count() - 418))



minFare = 0

qntl25 = dataset['Fare'].quantile(0.25)

qntl50 = dataset['Fare'].quantile(0.5)

qntl75 = dataset['Fare'].quantile(0.75)

qntplus = dataset['Fare'].quantile(0.75) + 1.5*(qntl75-qntl50)

maxFare = 1e6



dataset['Fare'] = pd.cut(dataset['Fare'], bins= [minFare,qntl25, qntl75,qntplus,maxFare], labels=['Low','Medium','High','VeryHigh'])

test['Fare'] = pd.cut(test['Fare'], bins= [minFare,qntl25, qntl75,qntplus,maxFare], labels=['Low','Medium','High','VeryHigh'])



sns.barplot(data=dataset, x='Fare', y='Survived')
#Categorize



encSex = preprocessing.OrdinalEncoder().fit(pd.DataFrame(dataset['Sex']))

encTitle = preprocessing.OrdinalEncoder().fit(pd.DataFrame(dataset['Title']))

encParchSibSp = preprocessing.OrdinalEncoder().fit(pd.DataFrame(dataset['Parch-SibSp']))

encFare = preprocessing.OrdinalEncoder().fit(pd.DataFrame(dataset['Fare']))

encEmbark = preprocessing.OneHotEncoder().fit(pd.DataFrame(dataset['Embarked']))





dataset['Sex']= encSex.transform(pd.DataFrame(dataset['Sex']))



dataset['Title']= encTitle.transform(pd.DataFrame(dataset['Title']))



dataset['Parch-SibSp']= encParchSibSp.transform(pd.DataFrame(dataset['Parch-SibSp']))



dataset['Fare']= encFare.transform(pd.DataFrame(dataset['Fare']))



dataset = pd.concat([dataset,pd.DataFrame(encEmbark.transform(pd.DataFrame(dataset['Embarked'])).toarray(),columns=['Embark_C','Embark_Q','Embark_S'])], axis=1)

dataset.drop(columns=['Embarked'],inplace=True)



test = pd.concat([test,pd.DataFrame(encEmbark.transform(pd.DataFrame(test['Embarked'])).toarray(),columns=['Embark_C','Embark_Q','Embark_S'])], axis=1)

test.drop(columns=['Embarked'],inplace=True)



test['Sex'] = encSex.transform(pd.DataFrame(test['Sex']))



test['Title'] = encTitle.transform(pd.DataFrame(test['Title']))



test['Parch-SibSp'] = encParchSibSp.transform(pd.DataFrame(test['Parch-SibSp']))



test['Fare']= encFare.transform(pd.DataFrame(test['Fare']))
dataset.drop(columns=['Cabin','PassengerId','Name','Ticket','Survived'], inplace=True)

test.drop(columns=['Cabin','PassengerId','Name','Ticket'], inplace=True)
dataset.describe(include='all')
sns.barplot(x=dataset.columns, y=ensemble.RandomForestClassifier().fit(preprocessing.StandardScaler().fit_transform(dataset),y).feature_importances_)

plt.xticks(rotation=45)

plt.axhline(y=0.05,c='yellow')

plt.axhline(y=0.1,c='orange')

plt.axhline(y=0.15,c='red')



sel_threshold = 0.00;
if sel_threshold >= 0.05 and sel_threshold < 0.1:

    dataset.drop(columns=['Embark_S','Embark_Q','Embark_C','Title'],inplace=True)

    test.drop(columns=['Embark_S','Embark_Q','Embark_C','Title'],inplace=True)

if sel_threshold >= 0.1 and sel_threshold < 0.15:

    dataset.drop(columns=['Embark_S','Embark_Q','Embark_C','Parch-SibSp','Fare','Title'],inplace=True)

    test.drop(columns=['Embark_S','Embark_Q','Embark_C','Parch-SibSp','Fare','Title'],inplace=True)

if sel_threshold == 0.15:

    dataset.drop(columns=['Embark_S','Embark_Q','Embark_C','Parch-SibSp','Pclass','Title','Fare'],inplace=True)

    test.drop(columns=['Embark_S','Embark_Q','Embark_C','Parch-SibSp','Pclass','Title','Fare'],inplace=True)
sns.heatmap(dataset.corr(),cmap='Blues')
dataset.head()
test.head()
scaler = preprocessing.StandardScaler()

X = scaler.fit_transform(dataset)

test_X = scaler.transform(test)
cv_size = 5
lr_model = linear_model.LogisticRegression()

scores_lr = model_selection.cross_validate(lr_model,X,y,cv=cv_size, scoring=('accuracy','f1'))

print('Trained in ' + '{:.2f}'.format(scores_lr['fit_time'].mean()) + 's , achieved Accuracy of ' +  '{:.3f}'.format(scores_lr['test_accuracy'].mean()) + ' ('+ '{:.3f}'.format(scores_lr['test_accuracy'].std()) +')')
lr_model = linear_model.LogisticRegression()
results_rf = []

indexes = []

for j in range(100,300,100):

    for i in range(2,10):

        for k in range(2,10):

            indexes.append('n'+str(j)+'d'+str(i)+'s'+str(k));

            rf_model = ensemble.RandomForestClassifier(n_estimators=j,max_depth=i,min_samples_split=k, n_jobs=-1,oob_score = True)

            #scores_rf = model_selection.cross_validate(rf_model, X, y,cv=cv_size, scoring=('accuracy','f1'))

            scores_rf = rf_model.fit(X,y).oob_score_

            #print('Trained in ' + '{:.2f}'.format(scores_rf['fit_time'].mean()) + 's , achieved Accuracy of ' +  '{:.3f}'.format(scores_rf['test_accuracy'].mean()) + ' ('+ '{:.3f}'.format(scores_rf['test_accuracy'].std()) +')')

            #results_rf.append({'acc_mean': scores_rf['test_accuracy'].mean(), 'acc_std': scores_rf['test_accuracy'].std(), 'f1_mean': scores_rf['test_f1'].mean()})

            results_rf.append({'acc_mean': scores_rf})

results_rf = pd.DataFrame(results_rf, index=indexes)
# fig, ax = plt.subplots(1,1,figsize=(20,10))

# for i in range(3):

#     ax[i].xaxis.set_ticklabels(indexes ,rotation=90)

#     ax[i].grid()



# ax[0].plot(results_rf['acc_mean'], c = 'red')

# ax[0].set_ylabel('accuracy')

# ax[1].plot(results_rf['acc_std'], c = 'blue')

# ax[1].set_ylabel('standard deviation')

# ax[2].plot(results_rf['f1_mean'], c= 'purple')

# ax[2].set_ylabel('f1 score')



# mam = max(results_rf['acc_mean'])

# mas = min(results_rf['acc_std'])

# mfm = max(results_rf['f1_mean'])

# ax[0].axhline(y=mam,ls='--',c='green')

# ax[1].axhline(y=mas,ls='--',c='green')

# ax[2].axhline(y=mfm,ls='--',c='green')

plt.figure(figsize=(20,5))

plt.plot(results_rf['acc_mean'],c='red')

plt.grid()

plt.xticks(indexes,rotation=90)

plt.ylabel('accuracy')

plt.axhline(y=max(results_rf['acc_mean']))
rf_model = ensemble.RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_split=4, n_jobs=-1)
results_et = []

indexes = []

for j in range(100,300,100):

    for i in range(2,10):

        for k in range(2,10):

            indexes.append('n'+str(j)+'d'+str(i)+'s'+str(k));

            et_model = ensemble.ExtraTreesClassifier(n_estimators=j,max_depth=i,min_samples_split=k,n_jobs=-1,oob_score=True,bootstrap=True)

            #scores_et = model_selection.cross_validate(et_model, X, y,cv=cv_size, scoring=('accuracy','f1'))

            scores_et = et_model.fit(X,y).oob_score_

            #print('Trained in ' + '{:.2f}'.format(scores_rf['fit_time'].mean()) + 's , achieved Accuracy of ' +  '{:.3f}'.format(scores_rf['test_accuracy'].mean()) + ' ('+ '{:.3f}'.format(scores_rf['test_accuracy'].std()) +')')

            #results_et.append({'acc_mean': scores_et['test_accuracy'].mean(), 'acc_std': scores_et['test_accuracy'].std(), 'f1_mean': scores_et['test_f1'].mean()})

            results_et.append({'acc_mean': scores_et})



results_et = pd.DataFrame(results_et, index=indexes)
# fig, ax = plt.subplots(3,1,figsize=(20,15))

# for i in range(3):

#     ax[i].xaxis.set_ticklabels(indexes ,rotation=90)

#     ax[i].grid()



# ax[0].plot(results_et['acc_mean'], c = 'red')

# ax[0].set_ylabel('accuracy')

# ax[1].plot(results_et['acc_std'], c = 'blue')

# ax[1].set_ylabel('standard deviation')

# ax[2].plot(results_et['f1_mean'], c= 'purple')

# ax[2].set_ylabel('f1 score')



# mam = max(results_et['acc_mean'])

# mas = min(results_et['acc_std'])

# mfm = max(results_et['f1_mean'])

# ax[0].axhline(y=mam,ls='--',c='green')

# ax[1].axhline(y=mas,ls='--',c='green')

# ax[2].axhline(y=mfm,ls='--',c='green')



plt.figure(figsize=(20,5))

plt.plot(results_et['acc_mean'],c='red')

plt.grid()

plt.xticks(indexes,rotation=90)

plt.ylabel('accuracy')

plt.axhline(y=max(results_et['acc_mean']))
et_model = ensemble.ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_split=4,n_jobs=-1,bootstrap=True)
results_svc = []

for i in np.arange(0.1,10,0.1):

    svm_model = svm.SVC(C=i)

    scores_svc = model_selection.cross_validate(svm_model,X,y,cv=cv_size, scoring=('accuracy','f1'))

    #print('Trained in ' + '{:.2f}'.format(scores_svc['fit_time'].mean()) + 's , achieved Accuracy of ' +  '{:.3f}'.format(scores_svc['test_accuracy'].mean()) + ' ('+ '{:.3f}'.format(scores_svc['test_accuracy'].std()) +')')

    results_svc.append({'acc_mean': scores_svc['test_accuracy'].mean(), 'acc_std': scores_svc['test_accuracy'].std(), 'f1_mean': scores_svc['test_f1'].mean()})

results_svc = pd.DataFrame(results_svc, index=np.arange(0.1,10,0.1))
fig, ax = plt.subplots(3,1,figsize=(20,10))

for i in range(3):

    ax[i].xaxis.set_ticks(np.arange(0.1,10,0.1))

    ax[i].xaxis.set_ticklabels(['{:.1f}'.format(x) for x in np.arange(0.1,10,0.1)] ,rotation=90)

    ax[i].grid()



ax[0].plot(results_svc['acc_mean'], c = 'red')

ax[0].set_ylabel('accuracy')

ax[1].plot(results_svc['acc_std'], c = 'blue')

ax[1].set_ylabel('std deviation')

ax[2].plot(results_svc['f1_mean'], c= 'purple')

ax[2].set_ylabel('f1 score')



mam = max(results_svc['acc_mean'])

mas = min(results_svc['acc_std'])

mfm = max(results_svc['f1_mean'])

ax[0].axhline(y=mam,ls='--',c='green')

ax[1].axhline(y=mas,ls='--',c='green')

ax[2].axhline(y=mfm,ls='--',c='green')
svm_model = svm.SVC(C=0.5)
results_knn = []

for k in range(5,50):

    knn_model = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform',algorithm='ball_tree')

    scores_knn = model_selection.cross_validate(knn_model,X,y,cv=cv_size,scoring=('accuracy','f1'))

    #print('Trained in ' + '{:.2f}'.format(scores_knn['fit_time'].mean()) + 's , achieved Accuracy of ' +  '{:.3f}'.format(scores_knn['test_accuracy'].mean()) + ' ('+ '{:.3f}'.format(scores_knn['test_accuracy'].std()) +')')

    results_knn.append({'acc_mean': scores_knn['test_accuracy'].mean(), 'acc_std': scores_knn['test_accuracy'].std(), 'f1_mean': scores_knn['test_f1'].mean()})

results_knn = pd.DataFrame(results_knn, index=range(5,50))
fig, ax = plt.subplots(3,1,figsize=(15,10))

for i in range(3):

    ax[i].xaxis.set_ticks(range(1,50))

    ax[i].grid()



ax[0].plot(results_knn['acc_mean'], c = 'red')

ax[0].set_ylabel('accuracy')

ax[1].plot(results_knn['acc_std'], c = 'blue')

ax[1].set_ylabel('std deviation')

ax[2].plot(results_knn['f1_mean'], c= 'purple')

ax[1].set_ylabel('f1 score')



mam = max(results_knn['acc_mean'])

mas = min(results_knn['acc_std'])

mfm = max(results_knn['f1_mean'])

ax[0].axhline(y=mam,ls='--',c='green')

ax[1].axhline(y=mas,ls='--',c='green')

ax[2].axhline(y=mfm,ls='--',c='green')
knn_model = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform',algorithm='ball_tree')
nb_model = naive_bayes.GaussianNB()

scores_nb = model_selection.cross_validate(nb_model,X,y,cv=cv_size, scoring=('accuracy','f1'))

print('Trained in ' + '{:.2f}'.format(scores_nb['fit_time'].mean()) + 's , achieved Accuracy of ' +  '{:.3f}'.format(scores_nb['test_accuracy'].mean()) + ' ('+ '{:.3f}'.format(scores_nb['test_accuracy'].std()) +')')

    
nb_model = naive_bayes.GaussianNB()
#Stacking or Voting?

stacked_model = ensemble.StackingClassifier(estimators=[('et',et_model),

                                                        ('rf',rf_model),

                                                        ('svm',svm_model),

                                                        ('lr',lr_model)

                                                       ])



voting_model = ensemble.VotingClassifier(estimators=[('et',et_model),

                                                     ('rf',rf_model), 

                                                     ('svm',svm_model),

                                                     ('lr',lr_model)

                                                    ])

                                            

scores_stack = model_selection.cross_validate(stacked_model,X,y,cv=cv_size,scoring='accuracy')

scores_voting = model_selection.cross_validate(voting_model,X,y,cv=cv_size,scoring='accuracy')

print('Trained in ' + '{:.2f}'.format(scores_stack['fit_time'].mean()) + 's , achieved Accuracy of ' +  '{:.3f}'.format(scores_stack['test_score'].mean()) + ' ('+ '{:.3f}'.format(scores_stack['test_score'].std()) +')')

print('Trained in ' + '{:.2f}'.format(scores_voting['fit_time'].mean()) + 's , achieved Accuracy of ' +  '{:.3f}'.format(scores_voting['test_score'].mean()) + ' ('+ '{:.3f}'.format(scores_voting['test_score'].std()) +')')
models = {'lr':lr_model,

          'rf':rf_model,

          'et':et_model,

          'svm': svm_model,

          'knn': knn_model,

          'nb': nb_model,

          'stack': stacked_model,

          'vote':voting_model}
for i in models.keys():

    m = models[i].fit(X,y)

    y_pred = m.predict(test_X)

    prediction = pd.read_csv('../input/titanic/gender_submission.csv')

    prediction['Survived'] = y_pred

    prediction.set_index('PassengerId')

    prediction.to_csv('submission-'+i+'.csv',index = False)