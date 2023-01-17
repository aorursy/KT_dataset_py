import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
example_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
master = pd.concat([train, test], axis=0).reset_index(drop=True)
master.info()
# Not included in final report
# plt.figure(dpi=100)
# sns.set_style('whitegrid')
# sns.countplot(x='Survived',data=master)
# plt.title('Figure 2.1 - Surivival of the Training Passengers')
# #plt.savefig("Figure 2.1.png")
survival_rate = sum(train['Survived'] / len(train)) * 100
print(f"The percentage of passgengers in the training data who survived is: {survival_rate:0.1f}%")
master['Title'] = master.apply(lambda x: 'boy' if 'Master' in x['Name'] else 'man', axis=1)
master['Title'] = master.apply(lambda x: 'woman' if 'female' in x['Sex'] else x['Title'], axis=1)
df_plot = master.groupby(['Survived', 'Title']).size().reset_index().pivot(columns='Survived', index='Title', values=0)
df_plot.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('Figure 1 - Survival Rate by Title')
# plt.savefig("Figure 1.png")
woman_boy_survival_rate = (len(master.query('PassengerId<=891 and (Title == "woman" or Title == "boy") and Survived == 1')) 
 / len(master.query('PassengerId<=891 and (Title == "woman" or Title == "boy")'))) * 100
woman_boy_survival_rate
print(f"The percentage of women and boys in the training data who survived: {woman_boy_survival_rate:0.1f}%")
man_survival_rate = (len(master.query('PassengerId<=891 and Title == "man" and Survived == 1')) 
 / len(master.query('PassengerId<=891 and Title == "man"'))) * 100
man_survival_rate
print(f"The percentage of men in the training data who survived: {man_survival_rate:0.1f}%")
df_plot = master.groupby(['Survived', 'Pclass']).size().reset_index().pivot(columns='Survived', index='Pclass', values=0)
df_plot.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('Figure 2 - Survival by Passenger Class')
# plt.savefig("Figure 2.png")
first_survival_rate = (len(master.query('PassengerId<=891 and Pclass == 1 and Survived == 1')) 
 / len(master.query('PassengerId<=891 and Pclass == 1'))) * 100
print(f"The percentage of First-Class passengers in the training data who survived: {first_survival_rate:0.1f}%")
second_survival_rate = (len(master.query('PassengerId<=891 and Pclass == 2 and Survived == 1')) 
 / len(master.query('PassengerId<=891 and Pclass == 2'))) * 100
print(f"The percentage of Second-Class passengers in the training data who survived: {second_survival_rate:0.1f}%")
third_survival_rate = (len(master.query('PassengerId<=891 and Pclass == 3 and Survived == 1')) 
 / len(master.query('PassengerId<=891 and Pclass == 3'))) * 100
print(f"The percentage of Second-Class passengers in the training data who survived: {third_survival_rate:0.1f}%")
master['FamilySize'] = master['SibSp'] + master['Parch'] + 1
plt.figure(figsize=(16,8))
sns.catplot(x='FamilySize', y='Survived', data=master, kind='bar', aspect=2)
plt.title('Figure 3 - Family Size and Survival')
# plt.savefig("Figure 3.png")
df_plot = master.groupby(['Survived', 'Embarked']).size().reset_index().pivot(columns='Survived', index='Embarked', values=0)
df_plot.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('Figure 4 - Survival by Point of Embarkation')
# plt.savefig("Figure 2.4.png")
C_rate = (len(master.query('PassengerId<=891 and Embarked == "C"')) 
 / len(master.query('PassengerId<=891'))) * 100
print(f"The percentage of passengers in the training data who embarked at Cherbourg: {C_rate:0.1f}%")

C_survival_rate = (len(master.query('PassengerId<=891 and Embarked == "C" and Survived == 1')) 
 / len(master.query('PassengerId<=891 and Embarked == "C"'))) * 100
print(f"The percentage of Cherbourg passengers in the training data who survived: {C_survival_rate:0.1f}%")
Q_rate = (len(master.query('PassengerId<=891 and Embarked == "Q"')) 
 / len(master.query('PassengerId<=891'))) * 100
print(f"The percentage of passengers in the training data who embarked at Queenstown: {Q_rate:0.1f}%")

Q_survival_rate = (len(master.query('PassengerId<=891 and Embarked == "Q" and Survived == 1')) 
 / len(master.query('PassengerId<=891 and Embarked == "Q"'))) * 100
print(f"The percentage of Queenstown passengers in the training data who survived: {Q_survival_rate:0.1f}%")
S_rate = (len(master.query('PassengerId<=891 and Embarked == "S"')) 
 / len(master.query('PassengerId<=891'))) * 100
print(f"The percentage of passengers in the training data who embarked at Southampton: {S_rate:0.1f}%")

S_survival_rate = (len(master.query('PassengerId<=891 and Embarked == "S" and Survived == 1')) 
 / len(master.query('PassengerId<=891 and Embarked == "S"'))) * 100
print(f"The percentage of Southampton passengers in the training data who survived: {S_survival_rate:0.1f}%")
# Table 1
master.loc[master['Ticket'] == '110413']
master['PartySize'] = master['Ticket'].value_counts()[master.loc[:,'Ticket']].values
master['FareAdj'] = master['Fare'] / master['PartySize']
plt.figure(figsize=(8,5))
sns.heatmap(master[['Survived', 'Age', 'FareAdj']].corr(), annot=True, fmt='0.2f', cmap='coolwarm')
plt.title('Figure 5 - Survival, Age and Fare')
#  plt.savefig("Figure 5.png")
plot_df = master.groupby(['Survived', 'Age']).size().reset_index()
plot_df.loc[plot_df['Survived'] == 0]['Age'].hist(bins=10, figsize=(8,5))
plot_df.loc[plot_df['Survived'] == 1]['Age'].hist(bins=10, figsize=(8,5), alpha=0.5)
plt.legend(['0','1'])
plt.xlabel("Age")
plt.ylabel("Count")
plt.title('Figure 6 - Distribution of Ages')
# plt.savefig("Figure 6.png")
plot_df = master.groupby(['Survived', 'FareAdj']).size().reset_index()
plot_df.loc[plot_df['Survived'] == 0]['FareAdj'].hist(bins=20, figsize=(8,5))
plot_df.loc[plot_df['Survived'] == 1]['FareAdj'].hist(bins=20, figsize=(8,5), alpha=0.5)
plt.legend(['0','1'])
plt.xlabel("Adjusted Per Person Fare")
plt.ylabel("Count")
plt.title('Figure 7 - Distribution of Fares')
# plt.savefig("Figure 7.png")
master.isnull().sum()
all_ages = master.loc[master['Age'] > 0].copy()
no_ages = master.loc[pd.isnull(master['Age'])].copy()
X_train = pd.get_dummies(all_ages[['Title', 'Pclass', 'SibSp', 'Parch']])
y_train = all_ages['Age']
X_test = pd.get_dummies(no_ages[['Title', 'Pclass', 'SibSp', 'Parch']])
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
missing_ages = pd.DataFrame(dtr.predict(X_test), index=X_test.index, columns=['Age'])
master['Age'] = master['Age'].combine_first(missing_ages['Age'])
master.isnull().sum()
all_fares = master.loc[master['Fare'] > 0].copy()
no_fares = master.loc[pd.isnull(master['Fare'])].copy()
X_train = pd.get_dummies(all_fares[['Title', 'Pclass', 'Embarked']])
y_train = all_fares['Fare']
X_test = pd.get_dummies(no_fares[['Title', 'Pclass', 'Embarked']])
X_test = X_test.reindex(columns=X_train.columns)
X_test = X_test.fillna(0)
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
missing_fares = pd.DataFrame(dtr.predict(X_test), index=X_test.index, columns=['Fare'])
master['Fare'] = master['Fare'].combine_first(missing_fares['Fare'])
master['FareAdj'] = master['Fare'] / master['PartySize']
master.isnull().sum()
X_men = master[(master['PassengerId'] <= 891) & (master['Title'] == 'man')]
fig, axes = plt.subplots(2,2, figsize=(15,5))
sns.kdeplot(X_men.loc[X_men['Survived']==0.0]['Age'], shade=True, color='r', ax=axes[0,0], legend=False)
sns.kdeplot(X_men.loc[X_men['Survived']==1.0]['Age'], shade=True, color='g', ax=axes[0,0], legend=False)
axes[0,0].set_title('Age')
sns.kdeplot(X_men.loc[X_men['Survived']==0.0]['FareAdj'], shade=True, color='r', ax=axes[0,1], legend=False)
sns.kdeplot(X_men.loc[X_men['Survived']==1.0]['FareAdj'], shade=True, color='g', ax=axes[0,1], legend=False)
axes[0,1].set_title('FareAdj')
axes[0,1].set_xlim(0,50)
sns.kdeplot(X_men.loc[X_men['Survived']==0.0]['FamilySize'], shade=True, color='r', ax=axes[1,0], bw=0, legend=False)
sns.kdeplot(X_men.loc[X_men['Survived']==1.0]['FamilySize'], shade=True, color='g', ax=axes[1,0], bw=0, legend=False)
axes[1,0].set_title('FamilySize')
axes[1,0].set_xlim(1.5,6.5)
sns.kdeplot(X_men.loc[X_men['Survived']==0.0]['Pclass'], shade=True, color='r', ax=axes[1,1], legend=False)
sns.kdeplot(X_men.loc[X_men['Survived']==1.0]['Pclass'], shade=True, color='g', ax=axes[1,1], legend=False)
axes[1,1].set_title('Pclass')
fig.legend(['Non-Survived','Survived'])
fig.tight_layout()
# plt.savefig("Figure 8.png")
y = X_men['Survived']
x1 = X_men['FareAdj']/10
x2 = (X_men['FamilySize'])+(X_men['Age']/70)
Pclass = X_men['Pclass']
X_men_2feats = pd.concat([y.astype(int),x1,x2,Pclass], axis=1)
X_men_2feats.columns = ['Survived', 'x1', 'x2', 'Pclass']
X_men_2feats.info()
plt.figure(figsize=(8,5))
plt_df = X_men_2feats.query('x1<6 and x2<6').groupby(['Pclass', 'x1']).size().reset_index()
plt_df.loc[plt_df['Pclass'] == 1]['x1'].hist(bins=12, alpha=0.7)
plt_df.loc[plt_df['Pclass'] == 2]['x1'].hist(bins=12, alpha=0.7)
plt_df.loc[plt_df['Pclass'] == 3]['x1'].hist(bins=12, alpha=0.7)
plt.legend([1,2,3])
plt.title('Figure 9 - Adjusted Fare Distribution by Class')
plt.xlabel('x1=Adjusted Fare/10')
# plt.savefig("Figure 9.png")
plt.figure(figsize=(8,5))
sns.scatterplot(x='x1', y='x2', hue='Survived', style='Pclass', data=X_men_2feats.query('x1<6 and x2<6'), alpha=0.5)
plt.xlabel('x1=Adjusted Fare/10')
plt.ylabel('x2=FamilySize + Age/70')
plt.title('Figure 10 - Survival Patterns of Adult Males')
# plt.savefig("Figure 10.png")
x1s = np.linspace(0,5,100)
x2s = np.linspace(1,3,100)
x1 = np.repeat(x1s, 100)
x2 = [j for i in range(100) for j in x2s]
g = pd.DataFrame([x1,x2]).T
g.columns = ['x1','x2']
from xgboost import XGBClassifier
xgb = XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1, gamma=0.1, colsample_bytree=1, min_child_weight=1)
history = xgb.fit(X_men_2feats[['x1','x2']], X_men_2feats['Survived'], eval_metric='error')
pred = xgb.predict(g[['x1','x2']])
print(f'The predicted survival rate for adult males is only {(sum(pred)/10000)*100}%, so they are going to be hard to find!')
g['Survived'] = pred
plt.figure(figsize=(10,7))
sns.scatterplot(x='x1', y='x2', hue='Survived', palette='pastel', data=g, alpha=0.6)
sns.scatterplot(x='x1', y='x2', style='Pclass', hue='Survived', palette='bright',data=X_men_2feats.query('x1<5 and x2<3'))
plt.legend()
plt.xlabel('x1=Adjusted Fare/10')
plt.ylabel('x2=FamilySize + Age/70')
plt.title('Figure 11 - XGBoost Predicted Patterns of Adult Male Survival')
# plt.savefig("Figure 11.png")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_men_2feats[['x1','x2']], X_men_2feats['Survived'], test_size=.3, random_state=42)
xgb = XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1, gamma=0.1, colsample_bytree=1, min_child_weight=1)
history = xgb.fit(X_train, y_train, eval_metric='error')
y_score = xgb.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve, auc, accuracy_score, average_precision_score, precision_recall_curve, plot_precision_recall_curve, classification_report
# Calculate the False Positive Rate and True Positive Rate
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5,5))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Figure 3.5 - Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("Figure 12.png")
y_pred = np.array(list(map(lambda x: 1 if x>0.50 else 0, y_score)))
# print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
y_pred = np.array(list(map(lambda x: 1 if x>0.75 else 0, y_score)))
print(classification_report(y_test,y_pred))
y_pred = np.array(list(map(lambda x: 1 if x>0.90 else 0, y_score)))
print(classification_report(y_test,y_pred))
y_pred = np.array(list(map(lambda x: 1 if x>0.92 else 0, y_score)))
print(classification_report(y_test,y_pred))
average_precision = average_precision_score(y_test, y_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
disp = plot_precision_recall_curve(xgb, X_test, y_test)
disp.ax_.set_title('Figure 13 - 2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))
# disp.ax_
from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb, X_men_2feats[['x1','x2']], X_men_2feats['Survived'], cv=10)
print("Scores: ")
print(scores)
print(f'Mean cross validation score: {scores.mean()*100:0.2f}%')
dataB = master[(master['PassengerId'] <= 891) & (master['Title'] == 'man')]
dataC = master[(master['PassengerId'] > 891) & (master['Title'] == 'man')]
dataTrain = pd.concat([dataB['Survived'].astype(int), dataB['FareAdj']/10, dataB['FamilySize']+(dataB['Age']/70)], axis=1)
dataTrain.columns=['Survived', 'x1', 'x2']
dataTest = pd.concat([dataC['Survived'], dataC['FareAdj']/10, dataC['FamilySize']+(dataC['Age']/70), dataC['PassengerId']], axis=1)
dataTest.columns=['Survived', 'x1', 'x2', 'PassengerId']
xgb = XGBClassifier(objective='binary:logistic', max_depth=4, learning_rate=0.1, gamma=0.1, colsample_bytree=1, min_child_weight=1)
history = xgb.fit(dataTrain[['x1','x2']], dataTrain['Survived'], eval_metric='error')
y_score = xgb.predict_proba(dataTest[['x1','x2']])[:,1]
y_pred = pd.DataFrame(np.array(list(map(lambda x: 1 if x>0.90 else 0, y_score))), index=dataTest.index)
y_pred.columns=['Survived']
sum(y_pred['Survived'])
master['Surname'] = master['Name'].apply(lambda x: x.split(',')[0])
import re
master['TicketX'] = master['Ticket'].apply(lambda x: re.sub('.$', 'X', x))
x = master[['Surname', 'Pclass', 'TicketX', 'Fare', 'Embarked']].to_string(header=False, index=False, index_names=False).split('\n')
vals = ['-'.join(i.split()) for i in x]
master['GroupId'] = vals
master.loc[master['Title'] == 'man', 'GroupId'] = 'NoGroup'
master['GroupId'].isnull().sum()
master.loc[master['Name'].apply(lambda x: 'Needs' in x)]
master.loc[892, 'GroupId'] = master.loc[774, 'GroupId']
master.loc[master.GroupId=='Richards-2-2910X-18.75000-S','GroupId'] = 'Hocking-2-2910X-23.00000-S'
master.loc[529,'GroupId'] = 'Hocking-2-2910X-23.00000-S'
master.loc[master.GroupId=='Hocking-2-2910X-23.00000-S']
master['GroupFreq'] = master['GroupId'].value_counts()[master.loc[:,'GroupId']].values
master.loc[master['GroupFreq'] == 1, 'GroupId'] = 'NoGroup'
master['GroupFreq'] = master['GroupId'].value_counts()[master.loc[:,'GroupId']].values
master['GroupId'].nunique()-1 # We don't count NoGroup
x = master[['Pclass', 'TicketX', 'Fare', 'Embarked']].to_string(header=False, index=False, index_names=False).split('\n')
master['TicketId'] = ['-'.join(i.split()) for i in x]
idx = master.query('Title != "man" and GroupId == "NoGroup"').index
print("Current number of single women and boys: ", len(idx))
for i in idx:
    z = master['GroupId'][master['TicketId'] == master['TicketId'][i]]
    q = [j for j in z if j != "NoGroup"]
    if len(q) > 0:
        master.loc[i,'GroupId'] = q[0]
print("Number of nannies and other female relatives found and added to their repsective families: ",
      len(idx) - len(master.query('Title != "man" and GroupId == "NoGroup"').index) )
master['GroupSurvival'] = master.groupby('GroupId')['Survived'].mean()[master.loc[:,'GroupId']].values
master['GroupSurvival'].value_counts()
master.query('GroupSurvival == "NaN"')['Name'].count()
idx = master.query('GroupSurvival == "NaN" and Pclass == 3').index
master.loc[idx, 'GroupSurvival'] = 0
idx = master.query('GroupSurvival == "NaN" and Pclass == 1').index
master.loc[idx, 'GroupSurvival'] = 1
master['GroupSurvival'].value_counts()
master['Predict'] = 0 
master.loc[master.Sex =="female", 'Predict'] = 1
idx = master.query('Title == "woman" and GroupSurvival == 0').index
master.loc[idx, 'Predict'] = 0
idx = master.query('Title == "boy" and GroupSurvival == 1').index
master.loc[idx, 'Predict'] = 1
master.query('Sex == "male" and Predict == 1 and PassengerId > 891')['Name']
master.query('Sex == "female" and Predict == 0 and PassengerId > 891')['Name']
submission = master.loc[891:, ['PassengerId', 'Predict']]
submission.reset_index(drop=True, inplace=True)
submission.rename(columns={'Predict':'Survived'}, inplace=True)
#submission.to_csv('2020-03-30_WCG.csv', index=False)
# Groups in the training data who either all survived or all perished
WCGtrain = master.query('PassengerId<=891 and (GroupSurvival==0 or GroupSurvival==1)')
# Groups in test data who either all survived or all perished
WCGtest = master.query('PassengerId>891 and (GroupSurvival==0 or GroupSurvival==1)')
# Single Women in the trainign set
dataB = master.query('PassengerId<=891 and Title=="woman" and FamilySize==1')
# Single Women in the test set
dataC = master.query('PassengerId>891 and Title=="woman" and FamilySize==1')
# Drop women from the test set who belonged to groups that either entirely survived or entirely perished
Cset = set(dataC.index)
WCGset = set(WCGtest.index)
drop_list = list(Cset & WCGset)
#drop_list
dataC.drop(drop_list, inplace=True, axis=0)
y = dataB['Survived']
x1 = dataB['FareAdj']/10
x2 = dataB['Age']/15
PassId = dataB['PassengerId']
Pclass = dataB['Pclass']
dataTrain = pd.concat([y,x1,x2,PassId,Pclass], axis=1)
dataTrain.columns=['Survived', 'x1','x2','PassengerId','Pclass']
y = dataC['Survived']
x1 = dataC['FareAdj']/10
x2 = dataC['Age']/15
PassId = dataC['PassengerId']
Pclass = dataC['Pclass']
dataTest = pd.concat([y,x1,x2,PassId,Pclass], axis=1)
dataTest.columns=['Survived', 'x1','x2','PassengerId','Pclass']
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = SVC(probability=True, random_state=1)
clf4 = DecisionTreeClassifier(max_depth=2)
clf5 = KNeighborsClassifier(n_neighbors=3)
eclf_hard = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('dtc', clf4), ('knn', clf5)], voting='hard', weights=[0.6,0.6,0.6,0.9,1])
eclf_soft = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('dtc', clf4), ('knn', clf5)], voting='soft', weights=[0.6,0.6,0.6,0.9,1])
classifiers = ['Logistic Regression', 'Random Forest', 'SVM', 'DecisionTree', 'KNN', 'HardVoteEnsemble', 'SoftVoteEnsemble']
print('\n5-fold cross validation:')
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf_hard, eclf_soft], classifiers):
    scores = model_selection.cross_val_score(clf, dataTrain[['x1','x2','Pclass']], dataTrain['Survived'], cv=5, scoring='accuracy')
    print(f"\tAccuracy: {scores.mean():0.2f} [{label}]")
eclf_soft.fit(dataTrain[['x1','x2','Pclass']], dataTrain['Survived'])
y_score = eclf_soft.predict_proba(dataTest[['x1','x2','Pclass']])[:,1]
# Apply a threshold function to tune the predictions 
y_pred = pd.DataFrame(np.array(list(map(lambda x: 0 if x<=0.30 else 1, y_score))), index=dataTest.index)
y_pred.columns=['Survived']
dataTest['Survived'] = y_pred['Survived']
fatal_idx = dataTest.loc[dataTest.Survived == 0].index
master2 = master.copy()
master2.loc[fatal_idx,'Predict'] = 0
fatal_ids = master2.loc[fatal_idx,'PassengerId'].values
submission2 = submission.copy()
submission2.loc[submission2['PassengerId'].isin(fatal_ids)]
submission2.loc[submission2['PassengerId'].isin(fatal_ids),'Survived'] = 0
#submission2.to_csv('2020-03-28_Ensemble.csv', index=False)