import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/adult-census-income/adult.csv")

df.head()
df.shape
df.info()
df.columns = ['age','workclass','finalweight',

              'education','educationNumber','maritalStatus',

              'occupation','relationship','race',

              'sex','capitalGain','capitalLoss',

              'hoursperweek','nativeCountry','income']

df.head()
df.describe().T
listItem = []

for col in df.columns :

    listItem.append([col, df[col].dtype, df[col].isna().sum(), round((df[col].isna().sum()/len(df[col]))*100,2),

                     df[col].nunique(), list(df[col].unique()[:3])])

dfDesc = pd.DataFrame( columns=['dataFeatures','dataType','null','nullPct','unique','uniqueSample'], data=listItem)

dfDesc
df['income'] = df['income'].apply(

    lambda x : 1 if x != '<=50K' else 0

)
sns.set_style('whitegrid')

sns.countplot(x=df['income'],palette='Set1')
categorical = [i for i in df.columns.drop(['income','nativeCountry']) if df[i].dtype == 'O']
plt.figure(figsize=(10,60))

for i in range(len(categorical)) :

    plt.subplot(7,1,i+1)

    sns.countplot(x='income', hue=f'{categorical[i]}', data=df, palette='Set1')

    plt.xticks(rotation=90)

plt.show()
# to see the ratio income per category in each feature

df.groupby(['sex'])['income'].mean() 
plt.figure(figsize=(10,80))

for i in range(len(categorical)) :

    plt.subplot(7,1,i+1)

    sns.countplot(x=f'{categorical[i]}', hue='income', data=df)

    plt.xticks(rotation=90)

plt.show()
df['occupation'].value_counts()
display(df[df['occupation']== '?'].head())

print(df[df['occupation']== '?'].shape)
df[(df['occupation']=='?')&(df['workclass']!='?')]
sns.countplot(df[df['occupation']=='?']['occupation'],hue=df['income'])
sns.countplot(df[df['occupation']=='Prof-specialty']['occupation'],hue=df['income'])
df['nativeCountry'].value_counts()
plt.figure(figsize=(15,10))

sns.countplot('nativeCountry',hue='income',data=df)

plt.xticks(rotation=90)

plt.show()
df['workclass'].value_counts()
df[(df['workclass']=='?')&(df['occupation']!='?')]
sns.countplot(df[df['workclass']=='Private']['workclass'],hue=df['income'])
sns.countplot(df[df['workclass']=='?']['workclass'],hue=df['income'])
df[['education','educationNumber']].sort_values(by=['educationNumber']).head()
listEdu = list(df['education'].unique())
listItem = []

for i in listEdu:

    listItem.append([i,np.unique(df[df['education']==i]['educationNumber'])[0]])
dfEdu = pd.DataFrame(listItem,columns=['education','educationNumber']).sort_values(by=['educationNumber'])

dfEdu = dfEdu.reset_index()

dfEdu.drop('index',axis=1,inplace=True)

dfEdu
df = df.drop('education',axis=1)
numerical = [i for i in df.columns.drop(['income','nativeCountry']) if df[i].dtype != 'O']
plt.figure(figsize=(15,8))

sns.heatmap(df.corr(), annot=True)
sns.pairplot(pd.concat([df[numerical],df['income']],axis=1),hue="income")
sns.distplot(df[df['income'] == 0]['age'], kde=True, color='darkred', bins=30)

sns.distplot(df[df['income'] == 1]['age'], kde=True, color='blue', bins=30)
sns.countplot(df[df['age']==90]['occupation'], palette = 'Set1')

plt.xticks(rotation=90)

plt.show()
sns.distplot(df[df['age']==90]['hoursperweek'])

plt.xticks(rotation=90)

plt.show()
df[(df['age']==90)&(df['hoursperweek']>70)]
sns.distplot(df[df['income'] == 0]['finalweight'], kde=True, color='darkred', bins=30)

sns.distplot(df[df['income'] == 1]['finalweight'], kde=True, color='blue', bins=30)
plt.figure(figsize=(8,10))

sns.distplot(df['capitalGain'],kde=False)

plt.show()
plt.figure(figsize=(8,6))

sns.countplot(df[(df['capitalGain']>0)]['workclass'],hue=df['income'])

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(8,10))

sns.distplot(df['hoursperweek'],kde=False)

plt.show()
df[df['hoursperweek']>70]['occupation'].value_counts().plot(kind='bar',title='Occupation hours')
sns.countplot(df[df['hoursperweek']>80]['occupation'],hue=df['income'])

plt.xticks(rotation=90)

plt.show()

print('Total',df[df['hoursperweek']>80]['occupation'].count())
df['occupation'] = df[['occupation','workclass']].apply(lambda x : 'None' if x['occupation'] == '?' and x['workclass']=='Never-worked' else x['occupation'],axis=1)
listdrop = df[(df['occupation']=='?')&(df['workclass']=='?')].index

df.drop(listdrop,axis=0,inplace=True)
df[numerical].isnull().sum()
df.describe()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
categorical = [i for i in df.columns.drop(['income']) if df[i].dtype == 'O']

print(categorical)
lewc = LabelEncoder()

lemar = LabelEncoder()

leoc = LabelEncoder()

lerl = LabelEncoder()

lerc = LabelEncoder()

lesx = LabelEncoder()

lenc = LabelEncoder()
lewc.fit(df['workclass'])

lemar.fit(df['maritalStatus'])

leoc.fit(df['occupation'])

lerl.fit(df['relationship'])

lerc.fit(df['race'])

lesx.fit(df['sex'])

lenc.fit(df['nativeCountry'])
# with open('lewc.pickle', 'wb') as f:

#     pickle.dump(lewc, f)

# with open('lemar.pickle', 'wb') as f:

#     pickle.dump(lemar, f)

# with open('leoc.pickle', 'wb') as f:

#     pickle.dump(leoc, f)

# with open('lerl.pickle', 'wb') as f:

#     pickle.dump(lerl, f)

# with open('lerc.pickle', 'wb') as f:

#     pickle.dump(lerc, f)

# with open('lesx.pickle', 'wb') as f:

#     pickle.dump(lesx, f)

# with open('lenc.pickle', 'wb') as f:

#     pickle.dump(lenc, f)
df['workclass'] = lewc.transform(df['workclass'])

df['maritalStatus'] = lemar.transform(df['maritalStatus'])

df['occupation'] = leoc.transform(df['occupation'])

df['relationship'] = lerl.transform(df['relationship'])

df['race'] = lerc.transform(df['race'])

df['sex'] = lesx.transform(df['sex'])

df['nativeCountry'] = lenc.transform(df['nativeCountry'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df[numerical])
# with open('scaler.pickle', 'wb') as f:

#     pickle.dump(scaler, f)
dfScaled = pd.DataFrame(scaler.transform(df[numerical]),columns=df[numerical].columns)
dfLabeled = pd.DataFrame(df[categorical],columns=df[categorical].columns)

dfLabeled = dfLabeled.reset_index(drop=True)
dfScaled = pd.concat([dfLabeled,dfScaled],axis=1)
dfScaled.head()
data = dfScaled.drop(['nativeCountry'],axis=1)

target= df['income']
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import  XGBClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, auc, log_loss, matthews_corrcoef,roc_auc_score, f1_score

from sklearn.model_selection import GridSearchCV, learning_curve,KFold, train_test_split
def calc_train_error(X_train, y_train, model):

#     '''returns in-sample error for already fit model.'''

    predictions = model.predict(X_train)

    predictProba = model.predict_proba(X_train)

    accuracy = accuracy_score(y_train, predictions)

    f1 = f1_score(y_train, predictions, average='macro')

    roc_auc = roc_auc_score(y_train, predictProba[:,1])

    logloss = log_loss(y_train,predictProba[:,1])

    report = classification_report(y_train, predictions)

    lossBuatan = (abs((y_train-predictProba[:,1]))).mean()

    return { 

        'report': report, 

        'f1' : f1, 

        'roc': roc_auc, 

        'accuracy': accuracy,

        'logloss': logloss,

        'lossBuatan': lossBuatan

    }

    

def calc_validation_error(X_test, y_test, model):

#     '''returns out-of-sample error for already fit model.'''

    predictions = model.predict(X_test)

    predictProba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(y_test, predictions, average='macro')

    roc_auc = roc_auc_score(y_test, predictProba[:,1])

    logloss = log_loss(y_test,predictProba[:,1])

    report = classification_report(y_test, predictions)

    lossBuatan = (abs((y_test-predictProba[:,1]))).mean()

    return { 

        'report': report, 

        'f1' : f1, 

        'roc': roc_auc, 

        'accuracy': accuracy,

        'logloss': logloss,

        'lossBuatan':lossBuatan

    }

    

def calc_metrics(X_train, y_train, X_test, y_test, model):

#     '''fits model and returns the in-sample error and out-of-sample error'''

    model.fit(X_train, y_train)

    train_error = calc_train_error(X_train, y_train, model)

    validation_error = calc_validation_error(X_test, y_test, model)

    return train_error, validation_error
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=.3,random_state=101)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_errors = []

validation_errors = []

for train_index, val_index in kf.split(data, target):

    

    # split data

    X_train, X_val = data.iloc[train_index], data.iloc[val_index]

    y_train, y_val = target.iloc[train_index], target.iloc[val_index]



    # instantiate model

    logreg = LogisticRegression(solver='lbfgs')



    #calculate errors

    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, logreg)

    train_errors.append(train_error)

    validation_errors.append(val_error)
listItem = []



for tr,val in zip(train_errors,validation_errors) :

    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],

                     tr['logloss'],val['logloss']])



listItem.append(list(np.mean(listItem,axis=0)))

    

dfEvalLR = pd.DataFrame(listItem, 

                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 

                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss'])

listIndex = list(dfEvalLR.index)

listIndex[-1] = 'Average'

dfEvalLR.index = listIndex

dfEvalLR
for item,rep in zip(range(1,6),train_errors) :

    print('Report Train ke ',item,':')

    print(rep['report'])
for item,rep in zip(range(1,6),validation_errors) :

    print('Report Test ke ',item,':')

    print(rep['report'])
train_sizes, train_scores, test_scores = learning_curve(estimator=logreg,

                                                       X=data,

                                                       y=target,

                                                       train_sizes=np.linspace(0.3, 0.8, 5),

                                                       cv=10,

                                                       scoring='accuracy')



print('\nTrain Scores : ')

print(train_scores)

# Mean value of accuracy against training data

train_mean = np.mean(train_scores, axis=1)

print('\nTrain Mean : ')

print(train_mean)

print('\nTrain Size : ')

print(train_sizes)

# Standard deviation of training accuracy per number of training samples

train_std = np.std(train_scores, axis=1)

print('\nTrain Std : ')

print(train_std)



# Same as above for test data

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



print('\nTest Scores : ')

print(test_scores)

print('\nTest Mean : ')

print(test_mean)

print('\nTest Std : ')

print(test_std)



# Plot training accuracies 

plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot the variance of training accuracies

plt.fill_between(train_sizes,

                train_mean + train_std,

                train_mean - train_std,

                alpha=0.15, color='red')



# Plot for test data as training data

plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 

        label='Test Accuracy')

plt.fill_between(train_sizes,

                test_mean + test_std,

                test_mean - test_std,

                alpha=0.15, color='blue')



plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
predictProbaTrain=logreg.predict_proba(X_train)
pred = predictProbaTrain[:,1]

fpr, tpr, threshold = roc_curve(y_train, pred)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
predictProbaTest=logreg.predict_proba(X_test)
preds = predictProbaTest[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
train_errors = []

validation_errors = []

for train_index, val_index in kf.split(data, target):

    

    # split data

    X_train, X_val = data.iloc[train_index], data.iloc[val_index]

    y_train, y_val = target.iloc[train_index], target.iloc[val_index]



    # instantiate model

    dtree = DecisionTreeClassifier(max_depth=7,min_samples_leaf=25)



    #calculate errors

    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, dtree)



    # append to appropriate list

    train_errors.append(train_error)

    validation_errors.append(val_error)
listItem = []



for tr,val in zip(train_errors,validation_errors) :

    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],

                     tr['logloss'],val['logloss'],tr['lossBuatan'],val['lossBuatan']])



listItem.append(list(np.mean(listItem,axis=0)))

    

dfEvalDTC = pd.DataFrame(listItem, 

                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 

                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss','Train loss Buatan','Test Loss Buatan'])

listIndex = list(dfEvalDTC.index)

listIndex[-1] = 'Average'

dfEvalDTC.index = listIndex

dfEvalDTC
for item,rep in zip(range(1,6),train_errors) :

    print('Report Train ke ',item,':')

    print(rep['report'])
for item,rep in zip(range(1,6),validation_errors) :

    print('Report Test ke ',item,':')

    print(rep['report'])
train_sizes, train_scores, test_scores = learning_curve(estimator=dtree,

                                                       X=data,

                                                       y=target,

                                                       train_sizes=np.linspace(0.3, 0.9, 5),

                                                       cv=10,

                                                       scoring='accuracy')



print('\nTrain Scores : ')

print(train_scores)

# Mean value of accuracy against training data

train_mean = np.mean(train_scores, axis=1)

print('\nTrain Mean : ')

print(train_mean)

print('\nTrain Size : ')

print(train_sizes)

# Standard deviation of training accuracy per number of training samples

train_std = np.std(train_scores, axis=1)

print('\nTrain Std : ')

print(train_std)



# Same as above for test data

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



print('\nTest Scores : ')

print(test_scores)

print('\nTest Mean : ')

print(test_mean)

print('\nTest Std : ')

print(test_std)



# Plot training accuracies 

plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot the variance of training accuracies

plt.fill_between(train_sizes,

                train_mean + train_std,

                train_mean - train_std,

                alpha=0.15, color='red')



# Plot for test data as training data

plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 

        label='Test Accuracy')

plt.fill_between(train_sizes,

                test_mean + test_std,

                test_mean - test_std,

                alpha=0.15, color='blue')



plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
predictProbaTrain=dtree.predict_proba(X_train)
pred = predictProbaTrain[:,1]

fpr, tpr, threshold = roc_curve(y_train, pred)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
predictProbaTest=dtree.predict_proba(X_test)
preds = predictProbaTest[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
coef1 = pd.Series(dtree.feature_importances_,data.columns).sort_values(ascending=False)

coef1.plot(kind='bar', title='Feature Importances')
train_errors = []

validation_errors = []

for train_index, val_index in kf.split(data, target):

    

    # split data

    X_train, X_val = data.iloc[train_index], data.iloc[val_index]

    y_train, y_val = target.iloc[train_index], target.iloc[val_index]

    

    # instantiate model

    rfc = RandomForestClassifier(n_estimators=300,max_depth=3,min_samples_leaf=10,random_state=101)



    #calculate errors

    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, rfc)



    # append to appropriate list

    train_errors.append(train_error)

    validation_errors.append(val_error)
listItem = []



for tr,val in zip(train_errors,validation_errors) :

    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],

                     tr['logloss'],val['logloss']])



listItem.append(list(np.mean(listItem,axis=0)))

    

dfEvalRFC = pd.DataFrame(listItem, 

                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 

                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss'])

listIndex = list(dfEvalRFC.index)

listIndex[-1] = 'Average'

dfEvalRFC.index = listIndex

dfEvalRFC
for item,rep in zip(range(1,6),train_errors) :

    print('Report Train ke ',item,':')

    print(rep['report'])
for item,rep in zip(range(1,6),validation_errors) :

    print('Report Test ke ',item,':')

    print(rep['report'])
train_sizes, train_scores, test_scores = learning_curve(estimator=rfc,

                                                       X=data,

                                                       y=target,

                                                       train_sizes=np.linspace(0.3, 0.9, 5),

                                                       cv=10,

                                                       scoring='accuracy')



print('\nTrain Scores : ')

print(train_scores)

# Mean value of accuracy against training data

train_mean = np.mean(train_scores, axis=1)

print('\nTrain Mean : ')

print(train_mean)

print('\nTrain Size : ')

print(train_sizes)

# Standard deviation of training accuracy per number of training samples

train_std = np.std(train_scores, axis=1)

print('\nTrain Std : ')

print(train_std)



# Same as above for test data

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



print('\nTest Scores : ')

print(test_scores)

print('\nTest Mean : ')

print(test_mean)

print('\nTest Std : ')

print(test_std)



# Plot training accuracies 

plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot the variance of training accuracies

plt.fill_between(train_sizes,

                train_mean + train_std,

                train_mean - train_std,

                alpha=0.15, color='red')



# Plot for test data as training data

plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 

        label='Test Accuracy')

plt.fill_between(train_sizes,

                test_mean + test_std,

                test_mean - test_std,

                alpha=0.15, color='blue')



plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
predictProbaTrain=rfc.predict_proba(X_train)
pred = predictProbaTrain[:,1]

fpr, tpr, threshold = roc_curve(y_train, pred)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
predictProbaTest=rfc.predict_proba(X_test)
preds = predictProbaTest[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
coef1 = pd.Series(rfc.feature_importances_,data.columns).sort_values(ascending=False)

coef1.plot(kind='bar', title='Feature Importances')
train_errors = []

validation_errors = []

for train_index, val_index in kf.split(data, target):

    

    # split data

    X_train, X_val = data.iloc[train_index], data.iloc[val_index]

    y_train, y_val = target.iloc[train_index], target.iloc[val_index]



    # instantiate model

    xgb = XGBClassifier(max_depth=10,min_child_weight=10, n_estimators=250, learning_rate=0.1)



    #calculate errors

    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, xgb)



    # append to appropriate list

    train_errors.append(train_error)

    validation_errors.append(val_error)
listItem = []



for tr,val in zip(train_errors,validation_errors) :

    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],

                     tr['logloss'],val['logloss']])



listItem.append(list(np.mean(listItem,axis=0)))

    

dfEvalXGB = pd.DataFrame(listItem, 

                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 

                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss'])

listIndex = list(dfEvalXGB.index)

listIndex[-1] = 'Average'

dfEvalXGB.index = listIndex

dfEvalXGB
for item,rep in zip(range(1,6),train_errors) :

    print('Report Train ke ',item,':')

    print(rep['report'])
for item,rep in zip(range(1,6),validation_errors) :

    print('Report Test ke ',item,':')

    print(rep['report'])
train_sizes, train_scores, test_scores = learning_curve(estimator=xgb,

                                                       X=data,

                                                       y=target,

                                                       train_sizes=np.linspace(0.3, 0.9, 5),

                                                       cv=10,

                                                       scoring='accuracy')



print('\nTrain Scores : ')

print(train_scores)

# Mean value of accuracy against training data

train_mean = np.mean(train_scores, axis=1)

print('\nTrain Mean : ')

print(train_mean)

print('\nTrain Size : ')

print(train_sizes)

# Standard deviation of training accuracy per number of training samples

train_std = np.std(train_scores, axis=1)

print('\nTrain Std : ')

print(train_std)



# Same as above for test data

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



print('\nTest Scores : ')

print(test_scores)

print('\nTest Mean : ')

print(test_mean)

print('\nTest Std : ')

print(test_std)



# Plot training accuracies 

plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot the variance of training accuracies

plt.fill_between(train_sizes,

                train_mean + train_std,

                train_mean - train_std,

                alpha=0.15, color='red')



# Plot for test data as training data

plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 

        label='Test Accuracy')

plt.fill_between(train_sizes,

                test_mean + test_std,

                test_mean - test_std,

                alpha=0.15, color='blue')



plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
predictProbaTrain=xgb.predict_proba(X_train)
pred = predictProbaTrain[:,1]

fpr, tpr, threshold = roc_curve(y_train, pred)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
predictProbaTest=xgb.predict_proba(X_test)
preds = predictProbaTest[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
coef1 = pd.Series(xgb.feature_importances_,data.columns).sort_values(ascending=False)

coef1.plot(kind='bar', title='Feature Importances')
train_errors = []

validation_errors = []

for train_index, val_index in kf.split(data, target):

    

    # split data

    X_train, X_val = data.iloc[train_index], data.iloc[val_index]

    y_train, y_val = target.iloc[train_index], target.iloc[val_index]



    # instantiate model

    gbc = GradientBoostingClassifier(max_depth=10, n_estimators=150, learning_rate=0.1)



    #calculate errors

    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, gbc)



    # append to appropriate list

    train_errors.append(train_error)

    validation_errors.append(val_error)
listItem = []



for tr,val in zip(train_errors,validation_errors) :

    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],

                     tr['logloss'],val['logloss']])



listItem.append(list(np.mean(listItem,axis=0)))

    

dfEvalGBC = pd.DataFrame(listItem, 

                    columns=['Train Accuracy', 'Test Accuracy', 'Train ROC AUC', 'Test ROC AUC', 

                            'Train F1 Score', 'Test F1 Score', 'Train Log Loss', 'Test Log Loss'])

listIndex = list(dfEvalGBC.index)

listIndex[-1] = 'Average'

dfEvalGBC.index = listIndex

dfEvalGBC
for item,rep in zip(range(1,6),train_errors) :

    print('Report Train ke ',item,':')

    print(rep['report'])
for item,rep in zip(range(1,6),validation_errors) :

    print('Report Test ke ',item,':')

    print(rep['report'])
train_sizes, train_scores, test_scores = learning_curve(estimator=xgb,

                                                       X=data,

                                                       y=target,

                                                       train_sizes=np.linspace(0.3, 0.9, 5),

                                                       cv=10,

                                                       scoring='accuracy')



print('\nTrain Scores : ')

print(train_scores)

# Mean value of accuracy against training data

train_mean = np.mean(train_scores, axis=1)

print('\nTrain Mean : ')

print(train_mean)

print('\nTrain Size : ')

print(train_sizes)

# Standard deviation of training accuracy per number of training samples

train_std = np.std(train_scores, axis=1)

print('\nTrain Std : ')

print(train_std)



# Same as above for test data

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



print('\nTest Scores : ')

print(test_scores)

print('\nTest Mean : ')

print(test_mean)

print('\nTest Std : ')

print(test_std)



# Plot training accuracies 

plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot the variance of training accuracies

plt.fill_between(train_sizes,

                train_mean + train_std,

                train_mean - train_std,

                alpha=0.15, color='red')



# Plot for test data as training data

plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 

        label='Test Accuracy')

plt.fill_between(train_sizes,

                test_mean + test_std,

                test_mean - test_std,

                alpha=0.15, color='blue')



plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
predictProbaTrain=xgb.predict_proba(X_train)
pred = predictProbaTrain[:,1]

fpr, tpr, threshold = roc_curve(y_train, pred)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
predictProbaTest=xgb.predict_proba(X_test)
preds = predictProbaTest[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
coef1 = pd.Series(xgb.feature_importances_,data.columns).sort_values(ascending=False)

coef1.plot(kind='bar', title='Feature Importances')
outside = ['Accuracy', 'Accuracy', 'Accuracy', 'Accuracy', 'Accuracy','Accuracy', 'Accuracy',

          'ROC_AUC', 'ROC_AUC', 'ROC_AUC', 'ROC_AUC', 'ROC_AUC','ROC_AUC', 'ROC_AUC',

          'F1','F1','F1','F1','F1', 'F1','F1',

          'LogLoss','LogLoss','LogLoss','LogLoss','LogLoss','LogLoss','LogLoss']

inside = [1,2,3,4,5,'Avg','Std', 1,2,3,4,5,'Avg','Std', 1,2,3,4,5,'Avg','Std', 1,2,3,4,5,'Avg','Std']

hier_index = list(zip(outside, inside))

hier_index = pd.MultiIndex.from_tuples(hier_index)
acc = []

roc = []

F1 = []

logloss = []



kol = {

    'acc' : 'Test Accuracy',

    'roc' : 'Test ROC AUC',

    'F1' : 'Test F1 Score',

    'logloss' : 'Test Log Loss'

}



for Elr,Edtc, Erfc, Exgb, Egbc in zip(dfEvalLR.iloc[:5].values,dfEvalDTC.iloc[:5].values,dfEvalRFC.iloc[:5].values, dfEvalXGB.iloc[:5].values, dfEvalGBC.iloc[:5].values):

    acc.append([Elr[1],Edtc[1],Erfc[1], Exgb[1], Egbc[1]])

    roc.append([Elr[3],Edtc[3],Erfc[3], Exgb[3], Egbc[3]])

    F1.append([Elr[5],Edtc[5],Erfc[5], Exgb[5], Egbc[5]])

    logloss.append([Elr[7],Edtc[7],Erfc[7], Exgb[7], Egbc[7]])



for i,j in zip([acc,roc,F1,logloss], ['acc','roc','F1','logloss']):

    i.append([dfEvalLR.iloc[:5][kol[j]].mean(),dfEvalDTC.iloc[:5][kol[j]].mean(),dfEvalRFC.iloc[:5][kol[j]].mean(), dfEvalXGB.iloc[:5][kol[j]].mean(), dfEvalGBC.iloc[:5][kol[j]].mean()])

    i.append([dfEvalLR.iloc[:5][kol[j]].std(),dfEvalDTC.iloc[:5][kol[j]].std(),dfEvalRFC.iloc[:5][kol[j]].std(), dfEvalXGB.iloc[:5][kol[j]].std(), dfEvalGBC.iloc[:5][kol[j]].std()])



dfEval = pd.concat([pd.DataFrame(acc),pd.DataFrame(roc),pd.DataFrame(F1),pd.DataFrame(logloss)], axis=0)

dfEval.columns = ['LR','DTC','RFC', 'XGB', 'GBC']

dfEval.index = hier_index

dfEval
data = dfScaled.drop(['nativeCountry'],axis=1)

target= df['income']
xtr,xts,ytr,yts = train_test_split(data,target,test_size=.3,random_state=101)
smot = SMOTE(random_state=101)

X_smot,ytr = smot.fit_sample(xtr,ytr)

X_smot = pd.DataFrame(X_smot, columns=xtr.columns)
xtr,xts,ytrn,yts = train_test_split(data,target,test_size=.3,random_state=101)
model = XGBClassifier(max_depth=10, n_estimators=250, learning_rate=0.1)
model.fit(X_smot,ytr)
predictTrain = model.predict(xtr)

predictTrain
print(classification_report(ytrn,predictTrain))
predictTest = model.predict(xts)

predictTest
print(classification_report(yts,predictTest))
predictProbaTrain=model.predict_proba(xtr)
pred = predictProbaTrain[:,1]

fpr, tpr, threshold = roc_curve(ytrn, pred)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
log_loss(ytrn,pred)
predictProbaTest=model.predict_proba(xts)
preds = predictProbaTest[:,1]

fpr, tpr, threshold = roc_curve(yts, preds)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
log_loss(yts,preds)
coef1 = pd.Series(model.feature_importances_,data.columns).sort_values(ascending=False)

coef1.plot(kind='bar', title='Feature Importances')
# with open('XGB.pickle', 'wb') as f:

#     pickle.dump(model, f)
predictProbaTest=model.predict_proba(xts)
preds = predictProbaTest[:,1]

fpr, tpr, threshold = roc_curve(yts, preds)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characterisitc')

plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(round(roc_auc,2)))

plt.legend(loc = 'lower right')

plt.plot([0,1], [0,1], 'r--')

# plt.xlim([0,1])

# plt.ylim([0,1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
print('FPR:',fpr[-760:-750])

print('TPR:',tpr[-760:-750])

print('THRESHOLD:',threshold[-760:-750])
listProba = []

for x,y,z in zip(tpr,fpr,threshold):

    listProba.append([x,y,z])

dfProba = pd.DataFrame(listProba, columns=['TPR','FPR','Threshold'])

dfProba.head()
dfProba[dfProba['TPR']>0.17].head(20)
predictions = [1 if i > 0.16 else 0 for i in preds]
print(classification_report(yts,predictions))
sns.countplot(predictions)
sns.countplot(yts)
dfProba[dfProba['FPR']<0.014].tail(50)
predictions = [1 if i > 0.78 else 0 for i in preds]
print(classification_report(yts,predictions))
sns.countplot(predictions)
sns.countplot(yts)