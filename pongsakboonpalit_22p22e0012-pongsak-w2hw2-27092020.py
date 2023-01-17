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
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import missingno
%matplotlib inline
%config inlinebackend.figure_format = 'retina' 

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
# View the training data
train.head()
train.Age.plot.hist()
test.head()
gender_submission.head()
train.describe().T
missingno.matrix(train, figsize = (30,10))
train.isnull().sum()
train.info()
# number of survived
fig = plt.figure(figsize=(20,2))
sns.countplot(y='Survived', data=train);
print(train.Survived.value_counts())
fig = plt.figure(figsize=(20,2))
sns.countplot(y='Sex', data=train);
print(train.Sex.value_counts())
train.groupby(by='Sex').agg(['sum','count'])
temp = train[['Survived','Pclass','Sex']]
temp['Sex'] = temp.Sex.map({'male':0,'female':1})
temp
train.pivot_table(index='Sex',columns='Survived',values='Pclass',aggfunc='count',margins=True)
train.pivot_table(index='SibSp',columns='Survived',values='Pclass',aggfunc='count',margins=True)
temp['SibSp'] = train.SibSp
temp
train.pivot_table(index='Parch',columns='Survived',values='Pclass',aggfunc='count',margins=True)
temp['Parch'] = train.Parch
temp
train.Ticket
temp['Fare'] = pd.cut(train['Fare'], bins=5)
temp

temp.pivot_table(index='Fare',columns='Survived',values='Pclass',aggfunc='count',margins=True)
train.Cabin.str[0]
train.Embarked
temp['Embarked'] = train.Embarked
temp
temp.pivot_table(index='Embarked',columns='Survived',values='Pclass',aggfunc='count',margins=True)
temp.columns
enc_cols = ['Embarked', 'Sex', 'Pclass']
selected_cols = ['SibSp', 'Parch', 'Fare', 'Embarked', 'Sex', 'Pclass']
def encode2predict(dataframe=train): 
    enc_cols = ['Embarked', 'Sex', 'Pclass']
    selected_cols = ['SibSp', 'Parch', 'Fare', 'Embarked', 'Sex', 'Pclass']
    return pd.get_dummies(train[selected_cols],columns=enc_cols)
X_train = pd.get_dummies(train[selected_cols],columns=enc_cols)
X_train = encode2predict(dataframe=train)
pre_test = test.fillna(test.median()) # replace median to 1 null Fare column
pre_test = pre_test.merge(gender_submission,on='PassengerId',how='inner')
pre_test = pre_test[['Survived']+selected_cols]
X_test = pd.get_dummies(pre_test.drop('Survived',axis=1),columns=enc_cols)
y_train = train[['Survived']]
y_test = pre_test[['Survived']]
#2.4 ทําการสร้างตัวจําแนกประเภท (Classifier) แบบต่างๆ
algo=[
    [DecisionTreeClassifier(), 'DecisionTreeClassifier'],
    [GaussianNB(), 'GaussianNB'],
    [MLPClassifier(), 'MLPClassifier']
]

model_scores=[]
train_scores=[]
for a in algo:
    model = a[0]
    model.fit(X_train, y_train)
    score=model.score(X_test, y_test)
    recall=metrics.recall_score(y_test,model.predict(X_test))
    precision=metrics.precision_score(y_test,model.predict(X_test))
    f1=metrics.f1_score(y_test,model.predict(X_test))
    model_scores.append([score,recall,precision,f1, a[1]])
    
    train_score=model.score(X_train, y_train)
    train_recall=metrics.recall_score(y_train,model.predict(X_train))
    train_precision=metrics.precision_score(y_train,model.predict(X_train))
    train_f1=metrics.f1_score(y_train,model.predict(X_train))
    train_scores.append([score,recall,precision,f1, a[1]])   
    
    #model_recall.append([recall,a[1]])
    y_train_pred=model.predict(X_train)
    y_pred=model.predict(X_test)
    print(f'train_{a[1]:20} score: {train_score:.04f} recall: {train_recall:.04f} precision: {train_precision:.04f} f1: {train_f1:.04f}')
    print(f'{a[1]:20} score: {score:.04f} recall: {recall:.04f} precision: {precision:.04f} f1: {f1:.04f}')
    print(f'train:\n{metrics.confusion_matrix(y_train, y_train_pred)},\n test:\n{metrics.confusion_matrix(y_test, y_pred)}')
    #print(metrics.classification_report(y_test, y_pred))
    print('-' * 100)

print(f'accuracy recall precision f1:\n{model_scores}')
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
print(f'accuracy: {model.score(X_test,y_test):.5f}  recall: {metrics.recall_score(y_test,model.predict(X_test)):.5f}  precision: {metrics.precision_score(y_test,model.predict(X_test)):.5f}  f1: {metrics.f1_score(y_test,model.predict(X_test)):.5f}')
print(metrics.confusion_matrix(y_test, model.predict(X_test)))
model = GaussianNB()
model.fit(X_train,y_train)
print(f'accuracy: {model.score(X_test,y_test):.5f}  recall: {metrics.recall_score(y_test,model.predict(X_test)):.5f}  precision: {metrics.precision_score(y_test,model.predict(X_test)):.5f}  f1: {metrics.f1_score(y_test,model.predict(X_test)):.5f}')
print(metrics.confusion_matrix(y_test, model.predict(X_test)))
model = MLPClassifier()
model.fit(X_train,y_train)
print(f'accuracy: {model.score(X_test,y_test):.5f}  recall: {metrics.recall_score(y_test,model.predict(X_test)):.5f}  precision: {metrics.precision_score(y_test,model.predict(X_test)):.5f}  f1: {metrics.f1_score(y_test,model.predict(X_test)):.5f}')
print(metrics.confusion_matrix(y_test, model.predict(X_test)))
# 2.7 Cross validation for each classifier function
def cross_valid(estimator=DecisionTreeClassifier,data=train,cv=5):
    X_train_t, X_train_v, y_train_t, y_train_v = train_test_split(X_train, y_train,
                                                 test_size = 0.2, random_state = 2012, shuffle = True)
    model = (estimator())
    model.fit(X_train_t, y_train_t)
    
    scores = ['accuracy','precision','recall','f1']
    for sc in scores:
        score_cv = model_selection.cross_val_score(model, X_train, y_train, cv=cv, scoring=sc)
        print(f'{sc} = {score_cv}')
        print(f'average of {sc} = {score_cv.mean()}\n')

cross_valid(estimator=DecisionTreeClassifier)
cross_valid(estimator=GaussianNB)
cross_valid(estimator=MLPClassifier)
def multialgo_score(X,y,score,cv=5):
    sc=[]
    X_all_data = pd.concat([X_train,X_test])
    y_all_data = pd.concat([y_train.reset_index(drop=True),gender_submission[['Survived']]],axis=0)
    for i in algo:    #2.7.4 แสดง F-measure ของทั้งชุดข้อมูล Train+Test
        print(f'algo: {i[1]}')
        model = i[0]
        model.fit(X_train,y_train)
        
        scores = [score]
        for sc in scores:
            score_cv = model_selection.cross_val_score(model, X, y, cv=cv, scoring=sc)
            print(f'{sc} = {score_cv}')
            print(f'average of {sc} = {score_cv.mean()}\n')
        print('-'*100)
#2.7.1 recall
multialgo_score(X=X_train,y=y_train,score='recall',cv=5)
#2.7.2 precisoin
multialgo_score(X=X_train,y=y_train,score='precision',cv=5)
#2.7.3 F-measure
multialgo_score(X=X_train,y=y_train,score='f1',cv=5)
#2.7.4
for i in algo:    #2.7.4 Average F-Measure ของทั้งชุดข้อมูล
    print(f'algo: {i[1]}')
    model = i[0]
    model.fit(X_train,y_train)
    
    f1_avg = metrics.precision_recall_fscore_support(y_test,model.predict(X_test), average='weighted')[2]
    print(f'average F1 = {f1_avg}')
    print('-'*100)

#ทำข้อมูลเทียบทั้ง train และ test
X_all_data = pd.concat([X_train,X_test])
y_all_data = pd.concat([y_train.reset_index(drop=True),gender_submission[['Survived']]],axis=0)
for i in algo:    #2.7.4 แสดง F-measure ของทั้งชุดข้อมูล Train+Test
    print(f'algo: {i[1]}')
    model = i[0]
    model.fit(X_train,y_train)
    
    scores = ['f1']
    for sc in scores:
        score_cv = model_selection.cross_val_score(model, X_all_data, y_all_data, cv=5, scoring=sc)
        print(f'{sc} = {score_cv}')
        print(f'average of {sc} = {score_cv.mean()}\n')
    print('-'*100)
model = MLPClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_all_data)
pred_submission = model.predict(X_test)
x=pd.concat([train[['PassengerId','Survived']],gender_submission])
compare = pd.concat([x.reset_index(drop=True),pd.Series(pred,name='predict')],axis=1)
compare['compare'] = compare.Survived==compare.predict
compare
metrics.confusion_matrix(y_test,model.predict(X_test))
predictions = pd.Series(pred_submission,name='Survived')

# Create a submisison dataframe and append the relevant columns
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = predictions # our model predictions on the test dataset
submission.head()
# What does our submission have to look like?
gender_submission.head()
# Let's convert our submission dataframe 'Survived' column to ints
submission['Survived'] = submission['Survived'].astype(int)
print('Converted Survived column to integers.')
# How does our submission dataframe look?
submission.head()
# Are our test and submission dataframes the same length?
if len(submission) == len(test):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")
# Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submission.to_csv('/kaggle/working/MLPClassifier_submission.csv', index=False)
print('Submission CSV is ready!')
# Check the submission csv to make sure it's in the right format
submissions_check = pd.read_csv("/kaggle/working/MLPClassifier_submission.csv")
submissions_check.head()


def compareplot(col,dataframe=train):
    survived = dataframe[dataframe.Survived == 1][col].value_counts()
    dead = dataframe[dataframe.Survived == 0][col].value_counts()
    df2 = pd.DataFrame([survived, dead], index=['survived', 'dead'])
    print(df2)
    df2.plot(kind='bar', stacked=True)
compareplot(dataframe=train,col='Sex')
train['Age_range'] = pd.cut(train['Age'], bins=5)
train['Fare_range'] = pd.cut(train['Fare'], bins=5)
compareplot('Fare_range')
