import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import auc,classification_report,confusion_matrix , roc_auc_score ,accuracy_score

from sklearn.metrics import plot_roc_curve

from sklearn.model_selection import StratifiedKFold



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier

import xgboost as xgb

import lightgbm 

from sklearn.tree import DecisionTreeClassifier , ExtraTreeClassifier

from sklearn.model_selection import KFold

from sklearn.svm.classes import SVC

from sklearn.metrics.classification import cohen_kappa_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import RidgeClassifier
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.info()
df = train.copy()
def word_tokenze(x):

    tokenizer = RegexpTokenizer(r'[A-Za-z]+[.]+')

    return str(tokenizer.tokenize(x)[0])
df['Title'] = train.Name.apply(word_tokenze)
df.head()
df['Age'] = df.groupby('Title')['Age'].apply(lambda x: x.fillna(x.mean()))
df.info()
grid = sns.FacetGrid(df , row = 'Embarked')

grid.map(sns.pointplot , 'Pclass' ,'Survived' , 'Sex' , palette = 'deep')

grid.add_legend()
grid = sns.FacetGrid(df , row='Embarked' ,col='Survived' ,size=3 ,aspect=2)

grid.map(plt.hist , 'Age' ,alpha =0.7 , bins = 20)

grid.add_legend()
pd.crosstab(df.Sex,df.Survived)
pd.crosstab(df.Survived , df.Embarked)
pd.crosstab(df.Survived ,df.SibSp)
pd.crosstab(df.Survived ,df.Parch)
pd.crosstab(df.Survived , df.Pclass)
(df.groupby('Title')['Survived'].mean()*100).round(2).astype(str)+'%'
(df.groupby('Sex')['Survived'].mean()*100).round(2).astype(str)+'%'
(df.groupby('Embarked')['Survived'].mean()*100).round(2).astype(str)+'%'
(df.groupby('SibSp')['Survived'].mean()*100).round(2).astype(str)+'%'
df.groupby('Pclass')['Fare'].mean()
(df.groupby('Parch')['Survived'].mean()*100).round(2).astype(str)+'%'
(df.groupby('Pclass')['Survived'].mean()*100).round(2).astype(str)+'%'
class FinalData:

    def __init__(self,train , test):

        self.train = train

        self.test = test

    

    def title_extract(self , series):

        tokenizer = RegexpTokenizer(r'[A-Za-z]+[.]+')

        return str(tokenizer.tokenize(series)[0])

    

    def final_dataset(self):

        combine = pd.concat([self.train , self.test] , ignore_index=True)

        combine['Title'] = combine.Name.apply(self.title_extract)

        combine = combine.drop('Name' ,axis = 1)

        combine['Age'] = combine.groupby('Title')['Age'].apply(lambda x: x.fillna(x.mean()))

        combine['Title'] = combine['Title'].replace(['Don.', 'Rev.', 'Dr.','Dona.',

        'Major.', 'Lady.', 'Sir.', 'Col.', 'Capt.',

       'Countess.', 'Jonkheer.'] , 'rare')

        combine['Title'] = combine['Title'].replace('Mme.' , 'Mrs.')

        combine['Title'] = combine['Title'].replace('Ms.' , 'Miss.')

        combine['Title'] = combine['Title'].replace('Mlle.' , 'Miss.')

        

        combine['Pclass_sex'] = combine['Pclass'].astype(str) + combine['Sex'].astype(str)

        pclass_sex = pd.get_dummies(combine.Pclass_sex)

        combine = pd.concat([combine , pclass_sex] , axis = 1)

        

        

        combine['Sex'] = combine['Sex'].map({'male':1 ,'female':0})

        combine['AgeBand'] = pd.cut(combine['Age'] , 5)

        combine.loc[ combine['Age'] <= 16, 'Age'] = 0

        combine.loc[(combine['Age'] > 16) & (combine['Age'] <= 32), 'Age'] = 1

        combine.loc[(combine['Age'] > 32) & (combine['Age'] <= 48), 'Age'] = 2

        combine.loc[(combine['Age'] > 48) & (combine['Age'] <= 64), 'Age'] = 3

        combine.loc[ combine['Age'] > 64, 'Age'] =4

        

        combine.loc[ combine['Fare'] <= 102, 'Fare'] = 0

        combine.loc[(combine['Fare'] > 102) & (combine['Fare'] <= 204), 'Fare'] = 1

        combine.loc[(combine['Fare'] > 204) & (combine['Fare'] <= 307), 'Fare'] = 2

        combine.loc[(combine['Fare'] > 307) & (combine['Fare'] <= 410), 'Fare'] = 3

        combine.loc[ combine['Fare'] > 410, 'Age'] =4

        

        combine['Embarked'] = pd.factorize(combine.Embarked)[0]

        combine.Title = pd.factorize(combine.Title)[0]

        combine = combine.drop(['PassengerId' , 'Ticket' , 'Cabin','AgeBand' , 'Pclass_sex'] , axis = 1)

        combine['Members'] = combine['SibSp'] + combine['Parch'] + 1

        

        combine['IsAlone'] = 1

        combine.loc[(combine['Members'] > 0), 'IsAlone'] = 0

        

        combine = combine.drop(['SibSp' , 'Parch'] , axis = 1)

        pclass_enc = pd.get_dummies(combine.Pclass)

        combine = pd.concat([combine , pclass_enc] , axis = 1)

        train_df = combine[combine.index<self.train.shape[0]]

        test_df = combine[combine.index>=self.train.shape[0]]

        

        return (train_df ,test_df)
train1, test1 =  FinalData(train , test).final_dataset()
test1.Fare.fillna(13.675550 , inplace = True)
train1 =train1.drop('Pclass' , axis =1 )
y = train1.Survived
X = train1.drop('Survived' , axis= 1)
X_train , X_test , y_train , y_test = train_test_split(X , y , random_state = 42 , test_size = 0.30)
X_test.shape
train1.head()
test1 = test1.drop(['Survived' , 'Pclass'] , axis = 1)
stacked_train = np.zeros((y_train.shape[0] ,8))
stacked_test = np.zeros((y_test.shape[0] , 8))
lr = LogisticRegression()

sgd = SGDClassifier()

perceptron = Perceptron()

gbc = GradientBoostingClassifier()

rfc = RandomForestClassifier()

enet = RidgeClassifier()

dtc = DecisionTreeClassifier()

extc = ExtraTreeClassifier()
from sklearn.preprocessing import MinMaxScaler

scaler1 = MinMaxScaler()

X_scaled = scaler1.fit_transform(X)
y_lr = cross_val_predict(lr , X_scaled , y , cv = 5)

y_sgd = cross_val_predict(sgd , X_scaled , y , cv = 5)

y_perceptron = cross_val_predict(perceptron , X_scaled , y , cv = 5)

y_gbc = cross_val_predict(gbc , X_scaled , y , cv = 5)

y_rfc = cross_val_predict(rfc , X_scaled , y, cv = 5)

y_enet = cross_val_predict(enet , X_scaled , y , cv = 5)

y_dtc = cross_val_predict(dtc , X_scaled , y, cv = 5)

y_extc = cross_val_predict(extc ,X_scaled , y , cv = 5)
stacked_train = np.c_[y_lr , y_sgd , y_perceptron ,y_gbc , y_rfc ,y_enet , y_dtc , y_extc]
lr.fit(X_scaled ,y)

sgd.fit(X_scaled ,y)

perceptron.fit(X_scaled ,y)

gbc.fit(X_scaled ,y)

rfc.fit(X_scaled ,y)

enet.fit(X_scaled ,y)

dtc.fit(X_scaled ,y)

extc.fit(X_scaled ,y)
y_p1 = lr.predict(test1)

y_p2 = sgd.predict(test1)

y_p3 = perceptron.predict(test1)

y_p4 = gbc.predict(test1)

y_p5 = rfc.predict(test1)

y_p6 = enet.predict(test1)

y_p7 = dtc.predict(test1)

y_p8 = extc.predict(test1)
stacked_test = np.c_[y_p1 , y_p2, y_p3, y_p4,y_p5,y_p6,y_p7,y_p8]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train11 = scaler.fit_transform(stacked_train)

test11 =  scaler.fit_transform(stacked_test)
lr1 =DecisionTreeClassifier()

lr1.fit(train11 , y)
lr1.score(train11 , y)
model = LogisticRegression()

model.fit(X_train , y_train)

model.score(X_train , y_train)

model.score(X_test , y_test)



tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)

cv = StratifiedKFold(n_splits=5)



fig, ax = plt.subplots(figsize=(10,10))

for i, (tr, val) in enumerate(cv.split(X_train, y_train)):

    model.fit(X_train.iloc[tr], y_train.iloc[tr])

    viz = plot_roc_curve(model, X_train.iloc[val], y_train.iloc[val],

                         name='ROC fold {}'.format(i),

                         alpha=0.4, lw=2, ax=ax)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

    aucs.append(viz.roc_auc)

    

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

        label='Chance', alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color='b',

        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

        lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                label=r'$\pm$ 1 std. dev.')



ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],

       title="Receiver operating characteristic curve")

ax.legend(loc="lower right")

plt.show()
y_predict = model.predict(X_test)

print(accuracy_score(y_test , y_predict))

print(roc_auc_score(y_test , y_predict))

print(classification_report(y_test , y_predict))
predt = lr1.predict(test11)
pid = test.PassengerId

submit = pd.DataFrame()

submit['PassengerId'] = pid

submit['Survived'] = predt.astype(int)

submit.to_csv('submission.csv' , index=False)