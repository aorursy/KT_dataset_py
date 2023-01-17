import numpy as np 

import pandas as pd 

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sns.set_style(style='white') 

sns.set(rc={

    'figure.figsize':(10,6), 

    'axes.facecolor': 'white',

    'axes.grid': True, 'grid.color': '.9',

    'axes.linewidth': 1.0,

    'grid.linestyle': u'-'},font_scale=1.5)

train_data = pd.read_csv("../input/titanic/train.csv")



#train_data = pd.read_csv("../data/train.csv")
train_data.head()
train_data.describe()
train_data.nunique()
print(*train_data.columns)
train_data['Pclass'].describe()
fig_pclass = train_data['Pclass'].value_counts()

fig_pclass.plot.pie().legend(labels=["Class 3", "Class 1", "Class 2"],

                            loc='center right', 

                            bbox_to_anchor=(2.25, 0.5)

                            ).set_title("Класс пассажиров")
pclass_1_surv = round((train_data[train_data['Pclass'] == 1].Survived == 1).value_counts()[1]/

                      len(train_data[train_data['Pclass'] == 1]) * 100, 2)



pclass_2_surv = round((train_data[train_data['Pclass'] == 2].Survived == 1).value_counts()[1]/

                      len(train_data[train_data['Pclass'] == 2]) * 100, 2)



pclass_3_surv = round((train_data[train_data['Pclass'] == 3].Survived == 1).value_counts()[1]/

                      len(train_data[train_data['Pclass'] == 3]) * 100, 2)





pclass_plot_df = pd.DataFrame({"Выжившие (%)":{"Class 1": pclass_1_surv,

                                               "Class 2": pclass_2_surv,

                                               "Class 3": pclass_3_surv

                                              },

                               "Не выжившие (%)":{"Class 1": 100-pclass_1_surv,

                                                  "Class 2": 100-pclass_2_surv, 

                                                  "Class 3": 100-pclass_3_surv

                                                 }

                              })



pclass_plot_df.plot.bar().set_title("Процентное соотношение выживних в разных классах")
fig_sex = (train_data['Sex'].value_counts(normalize = True) * 100).plot.bar()
print(train_data.pivot_table('PassengerId',

                             'Sex',

                             'Survived',

                             'count').plot(kind='bar', stacked=True))
train_data['Age'].value_counts()
train_data['Age'].describe()
train_data['Age_group_ST'] = pd.cut(train_data['Age'], [0, 10, 20, 30, 40, 50, 60, 70, 80])
sns_age = sns.countplot(x = "Age_group_ST", hue = "Survived", data = train_data, palette=["C1", "C0"])

sns_age.legend(labels = ["Умер", "Выжил"])
sns.distplot(train_data['Age'].dropna(),bins=30)
train_data['name_prefx_ST'] = train_data['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

train_data['name_prefx_ST'].value_counts()
td_group_sp = train_data.groupby(['Sex', 'Pclass'])
td_group_sp['Age'].apply(lambda x: x.fillna(x.median()))

train_data['Age'].fillna(train_data['Age'].median, inplace = True)
train_data.SibSp.describe()
train_data['SibSp_group_ST'] = pd.cut(train_data['SibSp'], [0, 1, 2, 3, 4, 5, 6, 7, 8], include_lowest = True)
sns_sibsp = sns.countplot(x = "SibSp_group_ST", 

                          hue = "Survived", 

                          data = train_data, 

                          palette=["C1", "C0"]).legend(labels = ["Умер", "Выжил"])

sns_sibsp.set_title("Соотношение выживших с близкими родств")
train_data['Parch'].describe()
train_data['parents_children_ST'] = pd.cut(train_data['Parch'], [0, 1, 2, 3, 4, 5, 6], include_lowest = True)

sns_parents = sns.countplot(x = "parents_children_ST",

                            hue = "Survived",

                            data = train_data,

                            palette=["C1", "C0"]).legend(labels = ["Умер", "Выжил"])

sns_parents.set_title("Соотношение выживших с родителями/детьми")
train_data['Family_ST'] = train_data['Parch'] + train_data['SibSp']



#train_data['Solo_ST'] = train_data['Family_ST'] == 0

train_data['Solo_ST'] = train_data['Family_ST'].map(lambda x: 0 if x else 1).astype('category')
train_data['Solo_ST'].head()
train_data.Fare.describe()
train_data['Fare_category_ST'] = pd.cut(train_data['Fare'],

                                        bins=[0, 7.90, 14.45, 31.28, 120], 

                                        labels=['Low', 'Low_Mid', 'High_Mid', 'High'])
x = sns.countplot(data = train_data,

                  x = "Fare_category_ST",

                  hue = "Survived",

                  palette=["C1", "C0"]).legend(labels = ["Умер", "Выжил"])

x.set_title("Выжившие в зависимости от тарифа")
train_data['Cabin'] = train_data['Cabin'].fillna('n0n')
train_data['Embarked'].describe()
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)
train_data['Sex'].head()
train_data['Sex'] = train_data['Sex'].map(lambda x: 1 if x == 'male' else 0)

train_data['Sex'] = train_data['Sex'].astype('category')
#train_EMB = pd.get_dummies(train_data['Embarked'], prefix="Emb_ST", drop_first = True)

#train_EMB.head()
train_data.columns
print(train_data.info())
train_data.head()
train_data = pd.concat([train_data,

                pd.get_dummies(train_data['Cabin'], prefix="Cabin"),

                pd.get_dummies(train_data['Age_group_ST'], prefix="Age_group_ST"),

                pd.get_dummies(train_data['name_prefx_ST'], prefix="name_pref_ST", drop_first = True),

                pd.get_dummies(train_data['Fare_category_ST'], prefix="Fare_ST", drop_first = True),

                pd.get_dummies(train_data['Pclass'], prefix="Class", drop_first = True),

                pd.get_dummies(train_data['Embarked'], prefix="Emb_ST", drop_first = True)

               ],axis=1)



#td['Sex'] = LabelEncoder().fit_transform(td['Sex'])

#td['Is_Alone'] = LabelEncoder().fit_transform(td['Is_Alone'])
print(train_data.info())
train_data.head()
train_data.drop(['Cabin', 'Age_group_ST', 'name_prefx_ST',

                 'SibSp_group_ST', 'parents_children_ST',

                 'Fare_category_ST', 'Pclass', 'Embarked', 

                 'Name', 'Ticket', 'SibSp', 

                 'Parch', 'Fare', 'Age'

                ], axis=1, inplace=True)





train_data.shape
train_data.columns
train_data.head()
train_data['Sex'] = train_data['Sex'].astype('uint8')

train_data['Solo_ST'] = train_data['Solo_ST'].astype('uint8')
col = list(train_data.columns)
import re



regex = re.compile(r"\[|\]|<", re.IGNORECASE)



train_data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train_data.columns.values]



# импорт моделей

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB





from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold





from sklearn.preprocessing import scale

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

from sklearn.metrics import mean_squared_error, confusion_matrix



#графики

import pylab as pl

import matplotlib.pyplot as plt

itog_val = {}

kfold = 5

random_state = 777
X = train_data.drop('Survived', axis=1)

y = train_data['Survived']



print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)



print(X_train.shape, y_train.shape)
# Модель RandomForestClassifier

model_rfc = RandomForestClassifier(random_state=random_state,

                                   max_depth=9, 

                                   min_samples_leaf=1,

                                   min_samples_split=4,

                                   n_estimators=180)
# Модель KNeighborsClassifier

model_knc = KNeighborsClassifier(n_neighbors=13)
# Модель LogisticRegression

model_lr = LogisticRegression(penalty='l2',  tol=0.0001, random_state=random_state) 
# Модель  GradientBoostingClassifier

model_gbt = GradientBoostingClassifier(learning_rate=0.1,

                                       max_features=17,

                                       min_samples_leaf=6,

                                       min_samples_split=2,

                                       n_estimators=200,

                                       random_state=random_state)
# Модель  XGBClassifier

model_xgbc = xgb.XGBClassifier(max_depth=10, 

                               min_child_weight=1,

                               n_estimators=400, 

                               n_jobs=-1,

                               verbose=1, 

                               learning_rate=0.15,

                               seed=42, 

                               random_state=random_state)
scores = cross_val_score(model_rfc, X, y, cv = kfold)

itog_val['RandomForestClassifier'] = scores.mean()
scores = cross_val_score(model_knc, X, y, cv = kfold)

itog_val['KNeighborsClassifier'] = scores.mean()
scores = cross_val_score(model_lr, X, y, cv = kfold)

itog_val['LogisticRegression'] = scores.mean()
scores = cross_val_score(model_gbt, X, y, cv = kfold)

itog_val['GradientBoostingClassifier'] = scores.mean()
scores = cross_val_score(model_xgbc, X, y, cv = kfold)

itog_val['XGBClassifier'] = scores.mean()
train_data.from_dict(data = itog_val, orient='index').plot(kind='bar', legend=False)
# прорисовка граффиков roc_auc, по моделям

pl.clf()

plt.figure(figsize=(8,6))





#RandomForestClassifier

probas = model_rfc.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandonForest',roc_auc))



#KNeighborsClassifier

probas = model_knc.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))



#LogisticRegression

probas = model_lr.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))



#GradientBoostingClassifier

probas = model_gbt.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('GradientBoostingClassifier',roc_auc))



# Модель  XGBClassifier 

probas = model_xgbc.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('XGBClassifier',roc_auc))







pl.plot([0, 1], [0, 1], 'k--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.0])



pl.xlabel('False Positive Rate')

pl.ylabel('True Positive Rate')

pl.legend(loc=0, fontsize='small')



pl.show()
def f_err_predict_test_train(model, X_train, X_test, y_train, y_test):

    # ошибки на предсказания меток моделью

    

    err_train = np.mean(y_train != model.predict(X_train))

    err_test  = np.mean(y_test  != model.predict(X_test))

    

    print("ошибки на обучающей: {0:.2f}%".format(err_train*100))

    print("ошибки на тестовой: {0:.2f}%".format(err_test*100))

    
# модель GradientBoostingClassifier

predict = model_gbt.fit(X_train, y_train).predict(X_test)

f_err_predict_test_train(model_gbt, X_train, X_test, y_train, y_test)



print(accuracy_score(y_test, predict)) # 0.8101694915254237
# модель LogisticRegression

predict = model_lr.fit(X_train, y_train).predict(X_test)

f_err_predict_test_train(model_lr, X_train, X_test, y_train, y_test)



print(accuracy_score(y_test, predict))  # 0.8169491525423729
#model RandomForestClassifier

model_rfc = RandomForestClassifier(random_state=random_state,

                                   max_depth=9, 

                                   min_samples_leaf=1,

                                   min_samples_split=4,

                                   n_estimators=180)



predict = model_rfc.fit(X_train, y_train).predict(X_test)

f_err_predict_test_train(model_rfc, X_train, X_test, y_train, y_test)



print(accuracy_score(y_test, predict))   # 0.8203389830508474
test_data = pd.read_csv("../input/titanic/train.csv")



#test_data = pd.read_csv("../data/test.csv")

test_data['Age_group_ST'] = pd.cut(test_data['Age'], [0, 10, 20, 30, 40, 50, 60, 70, 80])

test_data['name_prefx_ST'] = test_data['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())



tstd_group_sp = test_data.groupby(['Sex', 'Pclass'])

tstd_group_sp['Age'].apply(lambda x: x.fillna(x.median()))

test_data['Age'].fillna(test_data['Age'].median, inplace = True)





test_data['SibSp_group_ST'] = pd.cut(test_data['SibSp'], [0, 1, 2, 3, 4, 5, 6, 7, 8], include_lowest = True)



test_data['parents_children_ST'] = pd.cut(test_data['Parch'], [0, 1, 2, 3, 4, 5, 6], include_lowest = True)



test_data['Family_ST'] = test_data['Parch'] + test_data['SibSp']

test_data['Solo_ST'] = test_data['Family_ST'].map(lambda x: 0 if x else 1).astype('category')





test_data['Fare_category_ST'] = pd.cut(test_data['Fare'],

                                        bins=[0, 7.90, 14.45, 31.28, 120], 

                                        labels=['Low', 'Low_Mid', 'High_Mid', 'High'])





test_data['Cabin'] = test_data['Cabin'].fillna('n0n')





test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace = True)





test_data['Sex'] = test_data['Sex'].map(lambda x: 1 if x == 'male' else 0)

test_data['Sex'] = test_data['Sex'].astype('category')





test_data['Sex'] = test_data['Sex'].astype('uint8')

test_data['Solo_ST'] = test_data['Solo_ST'].astype('uint8')
test_data = pd.concat([test_data,

                pd.get_dummies(test_data['Cabin'], prefix="Cabin"),

                pd.get_dummies(test_data['Age_group_ST'], prefix="Age_group_ST"),

                pd.get_dummies(test_data['name_prefx_ST'], prefix="name_pref_ST", drop_first = True),

                pd.get_dummies(test_data['Fare_category_ST'], prefix="Fare_ST", drop_first = True),

                pd.get_dummies(test_data['Pclass'], prefix="Class", drop_first = True),

                pd.get_dummies(test_data['Embarked'], prefix="Emb_ST", drop_first = True)

               ],axis=1)



test_data.drop(['Cabin', 'Age_group_ST', 'name_prefx_ST',

                 'SibSp_group_ST', 'parents_children_ST',

                 'Fare_category_ST', 'Pclass', 'Embarked', 

                 'Name', 'Ticket', 'SibSp', 

                 'Parch', 'Fare', 'Age'], axis=1, inplace=True)


test_data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in test_data.columns.values]

test_data.info()
train_data.shape
common_features = list(set(train_data.columns).intersection(set(test_data.columns)))

y_train=train_data['Survived']

X_train=train_data[common_features]

X_test=test_data[common_features]
%%time

#model RandomForestClassifier



predict = model_rfc.fit(X_train, y_train).predict(X_test)
%%time

# модель GradientBoostingClassifier



predict1 = model_gbt.fit(X_train, y_train).predict(X_test)
from datetime import datetime

import os

date_current = datetime.today().strftime('%d_%m')

'''

if not os.path.exists('../data_out'):

    os.makedirs('../data_out')

'''


result = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':predict1})

result['Survived'] = result['Survived'].astype(int)



filename = f'../input/gender_submission.csv'

#filename = f'../data_out/titanic_predict_model_gbt_{date_current}.csv'

result.to_csv(filename,index=False)





print('Saved file:' + filename)
# 0.79425 with model GradientBoostingClassifier
