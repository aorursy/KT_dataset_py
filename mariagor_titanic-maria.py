# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


df = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df_test.head()
df_test['Age']

df_test.loc[df_test['Age'].isnull()]
df_test.at[df_test.PassengerId == 980, 'Age'] = 28

df_test.at[df_test.PassengerId == 1044, 'Fare'] = 8.0500

df_test.loc[df_test['Fare'].isnull()]
df['family'] = df.Parch + df.SibSp

df_test['family'] = df_test.Parch + df_test.SibSp



def make_title(name):

    arr_name = name.split()

    for p in arr_name : 

        if '.' in p:

            return p

    return ''

    

df_test['title'] = df_test.Name.apply(make_title)

df['title'] = df.Name.apply(make_title)
a, b = np.unique(df.title, return_counts=True)

plt.bar(a, b);
plt.hist(df.Survived); 
plt.hist(df.Pclass);
print (list(df))

print (list(df_test))
df_title_groped = df.groupby(['title'],as_index =False).mean()

ave_age_title = df_title_groped[['title','Age']]



df_title_groped_test = df_test.groupby(['title'],as_index =False).mean()

ave_age_title_test = df_title_groped_test[['title','Age']]





print (ave_age_title)

print (ave_age_title_test)


df = df.merge(ave_age_title, how='inner', on = ['title'])

df_test = df_test.merge(ave_age_title_test, how='inner', on = ['title'])

import math
def age_func(data):

    if math.isnan(data['Age_x']):

        return data['Age_y']

    else:

        return data['Age_x']



df["Age"] = df.apply(age_func, axis=1)

df_test["Age"] = df_test.apply(age_func, axis=1)

df.head()

df_test.head()
df_test.head()
def corr_color(daf):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(daf.corr());

    fig.colorbar(cax)

    ax.set_xticklabels(list(daf.columns.values));

    ax.set_yticklabels(list(daf.columns.values));

    
corr_color(df)
bin_gender = lambda x: 0 if x == 'male' else 1



df_gender = df.copy()

df_gender_test = df_test.copy()



df_gender.Sex = df.Sex.apply(bin_gender)

df_gender_test.Sex = df_test.Sex.apply(bin_gender)

print (df_gender[['Sex', 'Survived']].corr())



female_survived = df_gender['Survived'].loc[df_gender['Sex']==1]

male_survived = df_gender['Survived'].loc[df_gender['Sex']==0]



bars = ['Female', 'Male']

y_pos = np.arange(len(bars))

plt.barh(y_pos,[female_survived.sum(), male_survived.sum()]);

plt.yticks(y_pos, bars);
corr_color(df_gender[['Survived','Sex','Age_x']])

corr_color(df)
df = df_gender.drop('Age_x', axis=1)

df = df.drop('Age_y',axis=1)

df = df.drop('Cabin',axis=1)



df_test = df_gender_test.drop('Age_x', axis=1)

df_test = df_test.drop('Age_y',axis=1)

df_test = df_test.drop('Cabin',axis=1)



#print(df.head())

df_test.head()
df[['Age', 'Survived']].corr()
#df[df.PassengerId == 980].Age = '28'

#df_test.at[df_test.PassengerId == 980]

#df_test.at[df.PassengerId == 980, 'Age'] = '28'

#df_test.iloc[[411]]['Age']= 28

#df_test[df_test.Age.isnull()]

#df.loc[df.Embarked == np.nan, 'Embarked']

#df.at[df.Embarked == np.nan, 'Embarked'] = 'M'

#df

#plt.hist(df.Embarked)
df.Ticket.describe()
df['ticket_pre'] = df.Ticket.apply(lambda x:x.split()[0].replace('.','') if len(x.split())>1 else 'none')

df['ticket_suf'] = df.Ticket.apply(lambda x:x.split()[1].replace('.','') if len(x.split())>1 else x)



df_test['ticket_pre'] = df_test.Ticket.apply(lambda x:x.split()[0].replace('.','') if len(x.split())>1 else 'none')

df_test['ticket_suf'] = df_test.Ticket.apply(lambda x:x.split()[1].replace('.','') if len(x.split())>1 else x)
a, b = np.unique(df.ticket_pre, return_counts=True)

plt.bar(a, b);
a, b = np.unique(df.ticket_suf, return_counts=True)

plt.bar(a, b);
df.ticket_pre.describe()
df_test.Fare.describe()
df_test[df_test.isnull().any(axis=1)]
df_test.Pclass.describe()
df_test.loc[df_test.Embarked.isnull(),['ticket_suf', 'ticket_pre', 'Ticket', 'Embarked', 'Pclass']]
df_test.head()

#df_test.loc[df.Embarked.isnull(),['ticket_suf', 'ticket_pre', 'Ticket', 'Embarked', 'Pclass']]
df_grouped = df.loc[df.Embarked.notnull(),['Embarked', 'Pclass', 'title', 'Fare']].groupby('title')

df_grouped.head()
df.at[df.Embarked.isnull(),'Embarked']='M'

df_test.at[df_test.Embarked.isnull(),'Embarked']='M'
df_test.info()
df_test.loc[df_test['Sex'].isnull(),['ticket_suf', 'ticket_pre', 'Ticket', 'Embarked', 'Pclass', 'Name', 'title', 'Age','family', 'PassengerId', 'Sex']]
df['child'] = df.Age.apply(lambda x: 1 if x <= 16 else 0)

df_test['child'] = df_test.Age.apply(lambda x: 1 if x <= 16 else 0)
df.head()
df_test.head(50)
a,b = np.unique(df.Survived, return_counts = True)

plt.pie (b);
a,b = np.unique(df.child, return_counts = True)

plt.pie (b, labels = a);
total_ppl = len(df.Sex)

tot_woman = df.Sex.sum()

tot_man = total_ppl - tot_woman

df_sex_survived = df.Sex[df.Survived == 1]

df_sex_not_survived = df.Sex[df.Survived == 0]

s_woman = df_sex_survived.sum()

n_woman = df_sex_not_survived.sum()

s_man = len(df_sex_survived) - s_woman

n_man = len(df_sex_not_survived) - n_woman
hights = np.array([s_woman, s_man, n_woman, n_man])

loc = np.arange(len(hights))

plt.bar(loc ,hights, tick_label = ['s w', 's m', 'n w', 'n m'])
hights = np.array([s_woman/tot_woman, s_man/tot_man, n_woman/tot_woman, n_man/tot_man])

loc = np.arange(len(hights))

plt.bar(loc ,hights, tick_label = ['s w', 's m', 'n w', 'n m']);
import seaborn as sns
sns.catplot('Survived', data = df, kind='count', hue='title');
sns.catplot('child', data = df, kind='count', hue='Survived');
df.head()
sns.catplot('title', data = df, kind='count', hue='Survived');
sns.catplot('family', data = df, kind='count', hue='Survived');
sns.catplot('ticket_pre', data = df, kind='count', hue='Survived');
sns.catplot('Embarked', data = df, kind='count', hue='Survived');
sns.catplot('Pclass', data = df, kind='count', hue='Survived');
sns.catplot('Sex', data = df, kind='count', hue='Survived');
df['bin_age'] = df.Age.apply(lambda x: int(x/5)*5)

df_test['bin_age'] = df_test.Age.apply(lambda x: int(x/5)*5)

sns.catplot('bin_age', data = df, kind='count', hue='Survived');
sns.catplot('Survived', data = df, kind='count', hue='bin_age');
list(df_test)

df_test.head()
feature_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','family','title','Age','ticket_pre','ticket_suf','child','bin_age', 'PassengerId']
X = df[feature_names]

y = df['Survived']

x_test = df_test[feature_names]



X['label'] = 'train'

x_test['label'] = 'test'



concat_x = pd.concat([X , x_test])
X.isnull().values.any()
x_test.isnull().values.any()

#x_test
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
concat_x.describe()
#X = pd.get_dummies(X,drop_first=True)

#x_test = pd.get_dummies(x_test,drop_first=True)

concat_x = pd.get_dummies(concat_x, drop_first=True)



X = concat_x[concat_x['label_train'] == 1]

x_test = concat_x[concat_x['label_train'] == 0]



X = X.drop('label_train', axis=1)

x_test = x_test.drop('label_train', axis=1)



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1,test_size=0.2)
train_X.head()
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import balanced_accuracy_score
tree_model = DecisionTreeClassifier(random_state=1)

tree_model.fit(train_X, train_y)

val_predictions = tree_model.predict(val_X)

val_err = balanced_accuracy_score(val_y, val_predictions)

print(val_err)
tree_model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=1)

tree_model.fit(train_X, train_y)

preds_val = tree_model.predict(val_X)

val_err = balanced_accuracy_score(val_y, preds_val)

print(val_err)
import numpy

tree_model_2 = DecisionTreeClassifier(max_leaf_nodes=10, random_state=1)

tree_model_2.fit(X, y)

real_preds = tree_model_2.predict(x_test)

#df_pred = pd.concat([x_test['PassengerId'], real_preds])

#df_pred = [x_test['PassengerId'],real_preds]

numpy.savetxt("f_2.csv", np.c_[x_test['PassengerId'],real_preds], delimiter=",")

#numpy.savetxt("f_2.csv", df_pred, delimiter=",")
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score
rf_model = RandomForestClassifier(max_leaf_nodes=15,random_state=1)

rf_model.fit(train_X, train_y)

preds = rf_model.predict(val_X)

rf_val_err = balanced_accuracy_score(val_y, preds)

print(rf_val_err)

x_test.describe()
import numpy

real_model = RandomForestClassifier(max_leaf_nodes=15,random_state=1)

real_model.fit(X, y)

real_preds = real_model.predict(x_test)

df_pred = [x_test['PassengerId'],real_preds]

numpy.savetxt("f.csv", np.c_[x_test['PassengerId'],real_preds], delimiter=",")



#numpy.savetxt("f_3.csv", df_pred, delimiter=",")
feature_names_no = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']

df_no_changes = pd.read_csv("../input/train.csv")

df_no_changes = df_no_changes.dropna()

X_no = df_no_changes[feature_names_no]

y_no = df_no_changes['Survived']

X_no = pd.get_dummies(X_no,drop_first=True)



train_X_no, val_X_no, train_y_no, val_y_no = train_test_split(X_no, y_no, random_state = 1,test_size=0.2)

rf_model.fit(train_X_no, train_y_no)

preds_no = rf_model.predict(val_X_no)

rf_val_err_no = balanced_accuracy_score(val_y_no, preds_no)

print(rf_val_err_no)
from xgboost import XGBClassifier
my_model = XGBClassifier(n_estimators=10, learning_rate=0.01,max_leaf_nodes=200,random_state=1)

my_model.fit(train_X, train_y, early_stopping_rounds=10, 

             eval_set=[(val_X, val_y)],verbose=False)



predictions = my_model.predict(val_X)

print(balanced_accuracy_score(predictions, val_y))
