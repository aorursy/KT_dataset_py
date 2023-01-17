import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
data_train = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

data_test = pd.read_csv('../input/titanic/test.csv')
data_train.head()
data_train.info()

data_test.info()
age_nosurv = data_train[(data_train["Age"] > 0) & (data_train["Survived"] == 0)]

age_surv = data_train[(data_train["Age"] > 0) & (data_train["Survived"] == 1)]



plt.figure(figsize=(10,5))



sns.distplot(age_surv["Age"], bins=16, kde=False, color='g')

sns.distplot(age_nosurv["Age"], bins=16, kde=False, color='r')

plt.xlabel("Age",fontsize=15)

plt.ylabel("Survived count",fontsize=15)

plt.show()
# Categorizing age and add age class column with numerical label option

def add_ageclass(df_name,label=False):

    index = df_name.index

    

    df_name['age_class'] = np.nan



    ch_data = []

    for i in index:

        cur_data = df_name['Age'][i]

        if cur_data <=5:

            ch_data.append('toddler')

        elif cur_data>5 and cur_data <=16:

            ch_data.append('child')

        elif cur_data >16 and cur_data<= 32:

            ch_data.append('young adults')

        elif cur_data >32 and cur_data<= 55:

            ch_data.append('adult')  

        elif cur_data> 55:

            ch_data.append('elder')

        else:

            ch_data.append(np.nan)

            

    df_name['age_class'] = ch_data



    if(label==True):

        df_name['age_class'] = df_name['age_class'].map({'toddler': 0, 

                                           'child': 1, 

                                           'young adults': 2,

                                           'adult': 3, 

                                           'elder':4})



add_ageclass(data_train)
plt.figure(figsize=(12,5))



sns.catplot(x='age_class', y='Survived', data=data_train, kind="point");

sns.countplot(x='age_class', data=data_train, hue="Survived")

plt.xlabel("Age Class", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(rotation=45)

plt.show()
# Extract prefix from name column and Add name prefix column with numerical label option



def add_prefix(df_name, label=False):

    title = pd.DataFrame(df_name['Name'].apply(lambda st: st[st.find(", ")+1:st.find(".")]))

    df_name['prefix'] = title.values



    df_name['prefix'] = df_name['prefix'].map(

            {

            ' Mme':'Mrs',

            ' Ms':'Mrs',

            ' Mrs':'Mrs',

            ' Mlle':'Miss',

            ' Miss':'Miss',

            ' Mr':'Mr',

            ' Master':'Master',

            ' Jonkheer':'Honorable',

            ' Don':'Honorable',

            ' Dona':'Honorable',

            ' Sir':'Honorable',

            ' the Countess':'Honorable',

            ' Lady':'Honorable',

            ' Rev':'Other',

            ' Capt':'Other',

            ' Col':'Other',

            ' Major':'Other',

            ' Dr':'Other'

            })

            

    if(label==True):

        df_name['prefix'] = df_name['prefix'].map(

            {'Mr': 0,

             'Mrs': 1, 

             'Miss': 2,

             'Master': 3, 

             'Honorable':4,

             'Other':5})



add_prefix(data_train)
plt.figure(figsize=(10,5))



sns.countplot(x='prefix', data=data_train, hue="Survived")

plt.xlabel("Prefix", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(rotation=45)

plt.show()
sns.catplot(x="prefix", y='Survived', data=data_train, kind="point");
plt.figure(figsize=(10,5))

sns.violinplot(x="Pclass", y="Age", hue='Sex',split=True, data=data_train)
data_train.insert(len(data_train.columns), 'has_relative',int(0))
# Add relatives availability information



def add_relative(df_name):

    index = df_name.index

    

    df_name['has_relative']=0



    for i in index:

        if (df_name['SibSp'][i]>0 or df_name['Parch'][i]>0):

            df_name['has_relative'][i] = 1

        else:

            df_name['has_relative'][i] = 0



add_relative(data_train)
data_train
sns.catplot(x="Pclass", y='Survived', hue='has_relative', col='Sex', data=data_train, kind="point");
sns.catplot(x='age_class', y='Survived', hue='has_relative', data=data_train, order=['toddler','child','young adults', 'adult','elder'], kind="point");
add_ageclass(data_train, label=True)

add_prefix(data_train, label=True)
data_train
imp_age_trd = data_train[['Parch','prefix','Fare','Age']]

imp_age_trd.dropna(subset=['Age'],how='any',inplace=True)



x = imp_age_trd.iloc[:, :-1].values

y = imp_age_trd.iloc[:, -1].values



X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
regressor = RandomForestRegressor(n_estimators = 100, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print('Mean Absolute Error =', metrics.mean_absolute_error(y_test, y_pred))
imp_age_trd_a = data_train[['Parch','prefix', 'Fare','Age']]

imp_age_trd_a = imp_age_trd_a[imp_age_trd_a['Age'].isnull()]

x_trd_a = imp_age_trd_a.iloc[:,:-1].values
y_trd_a = regressor.predict(x_trd_a)



imp_age_trd_a_index = imp_age_trd_a.index



for i in range(len(imp_age_trd_a_index)):

    data_train['Age'][imp_age_trd_a_index[i]] = y_trd_a[i]

    

add_ageclass(data_train, label=True)

data_train.info()
add_ageclass(data_train, label=True)

add_prefix(data_train, label=True)



data_train['Sex'] = data_train['Sex'].map({'female': 0, 'male': 1})

data_train['Embarked'] = data_train['Embarked'].map({'S': 0, 'C': 1,'Q': 2})
data_train
data_train_mod = data_train[['Pclass','Sex', 'age_class', 'prefix', 'has_relative','Fare', 'Survived']]
x = data_train_mod.iloc[:,:-1].values

y = data_train_mod.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
model = Sequential()

model.add(Dense(24, input_dim=6,kernel_initializer='uniform', activation='relu'))

model.add(Dense(24, kernel_initializer='uniform', activation='relu'))

model.add(Dense(24, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=64)
y_pred = model.predict(X_test)

y_pred = (y_pred>0.5)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

(tn+tp)/len(y_pred)
add_relative(data_test)

add_prefix(data_test, label=True)



data_test['Fare'] = data_test['Fare'].replace(0.0, np.nan)

data_test["Fare"].fillna(data_test["Fare"].median(), inplace = True)



data_test['Sex'] = data_test['Sex'].map({'female': 0, 'male': 1})

data_test['Embarked'] = data_test['Embarked'].map({'S': 0, 'C': 1,'Q': 2})
imp_age_tsd_a = data_test[['Parch','prefix', 'Fare','Age']]

imp_age_tsd_a = imp_age_tsd_a[imp_age_tsd_a['Age'].isnull()]



x_tsd_a = imp_age_tsd_a.iloc[:,:-1].values



y_tsd_a = regressor.predict(x_tsd_a)



imp_age_tsd_a_index = imp_age_tsd_a.index



for i in range(len(imp_age_tsd_a_index)):

    data_test['Age'][imp_age_tsd_a_index[i]] = y_tsd_a[i]
add_ageclass(data_test, label=True)

data_test.info()
data_test_pred = data_test[['PassengerId','Pclass','Sex', 'age_class', 'prefix', 'has_relative','Fare']]
x_data_test = data_test_pred.iloc[:,1:].values

sc_test = StandardScaler()

x_data_test = sc_test.fit_transform(x_data_test)
y_data_test = model.predict(x_data_test)

y_data_test = (y_data_test >0.5).astype(int)

data_test_pred['Survived'] = y_data_test
data_test_pred
data_test_pred[['PassengerId', 'Survived']].to_csv (r'prediction_nn.csv', index = False, header=True)