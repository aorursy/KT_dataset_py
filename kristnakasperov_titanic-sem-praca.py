#import knižníc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # nastavenie predvoleného seaborn -> štatistická vizualizácia dát

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.shape #veľkosť tréningovej množiny
#df_test.shape #veľkosť testovacej množiny. Približne 2/3 celkových dát sa nachádzajú v tréningovej množine

#pre výpis jedno zakomnentovať
df_train.head()
#df_test.head()

#pre výpis jedno zakomnentovať
df_train.shape #veľkosť tréningovej množiny
df_test.shape #veľkosť testovacej množiny. Približne 2/3 celkových dát sa nachádzajú v tréningovej množine
#df_train.info() #informácie o množine df_train
      #hodnota veku a kabíny chýba v niektorých riadkoch -> vek je zadnaný v 714 záznamoch a kabína v 204 záznamoch
df_test.info() #informácie o množine df_test
      #hodnota veku a kabíny chýba v niektorých riadkoch -> vek je zadnaný v 332 záznamoch a kabína v 91 záznamoch

#pre výpis jedno zakomnentovať
df_train.isnull().sum() #počet chýbajúcich hodnôt veku a kabíny v množine df_train

df_test.isnull().sum() #počet chýbajúcich hodnôt veku a kabíny v množine df_test

# pre výpis jedno zakomnentovať
# roztriedenie ľudí, ktorý prežili/neprežili a následný výpis (množina df_train)
survived = df_train[df_train['Survived'] == 1]
not_survived = df_train[df_train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(df_train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(df_train)*100.0))
print ("Total: %i"%len(df_train))
#v tréningovej množine rozdelenie na 1.,2.,3. triedu
df_train.Pclass.value_counts() 
#zoradenie podľa tried a atribútu survived
df_train.groupby('Pclass').Survived.value_counts() 
#vyjadrenie, koľko prežilo v percentách
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() 
#graf predchádzajúceho príkazu
sns.barplot(x='Pclass', y='Survived', data=df_train) 
#výpis počtu žien a mužov z tréningovej množiny
df_train.Sex.value_counts()
#rozdelenie, žien a mužov prežilo
df_train.groupby('Sex').Survived.value_counts() 
#percentuálne vyjadrenie 
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean() 
#graf percentuálneho vyjadrenia
sns.barplot(x='Sex', y='Survived', data=df_train) 
#počet žien a mužov v jednotlivých triedach
css = pd.crosstab(df_train['Pclass'], df_train['Sex'])
print (css)

css.div(css.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')
#závislosť prežitia od pohlavia a triedy (v percentách) 
sns.factorplot('Sex', 'Survived', hue='Pclass', data=df_train)
#z grafu môžeme vidieť, že ženy z prvej a druhej triedy mali skoro 100% možnosť prežitia, naopak z 3 triedy len okolo 50%
#muži z prvej triedy mali možnosť prežitia okolo 35 % a z 2. a 3. triedy 10-15%
#závislosti prežitia od miesta nalodenia, pohlavia a cestovnej triedy
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=df_train)
#závislosť prežitia od veku
g = plt.figure(figsize=(15,5))
ax1 = g.add_subplot(141)
ax2 = g.add_subplot(142)
ax3 = g.add_subplot(143)
ax4 = g.add_subplot(144)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=df_train, split=True, ax=ax1)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=df_train, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=df_train, split=True, ax=ax3)
sns.violinplot(x="SibSp", y="Age", hue="Survived", data=df_train, split=True, ax=ax4)
#vykreslenie grafov
celkovo_prezili = df_train[df_train['Survived']==1]
celkovo_neprezili = df_train[df_train['Survived']==0]
muzi_prezi = df_train[(df_train['Survived']==1) & (df_train['Sex']=="male")]
zeny_prezi = df_train[(df_train['Survived']==1) & (df_train['Sex']=="female")]
muzi_neprezi = df_train[(df_train['Survived']==0) & (df_train['Sex']=="male")]
zeny_neprezi = df_train[(df_train['Survived']==0) & (df_train['Sex']=="female")]

plt.figure(figsize=[15,5]) #sivá farba - miesta kde sa prekrývajú hodnoty
plt.subplot(111)
sns.distplot(celkovo_prezili['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(celkovo_neprezili['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='yellow', axlabel='Age')

plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(zeny_prezi['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(zeny_neprezi['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='yellow', axlabel='Female Age')

plt.subplot(122)
sns.distplot(muzi_prezi['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(muzi_neprezi['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='yellow', axlabel='Male Age')
plt.figure(figsize=(15,6))
sns.heatmap(df_train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)
#kladné čísla - pozitívna korelácia, zvýšenie jednej funkcie, zvýši druhú funkciu a naopak
#záporné čisla - negatívna korelácia, zvýšenei jednej funkcie zníži druhú funkciu a naopak
train_test_data = [df_train, df_test] # kombinácia testovacej a trénovacej množiny
#extrahovanie titulu z mena a vytverenie nového stĺpca title
for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
df_train.head()
#Výpis vyskytujúcich sa titulov v trénovacej množine
pd.crosstab(df_train['Title'], df_train['Sex'])
#nahradenie menej vyskytujúcich sa titulov pomocou názvu "others"
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#náhrada jednotlivých titulov celým číslom
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)

df_train.head()
#náhradu číslom zopakujeme aj pre pohlavie
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

df_train.head()
#zisťujeme, ktorý hodnota nalodenia sa sa vyskytuje najviac 
df_train.Embarked.value_counts()

#nahradenie neexistujúcich hodnôt v stĺpci Embarked hodnotou S pretože je najviac sa vyskytujúca
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

df_train.head()

#náhradu číslom zopakujeme aj pre miesto nalodenia
for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_train.head()


for dataset in train_test_data:
    priemer_vek = dataset['Age'].mean()
    std_vek = dataset['Age'].std()
    vek_null_count = dataset['Age'].isnull().sum()
    
    vek_null = np.random.randint(priemer_vek - std_vek, priemer_vek + std_vek, size=vek_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = vek_null
    dataset['Age'] = dataset['Age'].astype(int)
    
df_train['Vek_kateg'] = pd.cut(df_train['Age'], 10)

print (df_train[['Vek_kateg', 'Survived']].groupby(['Vek_kateg'], as_index=False).mean())
df_train.head()
#náhradu číslom zopakujeme aj pre jednotlivé kategórie veku
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 8, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 8) & (dataset['Age'] <= 16), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 24), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 24) & (dataset['Age'] <= 32), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 40), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 48), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 56), 'Age'] = 6
    dataset.loc[(dataset['Age'] > 56) & (dataset['Age'] <= 64), 'Age'] = 7
    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 72), 'Age'] = 8
    dataset.loc[ dataset['Age'] > 72, 'Age'] = 9
    
df_train.head()
#nahradíme aj prázdne hodnoty cestovného za hodnoty medianu cestovného
for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(df_train['Fare'].median())

#cest_kateg - roztriedenie cestovného do 4 kategórií
df_train['cest_kateg'] = pd.qcut(df_train['Fare'], 4)
print (df_train[['cest_kateg', 'Survived']].groupby(['cest_kateg'], as_index=False).mean())
df_train.head()
#náhradu číslom zopakujeme aj pre jednotlivé hodnoty cestovných lístkov
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
df_train.head()
#poslednou úpravou je zistenie možnosti prežitia podľa počtu súrodencov
#preto si vytvoríme stĺpec rodina
for dataset in train_test_data:
    dataset['rodina'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (df_train[['rodina', 'Survived']].groupby(['rodina'], as_index=False).mean())
for dataset in train_test_data:
    dataset['sam'] = 0
    dataset.loc[dataset['rodina'] == 1, 'sam'] = 1
    
print (df_train[['sam', 'Survived']].groupby(['sam'], as_index=False).mean())
#odstránime nepotrebné stĺpce, v ktorých dáta nebolo možné nahradiť celočíselnými hodnotami
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'rodina']
df_train = df_train.drop(features_drop, axis=1)
df_test = df_test.drop(features_drop, axis=1)
df_train = df_train.drop(['PassengerId', 'Vek_kateg', 'cest_kateg'], axis=1)
df_train.head()
df_test.head()
#definujeme trénovaciu a testovaciu množinu
X_train = df_train.drop('Survived', axis=1)
y_train = df_train['Survived']
X_test = df_test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#logistická regresia
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')
#k najbližších susedov
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (str(acc_knn) + ' percent')
#rozhodovací strom
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (str(acc_decision_tree) + ' percent')
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": y_pred_decision_tree
    })

submission.to_csv('submission.csv', index=False)