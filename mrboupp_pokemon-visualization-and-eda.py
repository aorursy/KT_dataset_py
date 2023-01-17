# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

import seaborn as sns
pokemon = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

pokemon.head()
pokemon.info()
pokemon.fillna("NoType",inplace=True)
pokemon.info()
pokemon.describe()
pokemon.describe(include='O')
pokemon['Type 1'].unique()
pokemon['Type 2'].unique()
sns.countplot(pokemon['Type 1'],palette = 'Purples')
fig,ax = plt.subplots(1,figsize=(10,10))

sns.scatterplot(data=pokemon, x='Attack', y='Defense',hue='Type 2',alpha=0.5)

plt.show()
sns.jointplot(x="HP", y="Attack", data=pokemon);
sns.boxplot(y="HP", data=pokemon,x='Legendary',palette='Set3')
sns.swarmplot(y="HP", data=pokemon,x='Legendary',palette='Set3')
columns = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']

fig,(ax1) = plt.subplots(1,figsize=(15,10))

i=1



for column in columns:

    sns.lineplot(x=pokemon['Generation'],y=pokemon[column],ax = ax1)

    i=i+1

    

plt.show()
sns.heatmap(pokemon.corr())
pokemon.groupby(['Type 1','Type 2']).size().unstack().fillna(0).style.background_gradient(axis=1)
fig = sns.FacetGrid(pokemon,col='Legendary', row="Generation", size=7)

fig.map(plt.scatter,"HP","Attack") 

fig.add_legend()
fig, (ax) = plt.subplots(1,figsize=(15,10))



sns.kdeplot(pokemon.loc[(pokemon['Legendary']==False), 'Total'], color='green',shade=True, ax=ax).set_title('Total', fontsize=16)

sns.kdeplot(pokemon.loc[(pokemon['Legendary']==True), 'Total'],shade=True, ax=ax).set_title('Total', fontsize=16)

plt.show()
sns.barplot(x='HP',y='Name',data=pokemon.sort_values('HP',ascending=False).head(10),palette='Set3_r')
fig,ax = plt.subplots(1,figsize=(10,10))

sns.scatterplot(data=pokemon, x='Attack', y='Defense',alpha=0.5,hue='Legendary')

plt.show()
tmp1=pd.DataFrame(pokemon['Type 1'].value_counts())

tmp2=pd.DataFrame(pokemon['Type 2'].value_counts())

disp = tmp2.join(tmp1)

disp['Type 2'] *= -1

disp
fig, ax = plt.subplots(figsize=(20,10))



sns.barplot(

    x='Type 1',

    y=disp.index,

    data=disp,

    color="#FFD97D",

    label='Type 1',

    ax=ax,

)



sns.barplot(

    x='Type 2',

    y=disp.index,

    data=disp,

    color="#60D394",

    label='Type 2',

    ax=ax,

)



    



ax.set_xlabel("")  # x 軸のラベル

ax.set_ylabel("count", fontsize=12)  # y 軸のラベル

# x 軸の範囲を左右対称になるように調整する。

max_val = pokemon['Type 1'].value_counts().max()*1.1

ax.set_xlim(-max_val, max_val)

ax.legend()  # 凡例追加



plt.show()
fig ,ax = plt.subplots(1,figsize=(15,10))

sns.countplot(pokemon['Type 2'],data=pokemon,hue="Legendary",ax=ax,palette='Set2')

plt.show()
pokemon
pokemon.drop(['Type 1','Type 2','Generation','#','Name','Legendary'], axis=1,inplace=True)

pokemon
fig,ax = plt.subplots(1,figsize=(10,10))

sns.scatterplot(data=pokemon,alpha=0.5)

plt.show()
pokemon.describe()
# 正規化

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

data_norm = ms.fit_transform(pokemon)
fig,ax = plt.subplots(1,figsize=(10,10))

sns.scatterplot(data=data_norm,alpha=0.5)

plt.show()
df = pd.DataFrame(data_norm)

df
# 学習データと評価データへ分割するライブラリの導入

from sklearn.model_selection import train_test_split

output_df = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

data_train, data_test, label_train, label_test = train_test_split(

        data_norm, output_df['Type 1'], test_size=0.3)
from sklearn import svm

Standard = svm.LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-3)

Standard.fit(data_train, label_train)

acc_svm = round(Standard.score(data_test, label_test) * 100, 2)

acc_svm
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(data_train, label_train)

acc_random_forest = round(random_forest.score(data_test, label_test) * 100, 2)

acc_random_forest
from xgboost.sklearn import XGBClassifier

model = XGBClassifier(learning_rate=0.001,n_estimators=2500,

                                max_depth=4, min_child_weight=0,

                                gamma=1, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)

model.fit(data_train, label_train)

xgb_acc = round(model.score(data_test, label_test) * 100, 2)

xgb_acc
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



param_grid = {'max_depth': [2, 3, 4, 5, 6, 7],

              'min_samples_leaf': [1, 3, 5, 7, 10]}



forest = RandomForestClassifier(n_estimators=10, random_state=0)

grid_search = GridSearchCV(forest, param_grid, iid=True, cv=5, return_train_score=True)



# GridSearchCVは最良パラメータの探索だけでなく、それを使った学習メソッドも持っています

grid_search.fit(data_train, label_train)