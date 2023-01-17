import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df_titanic = pd.read_csv('../input/DS_Encuentro_05_titanic.csv')
df_titanic
df_titanic.info()
df_titanic.isna().sum()
df_titanic_filtered = df_titanic.copy()
df_titanic_filtered.drop(columns='Cabin', inplace=True)

df_titanic_filtered
df_titanic_filtered.isna().sum()
df_titanic_filtered['Embarked'].value_counts()
df_titanic_filtered['Embarked'].fillna(df_titanic_filtered['Embarked'].mode()[0], inplace = True)

df_titanic_filtered.isna().sum()
df_titanic_filtered['Age'].fillna(df_titanic_filtered.Age.mean(), inplace = True)
df_titanic_filtered.isna().sum()
df_titanic_filtered.describe(include=['O'])
len(df_titanic_filtered)
df_titanic_filtered.Survived.value_counts()
df_titanic_filtered.Embarked.value_counts()
df_titanic_filtered.Sex.value_counts()
df_titanic_filtered.Pclass.value_counts()
df_titanic_filtered.describe()
df_titanic_filtered.SibSp.value_counts()
df_titanic_filtered.Parch.value_counts()
df_titanic_filtered[['Sex','Survived']].groupby('Sex').mean()*100
pd.crosstab(index = df_titanic_filtered.Survived, columns = df_titanic_filtered.Sex, margins=True)
#Porcentaje total de pasajeros por sexo

df_titanic_filtered['Sex'].value_counts()*100/df_titanic_filtered.shape[0]
sns.barplot(x="Sex", y="Survived", data=df_titanic_filtered);
sns.distplot(df_titanic_filtered[df_titanic_filtered.Sex=='male'].Age)

sns.distplot(df_titanic_filtered[df_titanic_filtered.Sex=='female'].Age)
sns.catplot(data=df_titanic_filtered, y="Age",x="Sex",hue="Survived")
pd.crosstab(index=df_titanic_filtered.Survived, columns=df_titanic_filtered.Pclass, margins=True)
df_titanic_filtered[['Survived','Pclass']].groupby(['Pclass']).mean()*100
sns.catplot(data=df_titanic_filtered, y="Age",x="Pclass",hue="Survived")
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df_titanic_filtered);
pd.crosstab(index=df_titanic_filtered.Embarked, columns=df_titanic_filtered.Survived, margins=True)
df_titanic_filtered[['Survived','Embarked']].groupby(['Embarked']).mean()*100
sns.catplot(data=df_titanic_filtered, y="Age",x="Embarked",hue="Survived")
#retornamos la correlacion del data frame

corr = df_titanic_filtered.corr() 



# Con esa variable removemos las variables superiores ya que estan repetidas

bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)

corr = corr.where(bool_upper_matrix)



#Dibujamos el heatmap

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            cmap='RdYlGn');



fig=plt.gcf()

fig.set_size_inches(15,10)

plt.show()
# Armamos los bins (rangos)

bins = [0, 12, 18, 27, 60, 100]

# Esta variable nos va a servir para poder identificar cada rango

names = ['Childen', 'Teen', 'Young_adults', 'adults', 'Advanced_adults']



# Asignamos a esta variable la serie con las nuevas columnas en base a los bins y label

category = pd.cut(df_titanic_filtered.Age, bins, labels = names)
# Concatenamos al data frame a partir de una matriz dummie la variable category

df_titanic_filtered = pd.concat([df_titanic_filtered, pd.get_dummies(category)], axis=1)

df_titanic_filtered
pd.crosstab(index = df_titanic_filtered.Survived, 

            columns = df_titanic_filtered.Advanced_adults[df_titanic_filtered.Advanced_adults.values==1], 

            margins=True)
pd.crosstab(index = df_titanic_filtered.Survived, 

            columns = df_titanic_filtered.Childen[df_titanic_filtered.Childen.values==1], 

            margins=True)
df_titanic_filtered.loc[:,['Childen', 'Teen', 'Young_adults', 'adults', 'Advanced_adults']].sum().sort_values(ascending=False)
# Le sumaremos 1, para que cuente también a la persona individual

Family = df_titanic_filtered['SibSp'] + df_titanic_filtered['Parch'] + 1
#Agregamos la nueva columna al final del data frame

df_titanic_filtered['Family'] = Family

df_titanic_filtered
# Vemos en una tabla el tamaño de la familia vs la supervivencia

pd.crosstab(index = df_titanic_filtered['Family'], columns = df_titanic_filtered['Survived'], margins = True)
df_titanic_filtered[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# A través de una expresión regular obtenemos lo que esta en la feature "Name" desde la "coma" hasta el "." y se lo agregamos al dataframe como columna

df_titanic_filtered['Title'] = df_titanic_filtered.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df_titanic_filtered
pd.crosstab(index = df_titanic_filtered.Title, columns = df_titanic_filtered.Survived, margins = True)
# Reemplazamos lo dicho anteriormente

df_titanic_filtered['Title'].replace(to_replace=['Mlle', 'Ms', 'Mme'], value='Miss', inplace = True)

df_titanic_filtered['Title'].replace(to_replace=['Mme'], value='Mrs', inplace = True)

df_titanic_filtered['Title'].replace(to_replace=['Capt','Col','Countess','Dr','Jonkheer','Rev','Lady','Major','Sir', 'Don'], 

                                     value='Others',

                                     inplace = True)





pd.crosstab(index = df_titanic_filtered.Title, columns = df_titanic_filtered.Survived, margins = True)
# Visto con porcentajes

df_titanic_filtered[['Title','Survived']].groupby('Title').mean()*100
# Antes reemplazemos por valores numericos la columna "Sex" para que se pueda tener en cuenta en la correlación

df_titanic_filtered['Sex'] = df_titanic_filtered.Sex.map({'male':1, 'female':0})
#retornamos la correlacion del data frame

corr = df_titanic_filtered.corr() 



# Con esa variable removemos las variables superiores ya que estan repetidas

bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)

corr = corr.where(bool_upper_matrix)



#Dibujamos el heatmap

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            cmap='RdYlGn');



fig=plt.gcf()

fig.set_size_inches(15,10)

plt.show()