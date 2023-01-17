#imports and loading dataset



import os

print(os.listdir("../input"))

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df=pd.read_csv('../input/traint/traint.csv')

pd.read_csv('../input/traint/traint.csv')
#Get some info about the dataframe first

df=pd.read_csv('../input/traint/traint.csv')

dataframe_x = df



print("Spaltenanzahl=", len(dataframe_x.columns))





print("Anzahl an Reihen", len(dataframe_x.index))







print("Mittwelwert Age", dataframe_x['Age'].mean())



#Mittelwerte und Summen

print(df[:].mean(), df[:].sum)
#find all fields without data (=NULL) for each column

df=pd.read_csv('../input/traint/traint.csv')

titanic_df=pd.read_csv('../input/traint/traint.csv')

print("Missing values (total): ", np.count_nonzero(df.isnull(),))

titanic_df.isnull().sum()



df=pd.read_csv('../input/traint/traint.csv')

df.info()
titanic_df=pd.read_csv('../input/traint/traint.csv')

ax = titanic_df["Age"].hist(bins=15, color='teal', alpha=0.8)

ax.set(xlabel='Age', ylabel='Count')

plt.show()
train_data=pd.read_csv('../input/traint/traint.csv')

train_data["Age"].fillna(28, inplace=True)

train_data["Embarked"].fillna("S", inplace=True)

train_data.drop('Cabin', axis=1, inplace=True)



train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]

train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)



train_data.drop('SibSp', axis=1, inplace=True)

train_data.drop('Parch', axis=1, inplace=True)

train_data.drop('TravelBuds', axis=1, inplace=True)



train2 = pd.get_dummies(train_data, columns=["Pclass"])



train3 = pd.get_dummies(train2, columns=["Embarked"])



train4=pd.get_dummies(train3, columns=["Sex"])

train4.drop('Sex_female', axis=1, inplace=True)



train4.drop('PassengerId', axis=1, inplace=True)

train4.drop('Name', axis=1, inplace=True)

train4.drop('Ticket', axis=1, inplace=True)

train4.head(5)







df_final = train4







plt.figure(figsize=(15,8))

sns.kdeplot(titanic_df["Age"][df_final.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(titanic_df["Age"][df_final.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population')

plt.show()
plt.figure(figsize=(20,8))

avg_survival_byage = df_final[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")
plt.figure(figsize=(15,8))

sns.kdeplot(df_final["Fare"][titanic_df.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(df_final["Fare"][titanic_df.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare for Surviving Population and Deceased Population')

# limit x axis to zoom on most information. there are a few outliers in fare. 

plt.xlim(-20,200)

plt.show()
#survived per class

titanic_df=pd.read_csv('../input/traint/traint.csv')

sns.barplot('Pclass', 'Survived', data=titanic_df, color="darkturquoise")

plt.show()

titanic_df=pd.read_csv('../input/traint/traint.csv')

titanic_df_clean_age = titanic_df.dropna(subset=['Age'])



def scatter_plot_class(pclass):

    g = sns.FacetGrid(titanic_df_clean_age[titanic_df_clean_age['Pclass'] == pclass], 

                      col='Sex',

                      col_order=['male', 'female'],

                      hue='Survived', 

                      hue_kws=dict(marker=['v', '^']), 

                      height=6)

    g = (g.map(plt.scatter, 'Age', 'Fare', edgecolor='w', alpha=0.7, s=80).add_legend())

    plt.subplots_adjust(top=0.9)

    g.fig.suptitle('CLASS {}'.format(pclass))



# plotted separately because the fare scale for the first class makes it difficult to visualize second and third class charts

scatter_plot_class(1)

scatter_plot_class(2)

scatter_plot_class(3)
traint = pd.read_csv('../input/traint/traint.csv')

sns.violinplot(y="Age", x="Survived", data=traint)
traint = pd.read_csv('../input/traint/traint.csv')

#factor plot

g = sns.catplot("Pclass", "Survived", "Sex", data=traint, kind="bar", palette="muted", legend=True)

g = sns.catplot("Pclass", "Age","Survived", data=traint, kind="bar", palette="muted", legend=True) #Reihenfolge wichtig
df=pd.read_csv('../input/traint/traint.csv')



m,f=df[df['Survived']==1]['Sex'].value_counts()



plt.subplot(1,2,1)

plt.bar(0, m, 0.3, color='blue', label="m")

plt.bar(0.3, f, 0.3, color='pink', label="f")



plt.title("Survived")

plt.xlabel("Sex")

plt.ylabel("Count")

plt.legend(loc='best')



m_died,f_died=df[df['Survived']==0]['Sex'].value_counts()



plt.subplot(1,2,2)

plt.bar(0, m_died, 0.3, color='blue', label="m")

plt.bar(0.3, f_died, 0.3, color='pink', label="f")



plt.title("Died")

plt.xlabel("Sex")

plt.ylabel("Count")

plt.legend(loc='best')



plt.show()
plt.subplot(1,2,1) # Zeile, Spalte, Index des Subplots

plt.plot(died_class,"go")

plt.title("Class")

plt.subplot(1,2,2) #Zeile, Spalte, index des Subplots

plt.plot(died_age,"r+")

plt.title("Age")

plt.show


#work in progress



titanic_df=pd.read_csv('../input/traint/traint.csv')

df = pd.read_csv('../input/traint/traint.csv')



#Scatterplots

set1 = np.array([167,170,149,165,155,180,166,146,159,185,145,168,172,181,169])

set2 = np.array([86,74,66,78,68,79,90,73,70,88,66,84,67,84,77])



plt.xlim(140,200)

plt.ylim(60,100)

plt.scatter(set1, set2)

plt.title("Scatterplot for titanic.csv")

plt.xlabel("x-Axis (set1)")

plt.ylabel("y-Axis (set2)")

plt.show()