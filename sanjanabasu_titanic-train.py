import seaborn as sns

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt  

import seaborn as sns
data=pd.read_csv('../input/titanicdataset-traincsv/train.csv')
print("Train: rows:{} columns:{}".format(data.shape[0], data.shape[1]))
data.describe()
data['Age'].describe()
data.columns
data.info()
pal = {'male':"#ff9999", 'female':"#66b3ff"}

#draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=data,palette = pal)



#print percentages of females vs. males that survive

print("Percentage of females who survived:", data["Survived"][data["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", data["Survived"][data["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
# Create subplot

plt.subplots(figsize = (8,5))

sns.barplot(x = "Pclass", y = "Survived", data=data, linewidth=2)

plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 10)

plt.xlabel("Socio-Economic class", fontsize = 10);

plt.ylabel("% of Passenger Survived", fontsize = 10);

labels = ['1st', '2nd', '3rd']

val = [0,1,2] 

plt.xticks(val, labels);





#print percentages of 1st vs. 2nd and 3rd class

print("Percentage of 1st class who survived:", data["Survived"][data["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of 2nd class who survived:", data["Survived"][data["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of 3rd class who survived:", data["Survived"][data["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#create a subplot

f,ax=plt.subplots(1,2,figsize=(10,5))



# create bar plot using groupby

data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(color=['#a85ee0'],ax=ax[0])

ax[0].set_title('Survived vs Sex')



# create count plot

sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
# create subplot plot

f,ax=plt.subplots(1,2,figsize=(10,5))



# create bar plot using groupby

data['Pclass'].value_counts().plot.bar(color=['#080035','#0F006B','#8B80C7'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')



# create count plot

sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
# create subplot plot



f,ax=plt.subplots(1,2,figsize=(18,8))



# create violinplot plot using groupby



sns.violinplot("Pclass","Age", hue="Survived", data=data,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=data,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()

# create subplot plot

f,ax=plt.subplots(2,2,figsize=(20,8))



# create Bar (count) plot for Embarked vs. No. Of Passengers Boarded

sns.countplot('Embarked',data=data,ax=ax[0,0],color="#b4bf82")

ax[0,0].set_title('No. Of Passengers Boarded')



# create Bar (count) plot for Embarked vs. Male-Female Split

sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')



# create Bar (count) plot for Embarked vs Survived

sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')



# create Bar (count) plot for Embarked vs Pclass

sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()

ax= sns.boxplot(x="Pclass", y="Age", data=data)

ax= sns.stripplot(x="Pclass", y="Age", data=data, jitter=True, edgecolor="gray")

plt.show()
#create crosstab

tab = pd.crosstab(data['Sex'], data['Survived'])

print(tab)



dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Port embarked')

dummy = plt.ylabel('Percentage')
import seaborn as sns



sns.lmplot(x='Age', y='Fare', hue='Survived', 

           data=data.loc[data['Survived'].isin([1,0])], 

           fit_reg=False)
One_Way_Tables = pd.crosstab(index=data["Survived"],  # Make a crosstab

                     columns="count")                  # Name the count column



One_Way_Tables


survived_sex = pd.crosstab(index=data["Survived"], 

                           columns=data["Sex"])



survived_sex.index= ["died","survived"]



survived_sex
g = sns.heatmap(data.corr(),cmap="ocean",annot=True)
sns.pairplot(data, kind="scatter", hue="Survived", palette="Set2",plot_kws=dict(s=80, edgecolor="black", linewidth=0.4))

plt.show()

data = pd.read_csv('AirPassengers.csv')

#Parse strings to datetime type

data['Month'] = pd.to_datetime(data['Month'],infer_datetime_format=True) 

#convert from string to datetime

data = data.set_index(['Month'])
def plot_df(data, x, y, title="", xlabel='Passengers', ylabel='Value', dpi=100):

    plt.figure(figsize=(16,5), dpi=dpi)

    plt.plot(x, y, color='tab:red')

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()



plot_df(data, x=data.index, y=data['#Passengers'], title='Monthly AirPassengers from 1949-01 to 1960-12.')    