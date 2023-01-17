import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import seaborn as sns
df = pd.read_csv("../input/train.csv") 

df
df.drop(['Ticket','Cabin'], axis=1, inplace=True)
df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.2f%%')

plt.show()
# specifies the parameters of our graphs

fig = plt.figure(figsize=(18,6), dpi=1600) 

alpha=alpha_scatterplot = 0.2 

alpha_bar_chart = 0.55



# lets us plot many diffrent shaped graphs together 

ax1 = plt.subplot2grid((2,3),(0,0))

# plots a bar graph of those who surived vs those who did not.               

df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

# this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1

ax1.set_xlim(-1, 2)

# puts a title on our graph

plt.title("Distribution of Survival, (1 = Survived)")    



plt.subplot2grid((2,3),(0,1))

plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)

# sets the y axis lable

plt.ylabel("Age")

# formats the grid line style of our graphs                          

plt.grid(b=True, which='major', axis='y')  

plt.title("Survival by Age,  (1 = Survived)")



ax3 = plt.subplot2grid((2,3),(0,2))

df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)

ax3.set_ylim(-1, len(df.Pclass.value_counts()))

plt.title("Class Distribution")



plt.subplot2grid((2,3),(1,0), colspan=2)

# plots a kernel density estimate of the subset of the 1st class passangers's age

df.Age[df.Pclass == 1].plot(kind='kde')    

df.Age[df.Pclass == 2].plot(kind='kde')

df.Age[df.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 



ax5 = plt.subplot2grid((2,3),(1,2))

df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

ax5.set_xlim(-1, len(df.Embarked.value_counts()))

# specifies the parameters of our graphs

plt.title("Passengers per boarding location")
plt.show()
fig = plt.figure(figsize=(18,6))



#create a plot of two subsets, male and female, of the survived variable.

#After we do that we call value_counts() so it can be easily plotted as a bar graph. 

#'barh' is just a horizontal bar graph

df_male = df.Survived[df.Sex == 'male'].value_counts().sort_index()

df_female = df.Survived[df.Sex == 'female'].value_counts().sort_index()



ax1 = fig.add_subplot(121)

df_male.plot(kind='barh',label='Male', alpha=0.55)

df_female.plot(kind='barh', color='#FA2379',label='Female', alpha=0.55)

plt.title("Who Survived? with respect to Gender, (raw value counts) "); plt.legend(loc='best')

ax1.set_ylim(-1, 2) 

ax1.set_yticklabels([])

#adjust graph to display the proportions of survival by gender

ax2 = fig.add_subplot(122)

(df_male/float(df_male.sum())).plot(kind='barh',label='Male', alpha=0.55)  

(df_female/float(df_female.sum())).plot(kind='barh', color='#FA2379',label='Female', alpha=0.55)

plt.title("Who Survived proportionally? with respect to Gender"); plt.legend(loc='best')

ax2.set_yticklabels([])

ax2.set_ylim(-1, 2)
fig = plt.figure(figsize=(18,4), dpi=1600)

alpha_level = 0.65



# building on the previous code, here we create an additional subset with in the gender subset 

# we created for the survived variable. I know, thats a lot of subsets. After we do that we call 

# value_counts() so it it can be easily plotted as a bar graph. this is repeated for each gender 

# class pair.

ax1=fig.add_subplot(141)

female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()

female_highclass.plot(kind='bar', label='female, highclass', color='#FA2479', alpha=alpha_level)

ax1.set_xticklabels(["Survived", "Died"], rotation=0)

ax1.set_xlim(-1, len(female_highclass))

plt.title("Who Survived? with respect to Gender and Class"); plt.legend(loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()

female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)

ax2.set_xticklabels(["Died","Survived"], rotation=0)

ax2.set_xlim(-1, len(female_lowclass))

plt.legend(loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()

male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=alpha_level)

ax3.set_xticklabels(["Died","Survived"], rotation=0)

ax3.set_xlim(-1, len(male_lowclass))

plt.legend(loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()

male_highclass.plot(kind='bar', label='male, highclass', alpha=alpha_level, color='steelblue')

ax4.set_xticklabels(["Died","Survived"], rotation=0)

ax4.set_xlim(-1, len(male_highclass))

plt.legend(loc='best')

plt.show()
sns.FacetGrid(df, col='Survived').map(plt.hist, 'Age', bins=20)

plt.show()
sns.FacetGrid(df, col='Pclass').map(plt.hist, 'Age', bins=20)

plt.show()