# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
% matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
df=pd.read_csv(r"../input/Pokemon.csv")
df.head()
df.info()
# type 2 has some null values. we need to fill them with type 1
df['Type 2'].fillna(df['Type 1'],inplace=True)
df.info() # null values  filled with corressponding type 1 values.
#df.head()
# can drop # as indexing is already done
del df['#']
df.head()

df.columns.unique()
# consider type 1
df['Type 1'].value_counts()
# a count plot to better visualize.
sns.factorplot(x='Type 1',kind='count',data=df,size=5,aspect=3)
# a pie to visulaize the relative proportions.
labels = ['Water', 'Normal', 'Grass', 'Bug', 'ychic', 'Fire', 'Electric', 'Rock', 'Other']
sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]
colors = ['B', 'silver', 'G', '#ff4125', '#aa0b00', '#0000ff','#FFB6C1', '#FFED0D', '#16F5A7']
explode = (0.1, 0.0, 0.1, 0, 0.1, 0.0, 0.1, 0, 0.1) 
plt.pie(x=sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=0,counterclock=True)
plt.axis('scaled')
plt.title("Percentage of Different types of Type 1 Pokemons")
fig=plt.gcf()
fig.set_size_inches(9,9)
plt.show()
# consider type 2
df['Type 2'].value_counts()
# agin a countplot.
sns.factorplot(x='Type 2',kind='count',data=df,size=5,aspect=3)
# similarly a pie chart for type 2
labels = ['Poison', 'Fire', 'Flying', 'Dragon', 'Water', 'Bug', 'Normal',
       'Electric', 'Ground', 'Fairy', 'Grass', 'Fighting', 'Psychic',
       'Steel', 'Ice', 'Rock', 'Dark', 'Ghost']
sizes = [49,40,99,29,73,20,65,33,48,38,58,46,71,27,27,23,30,24]
colors = ['B', 'silver', 'G', '#ff4125', '#aa0b00', '#0000ff','#FFB6C1', '#FFED0D', '#16F5A7','B', 'silver', 'G', '#ff4125', '#aa0b00', '#0000ff','#FFB6C1','#ff4125', '#aa0b00']
explode = (0.1, 0.0, 0.1, 0, 0.1, 0.0, 0.1, 0, 0.1,0.0,0.1,0.0,0.1,0.0,0.1,0.0,0.1,0.0)
plt.pie(x=sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=0,counterclock=True)
plt.axis('scaled')
plt.title("Percentage of Different types of Type 2 Pokemons")
fig=plt.gcf()
fig.set_size_inches(9,9)
plt.show()
df['Legendary'].value_counts() # implies most of the pokemons were not legendary
sns.factorplot(x='Legendary',kind='count',data=df,size=5,aspect=1)
# similarly for Generation
df['Generation'].value_counts()
sns.factorplot(x='Generation',kind='count',data=df,size=5,aspect=1)
# viewing the descriptive measures of various  numeric features
df.describe()
sns.factorplot(data=df,kind='box',size=9,aspect=1.5)
cor_mat= df[['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,vmax=1.0, square=True,annot=True,cbar=True) 
cor_mat= df[['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,vmax=1.0, square=True,annot=True,cbar=True) 
# just to show full square. ::)))
cor_mat= df[['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr()
mask = np.array(cor_mat)
mask[:] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,vmax=1.0, square=True,annot=True,cbar=True) 
# similarly we can do this for type 2.
df.head()
# we can make a function that take 2 arguements -- the independent variable and the dependent variable.
# the dependent variable will be the categorical variable such as the type 1 or type 2 against which we want to plot--
# the independent variable which will be the numeric variable which we want to plot against the categorical variable.

def comp_against(dep_cat,indep_num,dfd):
#     fig, axes = plt.subplots(3,1)
#     fig.set_size_inches(15, 12)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='bar',size=5,aspect=3)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='swarm',size=5,aspect=3)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='box',size=5,aspect=3)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='strip',size=5,aspect=3)
    sns.factorplot(x=dep_cat,y=indep_num,data=dfd,kind='violin',size=5,aspect=3)
# now we can call the function like this. Below I have used Type 1 like a dep variable. Similarly we can do for others.
comp_against('Type 1','Total',df)
comp_against('Type 1','HP',df)
comp_against('Type 1','Attack',df)
comp_against('Type 1','Defense',df)
comp_against('Type 1','Sp. Atk',df)
comp_against('Type 1','Sp. Def',df)
# now similarly we can change the categorical variable like Type 2 etc... 
# and plot various numeric features like Total ,HP etc etc... .
def comp_pok(name1,name2,param):
    a = df[(df.Name == name1) | (df.Name ==name2)]
    sns.factorplot(x='Name',y=param,data=a,kind='bar',size=5,aspect=1,palette=['#0000ff','#FFB6C1'])
    
# calling the function with differnt paraemters for two dummy pokemons ---   Bulbasaur and Ivysaur
comp_pok('Bulbasaur','Ivysaur','Total')
comp_pok('Bulbasaur','Ivysaur','HP')
comp_pok('Bulbasaur','Ivysaur','Attack')
comp_pok('Bulbasaur','Ivysaur','Sp. Atk')
comp_pok('Bulbasaur','Ivysaur','Sp. Def')
comp_pok('Bulbasaur','Ivysaur','Defense')  
# and similarly... we can pass the names of the pokemons and the parameter to compare for any 2 pokemons.

