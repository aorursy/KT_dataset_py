# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from mpl_toolkits.mplot3d import Axes3D
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Pokemon.csv", index_col=0)
data.head(10)
data.tail(10)
data.shape
data.columns
 
data = data.rename(columns={"#":"index", "Type 1":"Type1","Type 2":"Type2",
                           "HP":"HP", "Sp. Atk": "Special_Attack",
                           "Sp. Def":"Special_Defense"})
data.info()
#some values in TYPE2 are empty and thus they have to be filled or deleted
data['Type2'].fillna(data['Type1'], inplace=True) 
#fill NaN values in Type2 with corresponding values of Type
data.head()
#let see the correlation between features:

f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax ,cmap="Blues")
plt.show()

fig, axarr = plt.subplots(3, 2, figsize=(13, 13))
#ax = plt.subplot2grid((3,2),(0,0), rowspan=1, colspan=1, fig=None) also could be used

# Histogram of HP
sns.distplot(data['HP'], hist=True, kde=False, bins=50, color = 'blue', 
             hist_kws={'edgecolor':'black'}, ax=axarr[0][0], axlabel='HP')

# Histogram of Attack
sns.distplot(data['Attack'], hist=True, kde=False, bins=50, color = 'blue', 
             hist_kws={'edgecolor':'black'},ax=axarr[0][1], axlabel='Attack' )

# Histogram of Defense
sns.distplot(data['Defense'], hist=True, kde=False, bins=50, color = 'blue', 
             hist_kws={'edgecolor':'black'}, ax=axarr[1][0], axlabel='Defense')

# Histogram of Special Attack
sns.distplot(data['Special_Attack'], hist=True, kde=False, bins=50, color = 'blue', 
             hist_kws={'edgecolor':'black'}, ax=axarr[1][1], axlabel='Special Attack')

#Histogram of Special Defense
sns.distplot(data['Special_Defense'], hist=True, kde=False, bins=50, color = 'blue', 
             hist_kws={'edgecolor':'black'}, ax=axarr[2][0], axlabel='Special Defense')

# Histogram of Speed
sns.distplot(data['Speed'], hist=True, kde=False, bins=50, color = 'blue', 
             hist_kws={'edgecolor':'black'}, ax=axarr[2][1], axlabel='Speed')

plt.show()
# Histogram of Total

sns.distplot(data['Total'], hist=True, kde=False, bins=50, color = 'purple', 
             hist_kws={'edgecolor':'black'}, axlabel='Total')
plt.axvline(data['Total'].mean(),linestyle='dashed',color='yellow') #line on average of Total

figsize=(5, 5)

plt.show()
%matplotlib inline
plt.rcParams['figure.figsize']=10,10  #to adjust the plot size

df = data.drop(['Total', 'Generation', 'Legendary'], axis=1)
sns.boxplot(data=df) 

plt.show()
# Density Plot
sns.kdeplot(data.Attack, data.Defense)
plt.show()
#prepare data frame
Ndf = data.iloc[:50,:]

# Creating trace1
trace1 = go.Scatter(y = Ndf.Attack,
                    x = Ndf.index,
                    mode = "lines+markers",
                    name = "Attack",
                    marker = dict(color = 'blue'))
# Creating trace2
trace2 = go.Scatter(y = Ndf.Defense,
                    x = Ndf.index,
                    mode = "lines",
                    name = "Defence",
                    marker = dict(color = 'red'))

dataS = [trace1, trace2]
layout = dict(title = 'Attack and Defense of the pokemons',
              xaxis= dict(title= 'Index',ticklen= 5,zeroline= False)
             )
fig = dict(data = dataS, layout = None)
iplot(fig)
x2011 = data.Type1[data.Generation == 1]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=400,
                          height=300
                         ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
                    y = Ndf.Attack,
                    x = Ndf.Defense,
                    z = Ndf.index,
                    mode = "markers",
                    name = "Attack",
                    marker=dict(size=10,color='rgb(255,0,0)'))

Data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=Data, layout=layout)

iplot(fig)
# Joint Distribution Plot
sns.jointplot(x='Attack', y='Defense', data=data)
plt.show()
#let's compare the Attack and Defense stats

sns.lmplot(x='Attack', y='Defense', data=data, size=4, aspect=2, fit_reg=True, 
           hue='Generation')
plt.show()

data.groupby(['Generation']).size().reset_index(name='counts')
#data.groupby(['Generation']).size()
sns.set(style="darkgrid")
sns.set_context(font_scale =20)
sns.countplot(x='Generation',data=data,saturation=0.75,palette="Blues_d", hue='Legendary')
plt.xlabel('Generation', fontsize=15) 
plt.ylabel('Number', fontsize=15) 
plt.title('Number of the pokemons around different generations', fontsize=15)

plt.show()
DType=data.groupby(['Generation','Type1']).count().reset_index()
DType=DType[['Generation','Type1','Total']]
DType=DType.pivot('Generation','Type1','Total')
DType[['Water','Fire','Grass','Dragon','Normal','Rock','Flying','Electric']].plot(marker='*')
fig=plt.gcf()
fig.set_size_inches(7,5)

DType=data.groupby(['Generation','Type2']).count().reset_index()
DType=DType[['Generation','Type2','Total']]
DType=DType.pivot('Generation','Type2','Total')
DType[['Water','Fire','Grass','Dragon','Normal','Rock','Flying','Electric']].plot(marker='*')
fig=plt.gcf()
fig.set_size_inches(7,5)

plt.show()
plt.subplots(figsize = (15,5))
plt.title('Strongest Genaration')
sns.violinplot(x='Generation',data=data, y = "Total")
plt.show()
print('The unique  pokemon types are: ','\n',data['Type1'].unique(),'\n','\n') #unique types of column
print('The number of unique types are: ''\n',data['Type1'].nunique()) #count of unique values 

gdf = data.groupby(by=['Type1', 'Type2']).agg(['max','min']).head(10)
gdf
data.groupby(['Type1']).size().reset_index(name='counts')
plt.subplots(figsize=(10,15))
ax = data['Type1'].value_counts().sort_values(ascending=True).plot.barh(width=.9,
                                                    color=sns.color_palette('inferno',40))
ax.set_xlabel('count')
ax.set_ylabel('types')
plt.title("Number of various pokemon/type1",loc='left', fontsize=16)

plt.show()
data.groupby(['Type2']).size().reset_index(name='counts')
#Lets plot both together in 2 rows 1 cols
#first row, first col
ax1 = plt.subplot2grid((2,1),(0,0))
labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'
sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]
explode = (0, 0, 0.0, 0, 0, 0, 0, 0, 0.1)  # explode the last slice 
plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True, startangle=90, 
        colors=sns.cubehelix_palette(8, start=.5, rot=-.75), 
        labeldistance=1.1, rotatelabels = False)

plt.axis('equal')
plt.title("Distribution of various pokemon/type1",loc='left', fontsize=16)
plt.plot()
fig=plt.gcf()
fig.set_size_inches(12,12)


#Second row first column
ax1 = plt.subplot2grid((2,1), (1, 0))

labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'
sizes = [73, 65, 58, 20, 71, 40, 33, 23,286]
explode2 = (0, 0, 0, 0, 0.1, 0, 0, 0, 0)
plt.pie(sizes, labels=labels,explode=explode2,autopct='%1.1f%%', shadow=True, startangle=90, 
        colors=sns.cubehelix_palette(8, start=.5, rot=-.75), 
        labeldistance=1.1, rotatelabels = False)

plt.axis('equal')
plt.title("Distribution of various pokemon/type2",loc='right', fontsize=16)
plt.plot()
fig=plt.gcf()
fig.set_size_inches(12,12)
plt.show()

plt.subplots(figsize = (15,3))
plt.title('Attack by Type1')
sns.violinplot(x = "Type1", y = "Attack",data = data)
plt.ylim(-100,250)

plt.subplots(figsize = (15,3))
plt.title('Attack by Type2')
sns.violinplot(x = "Type2", y = "Attack",data = data)
plt.ylim(-100,250)
plt.show()
# Melt DataFrame
#Let's melt all 6 of the stat columns into one
#The new Stat column indicates the original stats (HP, Attack, Defense, Sp. Attack...)
TD_data = data.drop(['Total', 'Generation','Legendary'], axis=1)
M_df = pd.melt(TD_data, 
                    id_vars=["Name", "Type1", "Type2"], # Variables to keep
                    var_name="Stat") # Name of melted variable
M_df.head()
# M_df Swarmplot 
plt.figure(figsize=(15,5))
sns.swarmplot(x='Stat', y='value', data=M_df, hue='Type1', palette=sns.color_palette("coolwarm", 18)
              , split=True)
# 4. Adjust the y-axis
plt.ylim(0, 260)
# 5. Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()
#Psychic Vs Poison
fire=data[(data['Type1']=='Psychic') | ((data['Type2'])=="Psychic")] #fire contains all fire pokemons
water=data[(data['Type1']=='Poison') | ((data['Type2'])=="Poison")]  #all water pokemins
plt.scatter(fire.Attack.head(50),fire.Defense.head(50),color='Y',label='Psychic',marker="*",s=50) 
plt.scatter(water.Attack.head(50),water.Defense.head(50),color='G',label="Poison",s=25)
plt.xlabel("Attack")
plt.ylabel("Defense")
plt.legend()
plt.plot()
fig=plt.gcf()  #get the current figure using .gcf()
fig.set_size_inches(12,6) #set the size for the figure
plt.show()
#we'll build a model with only a some featuresthat are int type
#We select multiple features
#Speed is prediction Target = y
pokemon_features = ['HP', 'Attack', 'Defense', 
                    'Special_Attack', 'Special_Defense','Generation']
DatatoModel = data[pokemon_features].reindex()
DatatoModel.head()
from sklearn.tree import DecisionTreeRegressor
y = data['Speed']  #prediction target
# Define model. Specify a number for random_state to ensure same results each run
ModelData = DecisionTreeRegressor(random_state=1)

# Fit model
ModelData.fit(DatatoModel, y) 
print("Making predictions for the following 5 pokemon:")
print(DatatoModel.head())
print("The predictions are")
print(ModelData.predict(DatatoModel.head()))
data.groupby('Legendary').apply(np.mean)
plt.figure(figsize=(12,6))
sns.distplot(data[data['Legendary']==False].Total, color="red", label="False" )
sns.distplot(data[data['Legendary']==True].Total, color="skyblue", label="True")
legend = [True, False]
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()
%matplotlib inline

sns.boxplot(x='Legendary', y='Total', data=data)
plt.show()
data['Legendary'].value_counts()
sns.countplot(x='Legendary', data = data, palette = 'hls')
plt.show()
NewData = data.copy()
NewData['Legendary'].replace([True,False],[1,0],inplace=True)
NewData['Legendary'].tail(10)
#we'll build a model with the data that categorical features droped.
#We select multiple features

features = ['HP', 'Attack', 'Defense', 
                    'Special_Attack', 'Special_Defense','Generation','Legendary']
LegModData = NewData[features].reindex()
LegModData.head()
#LegModData['Legendary'] = LegModData_target

decisiontree = DecisionTreeClassifier() # defining  new object
train = LegModData[50:]   # seperated the first 50 rows are as test rest as train dataset
test = LegModData[:50]

x_train = LegModData.drop('Legendary', axis=1) # x_train as legendary state droped
y_train = LegModData['Legendary']  # y_train is the legendary value

x_test = test.drop('Legendary', axis=1) # same thing for test dataset
y_test = test['Legendary']

decisiontree.fit(x_train, y_train) # model decisiontree fit to x_train, y_train values

pred = decisiontree.predict(x_test) # make prediction to the model about x_test and keep in pred 

print("accuracy:", accuracy_score(y_test, pred)) #to test the accuracy of the model
