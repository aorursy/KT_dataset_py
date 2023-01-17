# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# Any results you write to the current directory are saved as output.
#reading the data and also checking the computation time
%time fifa = pd.read_csv("/kaggle/input/fifa19/data.csv")
print(fifa.shape)
fifa.head()
def club(x):
    return fifa[fifa['Club']==x][['Name',"Jersey Number",'Position', 'Overall', 'Nationality', 'Age', 'Wage', 'Preferred Foot', 'International Reputation' , 'Weak Foot', 'Skill Moves', 'Work Rate','Height','Joined', 'Contract Valid Until','Value' ]]
data = club('Chelsea')
data
x= club("Chelsea")
x.shape
data.isnull().sum()
# filling the missing value for the contionus variables for proper data visualization.

data['Joined'].fillna('Jul 1, 2018', inplace = True)
data
plt.figure(figsize = (20,8))
plt.style.use('fivethirtyeight')
ax= sns.countplot('Position', data =data, palette = 'RdYlGn')
ax.set_xlabel(xlabel="Positions of players", fontsize=18)
ax.set_ylabel(ylabel = 'Players count', fontsize= 18)
ax.set_title(label = "Distribution of players position in Chelsea", fontsize = 20)
plt.show()
plt.figure(figsize =(16,8))
plt.style.use('classic')

sns.countplot(x= "Work Rate", data = data, palette = 'hls')
plt.title('Work rate among players in Chelsea', fontsize = 20)
plt.xlabel("Work rates of players", fontsize= 16)
plt.ylabel("Player count", fontsize = 16)
plt.show()
plt.rcParams['figure.figsize']= (10,5)
sns.countplot(data['Preferred Foot'], palette = 'copper')
plt.title('Preferred foot of players in chelsea', fontsize = 20)
plt.show()
labels = ['with 4.0 stars', 'with 3.0 stars', 'with 2.0 stars', 'with 1.0 star']
size = data['Weak Foot'].value_counts()
colors = plt.cm.RdYlBu(np.linspace(0,1,5))

plt.rcParams['figure.figsize']= (30,9)
plt.pie(size, labels = labels,  colors= colors)
plt.title('Weak foot Distribution among players')
plt.legend()
plt.show()




labels = ['with 4.0 index', 'with 3.0 index', 'with 2.0 index', 'with 1.0 index']
sizes = data['International Reputation'].value_counts()
colors = plt.cm.pink(np.linspace(0,1,5))



plt.rcParams['figure.figsize']= (18,9)
plt.pie(sizes,  labels = labels, colors = colors)
plt.title('International Reputation of players in the club')
plt.axis('equal')
plt.legend()
plt.show()
plt.figure(figsize = (18,8))
ax = sns.countplot(x= 'Skill Moves', data = data, palette ='pastel')
ax.set_title(label = 'Skilled players in Chelsea', fontsize = 20)
ax.set_xlabel(xlabel = 'Skill Ratings in stars', fontsize = 16)
ax.set_ylabel(ylabel = 'Players count', fontsize = 16)
plt.show()

plt.figure(figsize = (18,8))
ax = sns.countplot(x = 'Height', data = data, palette = 'copper')
ax.set_title(label = "Distribution of Height  in the club", fontsize = 20)
ax.set_xlabel(xlabel = 'Height in foot', fontsize = 16)
ax.set_ylabel(ylabel = 'Players Count', fontsize = 16)
plt.show()
plt.figure(figsize =(16,8))

sns.countplot(x='Nationality', data = data, palette= 'pink')
plt.title("Number of players form each country", fontsize=20)
plt.xlabel("Countries",fontsize=16)
plt.ylabel("Players Count", fontsize= 16)
plt.show()
sns.set(style = 'dark', palette = 'YlOrRd', color_codes = True)
x = data.Age
plt.figure(figsize =(16,8))
ax = sns.distplot(x, kde = False, color = 'r') 
ax.set_title(label= "Histogram of players age", fontsize = 20)
ax.set_xlabel(xlabel = "Players Age", fontsize = 16)
ax.set_ylabel(ylabel = "Players Count", fontsize = 16)
plt.show()
plt.figure(figsize =(16,8))
plt.scatter(data['Overall'], data['International Reputation'], s = data["Age"]*500, c = "maroon")
plt.title("Rating vs International Reputation", fontweight=20, fontsize=20)
plt.xlabel("Overall Ratings", fontsize = 16)
plt.ylabel("International Reputation", fontsize = 16)
plt.show()
data.sample(5)
data.loc[data.groupby(data['Position'])['Overall'].idxmax()][['Position', 'Name','Overall', 'Age','Wage' , 'Nationality']]
youngest = data.sort_values('Age', ascending = True)[["Name", "Age","Overall", "Nationality","Wage"]]
youngest.head(5)
Oldest = data.sort_values("Age", ascending = False)[["Name","Age","Nationality","Wage"]]
Oldest.head(5)