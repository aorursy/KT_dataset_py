#Wranlging
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Classification
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
rank = pd.read_csv('../input/timesData.csv')
rank.info()
rank.describe()
list(rank.columns.values)
#Distinct Years
rank.year.unique()
#Count of Institution by country in 2016
ins_count = rank[rank['year'] == 2016].groupby('country').size().sort_values(ascending = False)
plt.figure(figsize = (15,15))
ax = sns.barplot(x = ins_count.values, y = ins_count.index)
ax.set(xlabel = 'Country', ylabel = 'Number of Institution')
for i in ax.patches:
    ax.text(i.get_width()+1.0, i.get_y()+0.7,i.get_width().astype(int))
plt.show()
#Number of top 100 institution by country in 2016
top_count = rank[rank['year'] == 2016].head(100).groupby('country').size().sort_values(ascending = False)
plt.figure(figsize = (15,10))
ax = sns.barplot(x = top_count.values, y = top_count.index)
ax.set(xlabel = 'Country', ylabel = 'Number of top 100 Institution')
for i in ax.patches:
    ax.text(i.get_width()+0.2, i.get_y()+0.6,i.get_width().astype(int))
plt.show()
#Percentage of top 100 institution by country in 2016
per_count = top_count/ins_count
per_count.dropna(inplace = True)
per_count.sort_values(ascending = False, inplace = True)
plt.figure(figsize = (15,10))
ax = sns.barplot(x = per_count.values, y = per_count.index)
ax.set(xlabel = 'Country', ylabel = 'Percentage of top 100 Institution')
for i in ax.patches:
    ax.text(i.get_width(), i.get_y()+0.5,str(round(i.get_width()*100,1))+'%')
plt.show()
#Define a new dataframe
university_name = list(rank.university_name.unique())
non_decreasing = pd.DataFrame(data=university_name,columns=['university_name'])
non_decreasing.head()
#filtering 
def non_decreasing_rank(university_name):
    world_rank = list(rank[rank.university_name == university_name]['world_rank'])
    count = rank.groupby('university_name').size()[university_name]
    for i in range(1,count): #1,2,3
        if world_rank[i-1] < world_rank[i]:
            return False
    return True

#Moment of truth
non_decreasing[non_decreasing['university_name'].apply(non_decreasing_rank) == True]
#We are using 2015 figures
rank2015_raw = rank[rank.year == 2015]
rank2015_raw.drop(['country','year'],axis = 1, inplace = True)
rank2015_raw.head()
rank2015_raw.isnull().sum()
#I'm using forward fill because data is in order. I would assume university close to others in rank have similar values.
rank2015_raw.fillna(method='ffill',inplace = True)
rank2015_raw.isnull().sum()
rank2015_raw.isin(['-']).sum()
#Dropping rows with '-'
rank2015_raw = rank2015_raw[(~rank2015_raw['total_score'].isin(['-']))&(~rank2015_raw['income'].isin(['-']))]
rank2015_raw.isin(['-']).sum()
#Need to convert string to numeric
col = list(rank2015_raw.columns.values)

for i in range(2,8):
    rank2015_raw[col[i]] = pd.to_numeric(rank2015_raw[col[i]])
rank2015_raw.shape
factor = col[2:7]
for i in range(len(factor)):
    z = rank2015_raw[factor[i]]
    plt.figure(i)
    sns.regplot(x=z, y='total_score', data = rank2015_raw)
cor = pd.DataFrame()
for i in range(len(factor)):
    cor[factor[i]] = rank2015_raw[factor[i]]
cor['total_score'] = rank2015_raw.total_score
cor.corr() 
score = rank2015_raw.total_score
train = rank2015_raw[factor] 
lab_enc = preprocessing.LabelEncoder()
score_encoded = lab_enc.fit_transform(score)
x_train, y_train, x_test, y_test = train_test_split(train,score_encoded,train_size = 0.9, random_state = 0)
#Decision Tree
tree = DecisionTreeClassifier()
tree.fit(x_train,x_test)
y_pred = tree.predict(y_train)
y1 = lab_enc.inverse_transform(y_test)
y2 = lab_enc.inverse_transform(y_pred)
print(np.corrcoef(y1,y2))
sns.regplot(y1,y2)