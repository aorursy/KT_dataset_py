import numpy as np



import pandas as pd

import seaborn as sns

import matplotlib.ticker as mtick # For specifying the axes tick format 

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')





from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sub=pd.read_csv('../input/gender_submission.csv')

test.head()
f_test=pd.merge(sub,test ,on='PassengerId', how='inner')

f_test
df=pd.concat([train,f_test]).reset_index(drop=True)

df 
df.head()

df.tail()
df.shape#(1309, 12)

train.shape#(891, 12)

test.shape#(418, 11)
#train.isnull().sum()

#test.isnull().sum()

df.isnull().sum()
df.describe()
df.info()

df['Age'].hist()

plt.xlabel("AGE")

plt.ylabel("number of persons")

plt.show()

df.hist(figsize=(10,8))

plt.show()
sns.countplot(df["Survived"],label="Count")

figsize=(10,8)

plt.show()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),linewidths=0.1,linecolor='black',square=True,cmap='summer')





plt.show()
player_features = (   "Survived",'Sex', 'Age', 'SibSp','Parch',  'Fare',  'Embarked')



from math import pi

idx = 1

plt.figure(figsize=(15,45))

for position_name, features in df.groupby(df['Pclass'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    

    # number of variable

    categories=top_features.keys()

    N = len(categories)



    # We are going to plot the first line of the data frame.

    # But we need to repeat the first value to close the circular graph:

    values = list(top_features.values())

    values += values[:1]



    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    # Initialise the spider plot

    ax = plt.subplot(10, 3, idx, polar=True)



    # Draw one axe per variable + add labels labels yet

    plt.xticks(angles[:-1], categories, color='grey', size=8)

 # Draw ylabels

    #ax.set_rlabel_position(0)

    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    # Plot data

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    # Fill area

    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(position_name, size=11, y=1.1)

    

    idx += 1

plt.show()

df.head()
titanic = players = df[[   'Name','Pclass','Age',  'Fare',  'SibSp','Sex','Ticket', 'Cabin', 'Embarked','Survived']]



titanic.head()
import requests

import random

from math import pi



#import matplotlib.image as mpimg

#from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)



def details(row, title,  age,ticket,gender, club,cabin,embarked,sibsp,survived):

    



        

    r = lambda: random.randint(0,255)

    colorRandom = '#%02X%02X%02X' % (r(),r(),r())

    

    if colorRandom == '#ffffff':colorRandom = '#a5d6a7'

    

    basic_color = '#37474f'

    color_annotate = '#01579b'

    

    #img = mpimg.imread(flag_image)

    

    plt.figure(figsize=(15,8))

    categories=list(titanic)[1:]

    coulumnDontUseGraph = ['SibSp','Sex','Ticket','Cabin','Embarked','Survived']

    N = len(categories) - len(coulumnDontUseGraph)

    

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]

    

    ax = plt.subplot(111, projection='polar')

    ax.set_theta_offset(pi / 2)

    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='Black', size=10)

    ax.set_rlabel_position(0)

    plt.yticks([10,30,50], ["10","30","50"], color="grey", size=10)

    plt.ylim(0,80)

    

    values = titanic.loc[row].drop('Name').values.flatten().tolist() 

    valuesDontUseGraph = [  sibsp,gender,ticket,cabin,embarked,survived]

    values = [e for e in values if e not in (valuesDontUseGraph)]

    values += values[:1]

    

    ax.plot(angles, values, color= basic_color, linewidth=1, linestyle='solid')

    ax.fill(angles, values, color= colorRandom, alpha=0.5)

    

    

    ax.annotate('Ticket: ' + ticket.upper(), xy=(10,10), xytext=(103, 138),

                fontsize= 12,

                color = 'white',

                bbox={'facecolor': color_annotate, 'pad': 7})

    

    ax.annotate('Sex: ' + gender.upper(), xy=(10,10), xytext=(70, 140),

                fontsize= 12,

                color = 'white',

                bbox={'facecolor': color_annotate, 'pad': 7})

    

    ax.annotate('Survived: ' +str(survived), xy=(10,10), xytext=(56, 170),

                fontsize= 12,

                color = 'white',

                bbox={'facecolor': color_annotate, 'pad': 7})

                      

                      

    ax.annotate('Age: ' + str(age), xy=(10,10), xytext=(43, 180),

                fontsize= 15,

                color = 'white',

                bbox={'facecolor': color_annotate, 'pad': 7})

    

    ax.annotate('Pclass: ' + str(club), xy=(10,10), xytext=(92, 168),

                fontsize= 12,

                color = 'white',

                bbox={'facecolor': color_annotate, 'pad': 7})

    plt.title(title, size=20, color= basic_color)
# defining a polar graph



def graphPolar(id = 0):

    if 0 <= id < len(df.PassengerId):

        details(row = titanic.index[id], 

                title = titanic['Name'][id], 

                age = titanic['Age'][id],

                ticket=titanic['Ticket'][id],

                gender=titanic['Sex'][id],

                cabin=titanic['Cabin'][id],

                embarked=titanic['Embarked'][id],

                sibsp=titanic['SibSp'][id],

                survived=titanic['Survived'][id],

                

                club =titanic['Pclass'][id])

    else:

        print('The base has 1309 records. You can put positive numbers from 0 to 1309')
graphPolar(0) 

plt.show()
graphPolar(10) 

plt.show()
dv=df["Survived"]



iv=df.drop(["PassengerId","Survived","Name","Ticket","Cabin"], axis=1)

iv.head()

#dv
from sklearn.preprocessing import LabelEncoder

X_labelencoder = LabelEncoder()

iv.iloc[:, 1] = X_labelencoder.fit_transform(iv.iloc[:, 1].astype(str))

iv.iloc[:, 6] = X_labelencoder.fit_transform(iv.iloc[:, 6].astype(str))

iv.head()

from sklearn.preprocessing import Imputer

# First create an Imputer , Stratergy means what we want to write in place of missed value

missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', axis = 0)  #if missing values are represented by 9999 then write same here

# Set which columns imputer should perform



missingValueImputer = missingValueImputer.fit (iv[['Age']])

# update values of X with new values

iv[['Age']] = missingValueImputer.transform(iv[['Age']])



# Set which columns imputer should perform



iv[['Fare']] = missingValueImputer.fit_transform(iv[['Fare']])

iv.head()

iv.head()
X_train=iv[:891]

X_test=iv[891:]

y_train=dv[:891]

y_test=dv[891:]

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score



print("Accuracy is",accuracy_score(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier

modelrandom = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3, criterion = 'entropy')



modelrandom.fit(X_train, y_train)

pred=modelrandom.predict(X_test)

pred

from sklearn.metrics import accuracy_score





print("Accuracy is",accuracy_score(y_test, pred))
estimators=modelrandom.estimators_[5]

labels=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None, feature_names=labels, filled = True))

display(SVG(graph.pipe(format='svg')))
#naive bayes clasifier
#GaussianNB is specifically used when the features have continuous values.



from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train, y_train)



prediction = model.predict(X_test)









from sklearn.metrics import accuracy_score



print(accuracy_score(y_test, prediction))


