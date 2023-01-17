# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import math 
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.describe()
df = train.copy()
df.head()
# from train.describe() it is evident that only 38.38 % of the population on the ship survived , rest died
df.loc[(df.Survived == 1) & (df.Sex == "male") , :].count()
# there were 109 males across the ship who survived that accident
df.loc[(df.Survived == 1) & (df.Sex == "female") , :].count()
# there were 233 females across the ship who survived that accident
# look the following graph
sns.factorplot(x="Sex",col="Survived", data=df , kind="count",size=4, aspect=.7);
# this gives us the idea that males died more and females survived more
# similarly
sns.factorplot(x="Sex", hue = "Pclass" , col="Survived", data=df , kind="count",size=6, aspect=.7);
df.loc[(df.Survived == 1) & (df.Sex == "male") & (df.Pclass == 1)].count()
df.loc[(df.Survived == 1) & (df.Sex == "male") & (df.Pclass == 2) , :].count()
df.loc[(df.Survived == 1) & (df.Sex == "male") & (df.Pclass == 3) , :].count()
pd.crosstab(df.Pclass, df.Survived, margins=True).style.background_gradient(cmap='autumn_r')
# All in all including both the sexes 2nd class survived less than the other two clases
df.Survived[df.Pclass == 1].sum()/df[df.Pclass == 1].Survived.count()
df.Survived[df.Pclass == 2].sum()/df[df.Pclass == 2].Survived.count()
df.Survived[df.Pclass == 3].sum()/df[df.Pclass == 3].Survived.count()
# % survived in Pclass = 1  --> 62.96 %  , similarly calculated for others
sns.factorplot(x='Pclass',y='Survived', kind="point" ,data=df)
sns.factorplot('Pclass','Survived',kind="bar",hue='Sex',data=df)
# A cross-tabulation to further inspect
pd.crosstab([df.Sex, df.Survived], df.Pclass, margins=True).style.background_gradient(cmap='autumn_r')
# Almost all women in Pclass 1 and 2 survived and nearly all men in Pclass 2 and 3 died
# lets see how survivals varies with Embarked
sns.factorplot(x="Survived",col="Embarked",data=df ,hue="Pclass", kind="count",size=5, aspect=.7);
# this shows that those who were embarked S survived more than those who were survived C and then Q
# Most of the people who died were embarked S
# Also , people survived with embarked Q were mostly from Plass 3 females
# A more closer look with cross-tab
pd.crosstab([df.Survived], [df.Sex, df.Pclass, df.Embarked], margins=True).style.background_gradient(cmap='autumn_r')
# can also be viewed like this
plt.subplots(figsize = (10,5))
plt.title('Embarked vs Survived wih Sex')
sns.violinplot(x = "Survived", y = "Embarked", hue = "Sex",data = df)
plt.show()
# similarly with Pclass

sns.factorplot(x = "Survived", y = "Pclass",col = "Embarked" , hue = "Sex" , kind = "violin",data = df)

sns.factorplot(x="Sex", y="Survived",col="Embarked",data=df ,hue="Pclass",kind="bar",size=5, aspect=.7);
# Inferences from above graph

# the survived axis shows the % .
# which means embarked Q males in Pclass 1 and 2 were all died

# while embarked females in Pclass 1 and 2 all lived....
# also nearly Pclass 1 and 2 females of all embarked types lived
context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
df['Sex_bool']=df.Sex.map(context1)
df["Embarked_bool"] = df.Embarked.map(context2)
df.head()
correlation_map = df[['PassengerId', 'Survived', 'Pclass', 'Sex_bool', 'Age', 'SibSp',
       'Parch', 'Fare' , 'Embarked_bool']].corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig,ax= plt.subplots()
fig.set_size_inches(12,12)
sns.heatmap(correlation_map, mask=obj,vmax=.7, square=True,annot=True)
df.groupby("Pclass").Age.mean()

df.isnull().sum()
df.head()
for x in [train, test,df]:
    x['Age_bin']=np.nan
    for i in range(8,0,-1):
        x.loc[ x['Age'] <= i*10, 'Age_bin'] = i
df[["Age" , "Age_bin"]].head(10)
sns.factorplot('Age_bin','Survived', col='Pclass' , row = 'Sex',kind="bar", data=df)
sns.factorplot('Age_bin','Survived', col='Pclass' , row = 'Sex', kind="violin", data=df)
pd.crosstab([df.Sex, df.Survived], [df.Age_bin, df.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
#  All female in Pclass 3 and Age_bin = 5 died.
#  Males in Age_bin >= 2 and Pclass died more than survived or died greater than 50% .
sns.factorplot('SibSp', 'Survived', col='Pclass' , row = 'Sex', data=df )
#  Females in Pclass 1 and 2 with siblings upto 3 nearly all survived
#  For Pclass 3 , males and females showed a near decreasing trend as number of siblings increased .
#  For males, no survival rate above 0.5 for any values of SibSp. (less than 50 %)
pd.crosstab([df.Sex, df.Survived], [df.Parch, df.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
#  For males,all survival rates below 0.5 for any values of Parch, except for Parch = 2 and Pclass = 1.
sns.factorplot('Parch', 'Survived', col='Pclass' , row = 'Sex', kind="bar", data=df )
# the distribution of Age_bin , SibSp and Parch as follows
for x in [train, test , df]:
    x['Fare_bin']=np.nan
    for i in range(12,0,-1):
        x.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i
fig, axes = plt.subplots(4,1)
fig.set_size_inches(20, 18)
sns.kdeplot(df.SibSp , shade=True, color="red" , ax= axes[0])
sns.kdeplot(df.Parch , shade=True, color="red" , ax= axes[1])
sns.kdeplot(df.Age_bin , shade=True, color="red" , ax= axes[2])
sns.kdeplot(df.Fare , shade=True, color="red" , ax= axes[3])
plt.show()
# introducing Fare_bin the same way as done in the Age_bin above but with a gap of 50
df[["Fare" , "Fare_bin"]].head(10)
pd.crosstab([df.Sex, df.Survived], [df.Fare_bin, df.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
sns.factorplot('Fare_bin','Survived', col='Pclass' , row = 'Sex', data=df)
plt.show()
df_test = test.copy()
df_test.head()
df.drop(['PassengerId','Sex','Embarked','Name','Ticket', 'Cabin', 'Age', 'Fare'],axis=1,inplace=True)
df.head()
context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
df_test['Sex_bool']=df_test.Sex.map(context1)
df_test["Embarked_bool"] = df_test.Embarked.map(context2)
df_test.drop(['PassengerId','Sex','Embarked','Name','Ticket', 'Cabin', 'Age', 'Fare'],axis=1,inplace=True)
df_test.head()
df.isnull().sum()
df_test.isnull().sum()
#  Age_bin in both dataframes is still possessing null values
df_test.Age_bin.fillna(df_test.Age_bin.mean() , inplace=True)
df.Age_bin.fillna(df.Age_bin.mean() , inplace=True)
df.Embarked_bool.fillna(df.Embarked_bool.mean() , inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), df['Survived'], test_size=0.3, random_state=101)
MLA = []
x = [LinearSVC() , DecisionTreeClassifier() , LogisticRegression() , KNeighborsClassifier() , GaussianNB() ,
    RandomForestClassifier() , GradientBoostingClassifier()]

X = ["LinearSVC" , "DecisionTreeClassifier" , "LogisticRegression" , "KNeighborsClassifier" , "GaussianNB" ,
    "RandomForestClassifier" , "GradientBoostingClassifier"]

for i in range(0,len(x)):
    model = x[i]
    model.fit( X_train , y_train )
    pred = model.predict(X_test)
    MLA.append(accuracy_score(pred , y_test))
MLA
sns.kdeplot(MLA , shade=True, color="red")
#  this proves that much of the algorithms are giving the accuracy between 77 % to 80 % with some above 80 % .
#  thats a pretty much good estimation 
d = { "Accuracy" : MLA , "Algorithm" : X }
dfm = pd.DataFrame(d)
# making a dataframe of the list of accuracies calculated above
dfm   # a dataframe wilh all accuracies and their corresponding algorithm name
sns.barplot(x="Accuracy", y="Algorithm", data=dfm)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), df['Survived'], test_size=0.3, random_state=66)
model = KNeighborsClassifier(n_neighbors=6)
model.fit( X_train , y_train )
pred = model.predict(X_test)
answer = model.predict(df_test)
print (accuracy_score(pred , y_test))
#  lets check it till 30 neighbours that which has got the maximum accuracy score

KNNaccu = []
Neighbours = []

for neighbour in range(1,31):
    model = KNeighborsClassifier(n_neighbors=neighbour)
    model.fit( X_train , y_train )
    pred = model.predict(X_test)
    KNNaccu.append(accuracy_score(pred , y_test))
    Neighbours.append(neighbour)
d = { "Neighbours" : Neighbours , "Accuracy" : KNNaccu }
knndf = pd.DataFrame(d)
knndf.head()
sns.factorplot(x="Neighbours", y="Accuracy",size = 5 , aspect = 2 , data=knndf)
#  making a csv file of the predictions
d = { "PassengerId":test.PassengerId , "Survived":answer }
final = pd.DataFrame(d)
final.to_csv( 'titanic_again.csv' , index = False )
