import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('../input/pokemon/Pokemon.csv')
df.head()
# NUMERICAL FEATURE
# CATEGORICAL FEATURE

df.info()
print(len(df['Name'].unique()))
print('We can see here, coulmn Name has all unique features')
print('='*50)

print(df['Type 1'].unique())
print(len(df['Type 1'].unique()))
print('='*50)

print(df['Type 2'].unique())
print(len(df['Type 2'].unique()))
print('='*50)

print(df['Generation'].unique())
print(len(df['Generation'].unique()))
print('this is a discrete feature BTW')
print('='*50)

# we have:
# 3 categorical feature
# 1 binary (target feature) {we will convert it into discrete afterwards}
# 1 discrete feature
# 7 continous feature
df['Legendary'].replace(True,1,inplace=True)
df['Legendary'].replace(False,0,inplace=True)
categorical = [feature for feature in df.columns if df[feature].dtype == 'O' and feature not in 'Name']
categorical
continous = [feature for feature in df.columns if df[feature].dtype != 'O' and feature not in 'Generation'+'Legendary']
print(continous)
discrete = ['Generation','Legendary']
print(discrete)
df.Legendary.unique()
# Graphs for continous numerical features
# we make this graph for finding if any type of distribution is here in the feature

for feature in continous:
  plt.figure(figsize=(7,5))
  plt.hist(df[feature],bins=40)
  plt.xlabel(feature)
  plt.ylabel('count')
  plt.title(feature)
  plt.show()
# ralation of continous with target with help of scatter plot

for feature in continous:
    plt.scatter(df[feature],df['Legendary'])
    plt.title('correlation between ' +feature+ ' and Legendary')
    plt.xlabel(feature)
    plt.ylabel('Legendary')
    plt.show()
gb=df.groupby('Type 1')
print(gb['Legendary'].value_counts())
print('='*65)
gb=df.groupby('Type 2')
gb['Legendary'].value_counts()
# Graph for categorical features

plt.figure(figsize=(13,7))
sns.countplot(data=df,x='Type 1',hue='Legendary')
plt.show()
plt.figure(figsize=(13,7))
sns.countplot(data=df,x='Type 2',hue='Legendary')
plt.show()
gb = df.groupby('Generation')
gb['Legendary'].value_counts()
# Graphs for discrete numerical features

plt.figure(figsize=(13,7))
sns.countplot(data=df,x='Type 1',hue='Legendary')
plt.show()
# this graph lets us see corelation of features, between features and with the target feature.

plt.figure(figsize=(13,7))
sns.heatmap(df.corr(),annot=True)
# checking outliers with help of box plot 

for feature in continous:
    sns.boxplot(x=feature,data=df)
    plt.title(feature)
    plt.show()
df=pd.read_csv('../input/pokemon/Pokemon.csv')

df.head(10)
df.describe()

df['Type 2']
# this is a mathematical way to see find outliers, and many other things we can infer from it!
# importing all the necessary libraries for of feature engineering

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False,handle_unknown='error')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False) #setting with_mean False, is for a reason !

# here, we have missing values in Type 2, what we will do here is:
# fill it with 'No Type' and not with mode. As there are many pokemons with no Type 2 abilty.

df.iloc[:,3].fillna(value='No Type', inplace=True)
print(df.head(10))
# Here, I dropped all unique features as they are not going get us prediction.

df.drop(columns=['Name','#'],inplace=True)
# Here, I encoded all the values of categorical feature to numerical feature

temp = encoder.fit_transform(df.iloc[:,0:2])
temp = pd.DataFrame(temp)
df = pd.concat([df,temp],axis=1)
# Let's see if everything is A-OK

df 
# Now Split X and y

y = df['Legendary']
X = df.drop(columns=['Type 1','Type 2','Legendary'])
# Let's Scale our X for better model ! I have used StandardScaler, you can use any other here and see interesting effects !

temp=scaler.fit_transform(X)
X = pd.DataFrame(temp)
# Splitting our data for into train and test, for testing our accuracy

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train
# As our target is to classify, if the pokemon is legendary or not, We are going to use LogisticRegression !

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()

lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)
# Let's see our accuracy on train data !

lg.score(X_train,y_train)
# Now, Let's check our accuracy on test data !

lg.score(X_test,y_test)
# Checking our Model by 

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(lg, X_test, y_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
# Let's see our accuracy on train data !

rfc.score(X_train,y_train)
# Now, Let's check our accuracy on test data !

rfc.score(X_test,y_test)
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(rfc, X_test, y_test)