#import library
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras as kr
from keras.optimizers import SGD
import graphviz

from matplotlib import rcParams #deal with customizing plot parameters (fontsize, colorscheme, ...)
import re #use for string manipulation

%matplotlib inline

#import dataset
df_train = pd.read_csv("../input/train.csv") #891*12
df_test = pd.read_csv("../input/test.csv") #418*11
df_train.describe()
#Take a look to the estate/position of the passengers
df_train['Title'] = df_train.Name.apply(lambda x: re.search('([A-Z][a-z]+)\.', x).group(1))
#Do the same for the df_test
df_test['Title'] = df_test.Name.apply(lambda x: re.search('([A-Z][a-z]+)\.', x).group(1))
#Grouping some titles then visualizing
tit_dict = {"Capt": "Intelligentsia",
            "Col": "Intelligentsia",
            "Major": "Intelligentsia",
            "Dr": "Intelligentsia",
            "Rev": "Intelligentsia",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Sir": "Royalty",
            "the Countess": "Royalty",
            "Dona": "Royalty",
            "Lady": "Royalty",
            "Mme": "Mrs",
            "Mrs": "Mrs",
            "Mlle": "Miss",
            "Miss": "Miss",
            "Mr": "Mr",
            "Master": "Master"}

df_train['Title'] = df_train.Title.map(tit_dict)
df_test['Title'] = df_test.Title.map(tit_dict)

#Printing the chance to be survived by position
print("Title - Chances to be survived")
print(df_train.groupby("Title")["Survived"].mean())
#Data Visualization
plt.figure(figsize=(12,5))
sns.countplot(x='Title', data=df_train, palette='Set2', hue='Survived')
plt.xlabel('titles')
plt.ylabel('count')
plt.show()
#Age Distribution and Density
#Use the median to fulfill the NaN with age

died_age = df_train[(df_train["Age"]>=0) & (df_train["Survived"]==0)]
surv_age = df_train[(df_train["Age"]>=0) & (df_train["Survived"]==1)]

plt.figure(figsize=(12,5))

sns.distplot(died_age["Age"], color='g')
sns.distplot(surv_age["Age"], color='r')
plt.xlabel("Age")
plt.ylabel("Distribution and Density by Age")
plt.show()
#Group the median age w.r.t sex, pclass, title
age_grp = df_train.groupby(["Sex", "Pclass", "Title"])["Age"]
print(age_grp.median())
#Input to the Age NaN
df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby(['Sex', 'Pclass', 'Title']).Age.transform('median') #REMIND: pd.loc: access the cell with row & col

plt.figure(figsize=(12,5))
sns.distplot(df_train["Age"], color='r')
plt.xlabel("Age")
plt.show()
#Seperate by survivor
plt.figure(figsize=(12,5))

#use facetgrid
g = sns.FacetGrid(df_train, col='Survived')
#mapping a dataset onto multiple axes arrayed in a grid of rows and columns that correspond to levels of variables in the dataset
g = g.map(sns.distplot, "Age")
plt.show()
#Age intervals

intv = (0, 5, 12, 18, 25, 35, 60, 120)
cat = ['babies', 'children', 'teen', 'student', 'adult', 'elder', 'senior']
df_train["age_cat"] = pd.cut(df_train.Age, intv, labels=cat) #segment and sort data values into bins

#Do the same on the test
intv = (0, 5, 12, 18, 25, 35, 60, 120)
cat = ['babies', 'children', 'teen', 'student', 'adult', 'elder', 'senior']
df_test["age_cat"] = pd.cut(df_test.Age, intv, labels=cat) #segment and sort data values into bins
#Survived by age category
plt.figure(figsize=(12,5))

plt.subplot(211)
sns.countplot("age_cat", data=df_train, hue="Survived", palette="Set2")
plt.ylabel("count")
plt.xlabel("age_cat")
plt.title("Age Distribution")

plt.subplot(212)
sns.swarmplot(x="age_cat", y="Fare", data=df_train, hue="Survived", palette="Set2")
plt.ylabel("Fare Distribution")
plt.xlabel("age_cat")
plt.title("Fare Distribution")

plt.subplots_adjust(hspace=0.5) #amount of height reserved for space between subplots, expressed as a fraction of the average axis height
plt.show()
#Fare distribution to Survived or not
plt.figure(figsize=(12,5))
sns.distplot(df_train[df_train.Survived==0]["Fare"], color='r')
sns.distplot(df_train[df_train.Survived==1]["Fare"], color='g')
plt.title("Fare Distribution by Survived")
plt.xlabel("Fare")
plt.ylabel("Density")
plt.show()
#Treat the fare expend

#Fill NA with -0.5
df_train.Fare = df_train.Fare.fillna(-0.5)
#interval to categorize
quant = (-1, 0, 8, 15, 31, 600)
label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

df_train["Fare_cat"] = pd.cut(df_train.Fare, quant, labels=label_quants)

plt.figure(figsize=(12,5))
sns.countplot(x="Fare_cat", hue="Survived", data=df_train, palette='Set2')
plt.title("Count of survived & Fare expend")
plt.xlabel("fare_cat")
plt.ylabel("count")
plt.show()
#Do it with df_test

#Fill NA with -0.5
df_test.Fare = df_test.Fare.fillna(-0.5)
#interval to categorize
quant = (-1, 0, 8, 15, 31, 600)
label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

df_test["Fare_cat"] = pd.cut(df_test.Fare, quant, labels=label_quants)
#Work on name

#Drop some irrelevant
del df_train["Fare"]
del df_train["Age"]
del df_train["Ticket"]
del df_train["Cabin"]
del df_train["Name"]

#In df_test
del df_test["Fare"]
del df_test["Age"]
del df_test["Ticket"]
del df_test["Cabin"]
del df_test["Name"]
#Total survived or not
print(df_train.groupby("Survived")["PassengerId"].count())

plt.figure(figsize=(12,5))
sns.countplot(x="Survived", data=df_train, palette="Set2")
plt.title("Total Distribution by survived or died")
plt.xlabel('Target Distribution')
plt.ylabel('Count')
plt.show()
print(pd.crosstab(df_train.Survived, df_train.Sex))

plt.figure(figsize=(12,5))
sns.countplot(x="Sex", data=df_train, hue="Survived", palette='Set2')
plt.title('Sex Distribution by survived or not')
plt.xlabel('Sex Distribution')
plt.ylabel('Count')
plt.show()
#Pclass vs. Embarked
print(pd.crosstab(df_train.Pclass, df_train.Embarked))

plt.figure(figsize=(12,5))
sns.countplot(x="Embarked", data=df_train, hue="Pclass", palette='Set2')
plt.title("Embarked & Pclass")
plt.xlabel("Embarked with Pclass")
plt.ylabel("Count")
plt.show()
df_train["Embarked"] = df_train["Embarked"].fillna('S')

print(pd.crosstab(df_train.Survived, df_train.Embarked))

plt.figure(figsize=(12,5))
sns.countplot(x="Embarked", data=df_train, hue="Survived", palette='Set2')
plt.title("Class Distribution by survived or died")
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.show()
print(pd.crosstab(df_train.Survived, df_train.Pclass))

plt.figure(figsize=(12,5))
sns.countplot(x="Pclass", data=df_train, hue="Survived", palette="Set2")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.title("Class Distribution by Survived or died")
plt.show()
#SibSp & Parch
g = sns.catplot(x="SibSp", y="Survived", data=df_train, kind='bar', height=5, aspect=1.6, palette="Set2")
g.set_ylabels('Survived Probability')
g.set_xlabels('SibSp Number')
plt.show()
g = sns.factorplot(x="Parch", y="Survived", data=df_train, kind="bar", size=6, palette='Set2')
g = g.set_ylabels("Survival Probability")
del df_train["SibSp"]
del df_train["Parch"]

del df_test["SibSp"]
del df_test["Parch"]
df_train.head()
df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked', 'age_cat', 'Fare_cat', 'Title'], prefix=['Sex', 'Emb', 'Age', 'Fare', 'Prefix'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked', 'age_cat', 'Fare_cat', 'Title'], prefix=['Sex', 'Emb', 'Age', 'Fare', 'Prefix'], drop_first=True)
plt.figure(figsize=(15,12))
plt.title('Correlation of Features for Train Set')
sns.heatmap(df_train.astype(float).corr(), vmax=1.0, annot=True)
plt.show()
df_train.shape
train = df_train.drop(["Survived", "PassengerId"], axis=1) #Drop specified labels from rows or columns
train0 = df_train["Survived"]

test = df_test.drop(["PassengerId"], axis=1)

X_train = train.values
y_train = train0.values

X_test = test.values
X_test = X_test.astype(np.float64, copy=False)

#Feature Scaling
scaler = StandardScaler() #Standardize features by removing the mean and scaling to unit variance
X_train = scaler.fit_transform(X_train) #Fit to data, then transform it
X_test = scaler.fit_transform(X_test)
model = Sequential() 
#The Sequential model is a linear stack of layers
model.add(Dense(18,
                activation='relu',
                input_dim=19,
                kernel_initializer='uniform'))
#input_layer_neurons=20, hidden_layer_neurons/output=18
'''
There is no known way to determine a good network structure evaluating the number of inputs or outputs. 
It relies on the number of training examples, batch size, number of epochs, basically, in every significant parameter of the network.
Moreover, a high number of units can introduce problems like overfitting and exploding gradient problems. 
On the other side, a lower number of units can cause a model to have high bias and low accuracy values. 
Once again, it depends on the size of data used for training.

The term kernel_initializer is a fancy term for which statistical distribution or function to use for initialising the weights. 
In case of statistical distribution, the library will generate numbers from that statistical distribution and use as starting weights.
'''
model.add(Dropout(0.5))
'''
Dropout is a technique where randomly selected neurons are ignored during training. 
They are “dropped-out” randomly. 
This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

Generally, use a small dropout value of 20%-50% of neurons with 20% providing a good starting point. 
A probability too low has minimal effect and a value too high results in under-learning by the network.
'''
model.add(Dense(60,
                kernel_initializer='uniform',
                activation='relu'))
#hidden_layer_neurons=60
model.add(Dropout(0.5))
model.add(Dense(1,
                kernel_initializer='uniform',
                activation='sigmoid'))
#output_layers=1
model.summary()
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=60, epochs=30, verbose=2)
y_pred = model.predict(X_test)

submission = pd.read_csv("../input/gender_submission.csv", index_col='PassengerId')
submission['Survived'] = y_pred.astype(int)
submission.to_csv('TitanicKNN.csv')
score = model.evaluate(X_train, y_train, batch_size=30)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))