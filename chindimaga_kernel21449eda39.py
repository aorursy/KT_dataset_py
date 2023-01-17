# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import re



import warnings

warnings.filterwarnings('ignore')



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')



from collections import Counter



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Decision_Tree

from sklearn import tree
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

combine = [train_df, test_df]

dataset =  pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
train_df.shape, test_df.shape, dataset.shape
# Acquire filenames

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df.head()
train_df.describe(include=['object'])
train_df.describe()
train_df.info()

print('_'*40)

test_df.info()
# Summarie and statistics

train_df.describe()



# Review survived rate using percentiles

#train_df['Survived'].quantile([.61,.62])

#train_df['Pclass'].quantile([.4, .45])

#train_df['SibSp'].quantile([.65, .7])

#train_df[['Age', 'Fare']].quantile([.05,.1,.2,.4,.6,.8,.9,.99])
# Outlier detection 



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   
Outliers_to_drop = detect_outliers(train_df,2,["Age","SibSp","Parch","Fare"])
print(train_df.isnull().sum().sort_values(ascending=False).head())

test_df.isnull().sum().sort_values(ascending=False).head()
train_df.loc[Outliers_to_drop]
train_df.describe()
train_df.describe(include=['object'])
test_df.describe()
g = sns.heatmap(train_df[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True,fmt = ".2f",cmap = "coolwarm",alpha=1,vmin=-1, vmax=1)
# Explore SibSp feature vs Survived

g = sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Parch feature vs Survived

g  = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Parch feature vs Survived

g  = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Age vs Survived

g = sns.FacetGrid(train_df, col='Survived')

g = g.map(sns.distplot, "Age")
# Explore Age distibution 

g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 0) & (train_df["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 1) & (train_df["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xticks(range(0,100,10))

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
dataset["Fare"].isnull().sum()
#Fill Fare missing values with the median value

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
# Explore Fare distribution 

g = sns.distplot(dataset["Fare"], color="r", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")
# Apply log to Fare to reduce skewness distribution

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i+1))
g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")
g = sns.barplot(x="Sex",y="Survived",data=train_df)

g = g.set_ylabel("Survival Probability")
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Explore Pclass vs Survived

g = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Survived by Sex

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train_df,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
dataset["Embarked"].isnull().sum()

#Fill Embarked nan values of dataset set with 'S' most frequent value

dataset["Embarked"] = dataset["Embarked"].fillna("S")
# Explore Embarked vs Survived 

g = sns.factorplot(x="Embarked", y="Survived",  data=train_df,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Embarked 

g = sns.factorplot("Pclass", col="Embarked",  data=train_df,

                   size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
# convert Sex into categorical value 0 for male and 1 for female

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),fmt = ".2f",cmap="BrBG",annot=True,alpha=.8,vmin=-1, vmax=1)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

g = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

g.map(plt.hist, 'Age', bins=20, alpha=.7)

g.add_legend()
## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = dataset["Age"].median()

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med
dataset['Age'].isnull().sum()
g = sns.factorplot(x="Survived", y = "Age",data = train_df, kind="box")

g = sns.factorplot(x="Survived", y = "Age",data = train_df, kind="violin")
print("Before", dataset.shape)



dataset.drop(['Cabin', 'Ticket', 'PassengerId'], axis=True, inplace=True)



print('After', dataset.shape)
dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(dataset['Title'], dataset['Sex'])
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title 

dataset["Title"] = dataset["Title"].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona',"Ms" ,"Mme","Mlle"],'Rare')
g = sns.countplot(dataset["Title"])
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("survival probability")
# Drop Name variable

dataset.drop(labels = ["Name"], axis = 1, inplace = True)
# Create a family size descriptor from SibSp and Parch

dataset['Familysize'] = dataset['Parch'] + dataset['SibSp'] + 1
g = sns.factorplot(x="Familysize",y="Survived",data = dataset)

g = g.set_ylabels("Survival Probability")
# Create new feature of family size

bins = [0,1,2,4,8]

labels = ['Single','SmallF','MedF','LargeF']

dataset["Familysize"] = pd.cut(dataset["Familysize"], bins, labels=labels)
g = sns.factorplot(x="Familysize",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("survival probability")
# Drop Fsize/SibSp/Parch variable

dataset.drop(labels = ['SibSp','Parch'], axis = 1, inplace = True)
# Create categorical values for Pclass

dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset["Sex"] = dataset["Sex"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")

dataset = pd.get_dummies(dataset, columns = ["Embarked"],prefix="Em")

dataset = pd.get_dummies(dataset, columns = ["Familysize"],prefix="Fs")
dataset = pd.get_dummies(dataset)
dataset.head()
train_df.shape
## Separate train dataset and test dataset



train = dataset[:train_df.shape[0]]

X_test = dataset[train_df.shape[0]:]

X_test.drop(labels=["Survived"],axis = 1,inplace=True)
## Separate train features and label 

X_train = train.drop(labels = ["Survived"],axis = 1)

Y_train = train["Survived"].astype(int)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': [ 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
x=np.asarray(X_train)

y=np.asarray(Y_train)
import torch

import torch.nn as nn

import torch.nn.functional as F

X = torch.from_numpy(x).type(torch.FloatTensor)

y = torch.from_numpy(y).type(torch.LongTensor)
X_train.shape
#our class must extend nn.Module

class MyClassifier(nn.Module):

    def __init__(self):

        super(MyClassifier,self).__init__()

        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer

        #This applies Linear transformation to input data. 

        self.fc1 = nn.Linear(19,40)

        

        #This applies linear transformation to produce output data

        self.fc2 = nn.Linear(40,20)

        self.fc3 = nn.Linear(20,2)

        

    #This must be implemented

    def forward(self,x):

        #Output of the first layer

        x = self.fc1(x)

        #Activation function is Relu. Feel free to experiment with this

        x = F.tanh(x)

        #This produces output

        x = self.fc2(x)

        x = F.tanh(x)

        x = self.fc3(x)

        return x

        

    #This function takes an input and predicts the class, (0 or 1)        

    def predict(self,x):

        #Apply softmax to output. 

        pred = F.softmax(self.forward(x))

        ans = []

        #Pick the class with maximum weight

        for t in pred:

            if t[0]>t[1]:

                ans.append(0)

            else:

                ans.append(1)

        return torch.tensor(ans)

#Initialize the model        

model = MyClassifier()

#Define loss criterion

criterion = nn.CrossEntropyLoss()

#Define the optimizer

optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
#Number of epochs

epochs = 100000

#List to store losses

losses = []

for i in range(epochs):

    #Precit the output for Given input

    y_pred = model.forward(X)

    #Compute Cross entropy loss

    loss = criterion(y_pred,y)

    #Add loss to the list

    losses.append(loss.item())

    #Clear the previous gradients

    optimizer.zero_grad()

    #Compute gradients

    loss.backward()

    #Adjust weights

    optimizer.step()
from sklearn.metrics import accuracy_score

print(accuracy_score(model.predict(X),y))
z=np.asarray(X_test)

T = torch.from_numpy(z).type(torch.FloatTensor)

Y_pred=model.predict(T)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)

submission