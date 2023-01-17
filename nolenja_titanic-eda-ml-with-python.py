%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
!pip install plotnine 
from plotnine import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, LabelBinarizer, scale, Normalizer, PowerTransformer, MaxAbsScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn. ensemble import RandomForestClassifier, VotingClassifier
!pip install xgboost
import xgboost as xgb 
#!pip install lightgbm
import lightgbm as lgb
# color palletes
male_female_pal = ['#3489d6', '#e64072']
survival_pal = ['#2a2a2a', '#ff0000']
sns.set_palette(survival_pal)
sns.set_style("whitegrid")

train = pd.read_csv('../input/train.csv')
print(train.shape)
train.head()
test = pd.read_csv('../input/test.csv')
print(test.shape)
test.head()
gender_submission = pd.read_csv('../input/gender_submission.csv')
print(gender_submission.shape)
gender_submission.head()
combi = pd.concat([train, test]) 
print(combi.shape)
combi.head()
combi.info()
!pip install missingno 
import missingno as msno

msno.matrix(combi)
combi["Sex"]
plt.figure(figsize=(6,6))
combi["Sex"].value_counts().plot.pie(autopct = "%1.1f%%",colors = sns.color_palette("prism",3),fontsize=20,
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("Gender composition",fontsize=20)
plt.show()
combi["Survived"]
plt.figure(figsize=(6,6))
combi["Survived"].value_counts().plot.pie(autopct = "%1.1f%%",colors = sns.color_palette("prism",3),fontsize=20,
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("Survival composition",fontsize=20)
plt.show()
(ggplot(combi)
 + aes(x='Sex', y='Survived')
 + geom_col()
 + ggtitle('Percentage of survival according to Sex')
 + theme(text=element_text(family='NanumBarunGothic'))
)
sns.barplot(x="Sex", y="Survived", data=combi) 
#print percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("Percentage of females who survived:", combi["Survived"][combi["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived:", combi["Survived"][combi["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
# females have a much higher chance of survival than males.

(ggplot(combi)
 + aes(x='Sex', y='Survived', fill='Pclass')
 + geom_col()
)
(ggplot(combi)
 + aes(x='Age', y='Fare') 
 + geom_point()
 + ggtitle('Fare accoprding to age')
 + theme(text=element_text(family='NanumBarunGothic'))
)
(ggplot(train)
 + aes(x='Age', y='Fare', color='Survived') 
 + geom_point()
 + stat_smooth()
 + ggtitle('Fare survival percentage and age')
 + theme(text=element_text(family='NanumBarunGothic'))
)
cat_cols = ['Pclass', 'Sex', 'Embarked']
fig, ax = plt.subplots(1, 3, figsize=(15, 4)) 
for ind, val in enumerate(cat_cols):
    sns.countplot(x=val, hue='Survived', data=combi, ax=ax[ind]) 
    plt.legend(['Not Survived', 'Survived'])
g = sns.FacetGrid(combi, col='Embarked', size=4) 
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=[1, 2, 3], hue_order=['male', 'female'], palette=male_female_pal) 
g.add_legend()
plt.show()
g = sns.FacetGrid(combi, col='SibSp', size=4) 
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=[1, 2, 3], hue_order=['male', 'female'], palette=male_female_pal) 
g.add_legend()
plt.show()
g=sns.factorplot(x="SibSp",y="Survived",data=combi,kind="bar",size=6)
g.set_ylabels("Survived Probability")
plt.show()
pal = {1:"green", 0:"Pink"}
sns.set(style="darkgrid")
plt.subplots(figsize = (10,8))
ax = sns.countplot(x = "Sex",  
                   hue="Survived", 
                   data = combi, 
                   linewidth=4, 
                   palette = pal)
## Fixing title, xlabel and ylabel
plt.title("Gender distribution according to survival", fontsize = 15, pad=40)
plt.xlabel("Sex", fontsize = 15);
plt.ylabel("Number of passenger survived", fontsize = 15)
## Fixing xticks
#labels = ['Female', 'Male']
#plt.xticks(sorted(train.Sex.unique()), labels)
## Fixing legends
leg = ax.get_legend()
leg.set_title("Survived")
legs = leg.texts
legs[0].set_text("No")
legs[1].set_text("Yes")
plt.show()
fig = plt.figure(figsize=(11,6),)
ax=sns.kdeplot(combi.loc[(combi['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(combi.loc[(combi['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age distribution according to survival', fontsize = 15, pad = 40)
plt.xlabel("Age", fontsize = 15, labelpad = 20)
plt.ylabel('Frequency', fontsize = 15, labelpad= 20);
f, ax = plt.subplots(1,2, figsize = (13,7))
combi[combi['Survived']==0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black',color='red') 
ax[0].set_title('Survived = 0')
x1 = list(range(0,85,5))
ax[0].set_xticks(x1)
combi[combi['Survived']==1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black',color='green') 
ax[1].set_title('Survived = 1')
x2 = list(range(0,85,5))
ax[1].set_xticks(x2)
combi["Gender_encode"] = (combi["Sex"] == "male").astype(int) 
print(combi.shape)
combi[["Sex", "Gender_encode"]].head()
combi.isnull().sum() 
mean_fare = train["Fare"].mean()
mean_fare
print("Fare(Mean) = ${0:.3f}".format(mean_fare))
combi["Fare_fillout"] = combi["Fare"] 


combi.loc[pd.isnull(combi["Fare"]), "Fare_fillout"] = mean_fare  
missing_fare = combi[pd.isnull(combi["Fare"])]  
print(missing_fare.shape)
missing_fare[["Fare", "Fare_fillout"]].head()
print(combi.info())  
combi["Embarked"].unique() 

embarked = pd.get_dummies(combi["Embarked"], prefix="Embarked").astype(np.bool) 

print(embarked.shape)

embarked.head()

combi = pd.concat([combi, embarked], axis=1)
print(combi.shape)
combi[["Embarked", "Embarked_C", "Embarked_Q", "Embarked_S"]].head()
combi["Family"] = combi["SibSp"] + combi["Parch"] 
print(combi.shape)

combi[["SibSp", "Parch", "Family"]].head(10)
combi.head()
train = combi[pd.notnull(combi["Survived"])] 
print(train.shape)
train.head()
test = combi[pd.isnull(combi["Survived"])] 
test.drop("Survived", axis=1, inplace=True) 
print(test.shape)
test.head()
print(train.info())     
feature_names = ["Pclass", "Gender_encode", "Age", "Fare_fillout", "Family"]
feature_names = feature_names + list(embarked.columns)
feature_names
label_name = "Survived"
label_name

X_train = train[feature_names]
print(X_train.shape)

X_train.head()
X_train.isnull().sum() 
X_train['Age'].fillna(X_train['Age'].mean(), inplace = True) 
X_train.isnull().sum() 
y_train = train[label_name]
print(y_train.shape)
y_train.head()

X_test = test[feature_names]
print(X_test.shape)
X_test.head()
X_test.isnull().sum() 
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True) 
X_test.isnull().sum() 
seed = 37
model = DecisionTreeClassifier(max_depth=5, random_state=seed)

model.fit(X_train, y_train)


prediction = model.predict(X_train)
acc_perceptron = round(accuracy_score(prediction, y_train) * 100, 2)
print(acc_perceptron)

prediction = model.predict(X_test)
print(prediction.shape)
prediction[:20]

















































# svm




















