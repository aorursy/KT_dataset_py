# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
print('size of train data:',train.shape)
print('size of test data:',test.shape)
train.describe()
train.info()
train.PassengerId.value_counts()
train.Survived.value_counts()
train.Pclass.value_counts()
print("Unique values:",train.Name.nunique())

print('\n Sample values present:\n',train.Name.value_counts())
train.Sex.value_counts()
print("Age varies from: ",train.Age.min(),'yrs to',train.Age.max(),'yrs')
train.SibSp.value_counts()
train.Parch.value_counts()
train.Ticket.value_counts()
print("Fare varies from: ",train.Fare.min(),'to',train.Fare.max())
train.Cabin.value_counts()
train.Embarked.value_counts()
# Correlation matrix between numerical values

plt.figure(figsize=(10, 5))
cor = train.corr()
ax = sns.heatmap(cor,annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
#A bar plot of survival by sex
train.groupby(['Sex', 'Survived'])['Pclass'].count().plot(kind = 'bar')
plt.title('Comparing based on Sex and Survival')
plt.show()

#percentages of females vs. males that survive
d = train[train['Survived'] == 1].groupby('Sex').count()['Pclass']
(d / d.sum()) * 100
#SibSp feature vs Survived

g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
# Passengers from which class survived the most?

sns.countplot(x = 'Pclass', hue = 'Survived', data = train)
plt.show()

#print percentages of 1st vs. 2nd and 3rd class
train[train['Survived'] == 1]['Pclass'].value_counts(normalize = True) * 100
#Pclass vs. Age sliced by Survived

# create subplot plot
fig, axes = plt.subplots(1, 2, figsize = (15, 5))

# create violinplot plot using groupby
sns.violinplot(x = 'Pclass', y = 'Age', data = train, hue = 'Survived', split = True, ax = axes[0])
sns.violinplot(x = 'Sex', y = 'Age', data = train, hue = 'Survived', split = True, ax = axes[1])
plt.show()
#Embarked vs (No. of passengers, sex, Survived , Pclass)

# create subplot plot
fig, axes = plt.subplots(2, 2, figsize = (15, 10))

# create Bar (count) plot for Embarked vs. No. Of Passengers Boarded
sns.countplot(x = 'Embarked', data = train, ax = axes[0,0])

# create Bar (count) plot for Embarked vs. Male-Female Split
sns.countplot(x = 'Embarked', hue = 'Sex', data = train, ax = axes[0,1])

# create Bar (count) plot for Embarked vs Survived
sns.countplot(x = 'Embarked', hue = 'Survived', data = train, ax = axes[1,0])

# create Bar (count) plot for Embarked vs Pclass
sns.countplot(x = 'Embarked', hue = 'Pclass', data = train, ax = axes[1,1])
plt.show()
#age v/s fare with target variable(Survived)

sns.scatterplot(x = train['Age'], y = train['Fare'], hue = train['Survived'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()
# lets combine the data for data prep

test['Survived']=np.nan
train['data']='train'
test['data']='test'
test=test[train.columns]

combined = pd.concat([train,test], sort = False , ignore_index= True)
combined.head()
# Extracting relevant information from the Name feature and creating a new feature Title

combined['Name'][0].split(',')[1].split('.')[0]
combined['Name'].size
lst = []

for i in range(0,combined['Name'].size):
    a = combined['Name'][i].split(',')[1].split('.')[0]
    lst.append(a)
combined['Title'] = lst
combined.head()
combined['Title'].nunique()
combined.Title.value_counts()
titles_ignore = [' Dona',' Don', ' Rev', ' Dr', ' Mme',' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',' the Countess', ' Jonkheer']
def title_change(x):
    if x in titles_ignore:
        return("Others")
    else:
        return(x)
    
combined.Title.apply(title_change).unique()

combined['Title'] = combined.Title.apply(title_change)

combined['Title'] = combined['Title'].str.replace('Ms', 'Miss')
combined.Title.value_counts()
combined.head()
# creating another feature "family" from "SibSp" and "Parch"

combined['Family'] = combined['SibSp']+combined['Parch']+1
def familycat(x):
    if(x>4):
        return('Large')
    elif(x>=2):
        return("Small")
    else:
        return("Singles")
combined['Family_Category'] = combined.Family.apply(familycat)
combined.Family_Category.value_counts().plot(kind='pie',autopct='%1.2f%%')
plt.show()
combined.head()
#Missing values

combined.isna().sum()
combined.drop('Cabin',axis=1,inplace= True)
#Filling the missing values in age

combined['age_bins'] = pd.cut(x=combined['Age'], bins=[0,13,19,40,60,80])
a = combined.groupby('age_bins').count()['Age']
print('Frequency distribution of the customers: \n',a)
print('Median values:',combined.groupby('Title').median()['Age'])
print('\n Mean values:',combined.groupby('Title').mean()['Age'])
#Filling the null values in the Age feature based on the mean age of each of the title

def impute_age(cols):
    Age = cols[0]
    Title = cols[1]
    
    if pd.isnull(Age):
        
        if Title == 'Master':
            return 5
        elif Title == 'Miss':
            return 22
        elif Title == 'Mr':
            return 32
        elif Title == 'Mrs':
            return 37
        elif Title == 'Ms':
            return 28
        else:
            return 43
    else:
        return Age

combined['Age']=combined[['Age','Title']].apply(impute_age,axis=1)

combined.Age.isnull().sum()
combined.isna().sum()
#Missing value in Fare

combined[combined['Fare'].isnull()]
combined.groupby('Pclass').mean()['Fare']
combined.Fare.fillna(value=13.30,inplace=True)
#missing values in Embarked

#As we saw that maximum passengers boarded from Port S, we replace NaN with S.

combined.Embarked.fillna(value='S',inplace=True)
combined.isna().sum()
combined.drop(['Name','Ticket','age_bins','Family','SibSp','Parch'],axis=1,inplace=True)
combined.head()
#Encoding String Values to Numeric values

combined = pd.get_dummies(combined,columns=['Sex','Embarked','Title','Family_Category'],drop_first = True)
combined.head()
combined.info()
#splitting the data back into train and test as it was already provided

train = combined[combined['data']=='train']
train.drop(['data','PassengerId'],axis=1,inplace=True)

test = combined[combined['data']=='test']
submit = test['PassengerId']
test.drop(['Survived','data','PassengerId'],axis=1,inplace=True)

del combined
#For submission

submission = pd.DataFrame()
submission['PassengerId'] = submit
submission['Survived'] = np.nan
train["Survived"] = train["Survived"].astype(int)

y = train["Survived"]
X = train.drop(labels = ["Survived"],axis = 1)

#train test split for model building
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
#scaling the data
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score, roc_auc_score,roc_curve,classification_report
lr = LogisticRegression(solver='liblinear',random_state=3)
lr.fit(X_train,y_train)

y_test_pred = lr.predict(X_test)

print('Accuracy_score:',accuracy_score(y_test,y_test_pred))
#OR 
#CV
from sklearn import model_selection

scoresdt = model_selection.cross_val_score(lr, X_train, y_train, cv=10)
print(scoresdt)

print('\n The mean score we got from 10-Fold CV:',np.mean(scoresdt))
#Classification Report

print(classification_report(y_test,y_test_pred))
y_test_prob = lr.predict_proba(X_test)[:,1]

print('AUC:',roc_auc_score(y_test,y_test_prob))
fpr,tpr,thresholds = roc_curve(y_test,y_test_prob) #thresholds here are the cutoffs
plt.plot(fpr,tpr,'r')
plt.plot(fpr,fpr,'g')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
#Other Classification models

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
gnb = GaussianNB()
er = ExtraTreesClassifier()
xgb = XGBClassifier()
lgb = LGBMClassifier()
#CV

seed = 7

models = []

models.append(('Losgistic Regression', lr))
models.append(('Random Forest', rf))
models.append(('KNN', knn))
models.append(('Decision Tree', dt))
models.append(('Gaussian', gnb))
models.append(('ExtraTreesRegressor', er))
models.append(('XGBRegressor', xgb))
models.append(('LGBMRegressor', lgb))
#Cross validation and accuracy score comparision of various models

from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')

results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f " % (name, cv_results.mean())
    print(msg)
#Boxplot for algorithm comparision

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111)
fig.suptitle('Algorithm Comparison')
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
