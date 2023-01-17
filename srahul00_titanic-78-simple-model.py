import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style 
style.use('ggplot')
#Reading Training Data
df = pd.read_csv('/kaggle/input/titanic/train.csv')
#Looking at the data
df
#Checking the Null values, and stats of the data
df.info()
df.describe(include = 'all')
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df['Title'].unique()
survivedPercentages = []
survivedCounts = []
fig, ax = plt.subplots()
for idx, i in enumerate(df['Title'].unique()):
    #print(i)
    tempdf = df[df['Title'] == i]
    totalCount = len(tempdf)
    survivedPercentages.append(len(tempdf[tempdf['Survived'] == 1])/totalCount*100)
    survivedCounts.append(len(tempdf[tempdf['Survived'] == 1]))
    
xrange = range(len(df['Title'].unique()))
ax.bar(xrange, survivedPercentages, color = "green")    
ax.set_xticks(xrange)
ax.set_xticklabels(df['Title'].unique(), rotation = 90)
ax.set_yticks(range(-5, 111, 5))
for idx, val in enumerate(survivedCounts):
    ax.text(idx, survivedPercentages[idx]+2, str(survivedCounts[idx]), horizontalalignment = 'center')
plt.title("Title vs Survived Percentage")
plt.xlabel("Title")
plt.ylabel("Survived Percentage (and Count)")
plt.show()
#df
#0/B - Boy, 1/G - Girl, 2/W - Woman, 3/M - Man (M to be filled as default)
#TitleDict = {'Master' : 'B', 'Miss' : 'G','Ms' : 'G', 'Mlle' : 'G', 'Mrs' : 'W', 'Mme' : 'W', 'Lady' : 'W', 'the Countess' : 'W'}
TitleDict = {'Master' : 1, 'Miss' : 1,'Ms' : 1, 'Mlle' : 1, 'Mrs' : 2, 'Mme' : 2, 'Lady' : 2, 'the Countess' : 2}
CabinDict = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8,}
EmbarkedDict = {'C' : 1, 'S' : 2, 'Q' : 3}
SexDict = {'female' : 0, 'male' : 1}
#Average of Mean and Median
fillAge = (df['Age'].mean() + df['Age'].median())/2
fillFare = (df['Fare'].mean() + df['Fare'].median())/2
print(f"Fill Age: {fillAge} \tFill Fare: {fillFare}")
df['Age'].fillna(fillAge, inplace = True)
df['Sex'] = df['Sex'].map(SexDict)

df['Title'] = df['Title'].map(TitleDict)
df['Title'].fillna(3, inplace = True)

df['Cabin'].fillna('C', inplace = True)
df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
df['Cabin'] = df['Cabin'].map(CabinDict)

df['Embarked'].fillna('S', inplace = True)
df['Embarked'] = df['Embarked'].map(EmbarkedDict)
plt.figure(figsize = (10, 7))
plt.title('Correlation Matrix')
sns.heatmap(df.corr(), annot = True, fmt = ".2f")
X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis = 1) 
y = df['Survived'].values

#Dummies to simulate One-Hot Encoding
X = pd.get_dummies(X)
columns = X.columns
#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
clf = RandomForestClassifier(n_estimators = 100, max_depth=7, min_samples_split=3, n_jobs=-1)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Accuracy on Validation Set: {acc*100}")
featureimp = []
for i in range(len(X[0])):
    Xtemp = X_test
    Xtemp[:][i] = 0
    score = clf.score(Xtemp, y_test)
    #featureimp.append(score)
    featureimp.append(acc - score)
#featureimp
plt.barh(columns, featureimp)
plt.title("Feature Importance")
plt.show()
X_Test = pd.read_csv('/kaggle/input/titanic/test.csv')
X_Test['Age'].fillna(fillAge, inplace = True)
X_Test['Sex'] = X_Test['Sex'].map(SexDict)

X_Test['Title'] = X_Test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
X_Test['Title'] = X_Test['Title'].map(TitleDict)
X_Test['Title'].fillna(3, inplace = True)

X_Test['Cabin'].fillna('C', inplace = True)
X_Test['Cabin'] = X_Test['Cabin'].apply(lambda x: x[0])
X_Test['Cabin'] = X_Test['Cabin'].map(CabinDict)

X_Test['Embarked'].fillna('S', inplace = True)
X_Test['Embarked'] = X_Test['Embarked'].map(EmbarkedDict)

X_Test['Fare'].fillna(fillFare, inplace = True)

#Changing from float64 to float32 since RF implicitly converts while fitting to float32
#Which doesn't happen during predicting
X_Test['Age'] = X_Test['Age'].astype(np.float32)
X_Test['Fare'] = X_Test['Fare'].astype(np.float32)
X_Test['Title'] = X_Test['Title'].astype(np.float32)
PID = X_Test['PassengerId']
X_Test.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True) 
X_Test
X_Test.info()
X_Test = pd.get_dummies(X_Test)
X_Test = scaler.fit_transform(X_Test)
ypred = clf.predict(X_Test)
#ypred
submission = pd.DataFrame({'PassengerId': PID, 'Survived': ypred})
submission.to_csv("TitanicRF.csv", index = False)
submission.head()