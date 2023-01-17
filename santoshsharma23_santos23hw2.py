import pandas as pd
#gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

train_df = pd.read_csv("../input/titanic/train.csv")
survived=train_df[train_df["Survived"]==1]

not_survived=train_df[train_df["Survived"]==0]
#Finading correlation between survived and class

#Total

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count().sort_values(by='Survived', ascending=False)
#Survived

survived[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count().sort_values(by='Survived', ascending=False)
# Survived divided by Total for all classes

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Finding correlation between survived and sex

#Total male & female

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).count().sort_values(by='Survived', ascending=False)
#Survived male & female

survived[["Sex", "Survived"]].groupby(['Sex'], as_index=False).count().sort_values(by='Survived', ascending=False)
# Survived divided by Total for boh male and female

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Correlation between Embarked and survived

#Total

train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).count().sort_values(by='Survived', ascending=False)
#Correlation between Embarked and survived

#Survived

survived[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).count().sort_values(by='Survived', ascending=False)
train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import seaborn as sns
import matplotlib.pyplot as plt
#Survived vs Age

grid = sns.FacetGrid(train_df, col='Survived')

grid.map(plt.hist, 'Age', bins=20)
#Survived Pclass vs Age

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass')

grid.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare',ci=None)

grid.add_legend()
frames = [train_df, test_df]

combined_df=pd.concat(frames, ignore_index=True, sort = False)
dd=train_df['Ticket'].duplicated()

dd
dd[dd==True].count()
train_df.rename(columns={'Sex':'Gender'},inplace=True)

#test_df.rename(columns={'Sex':'Gender'})

#combined_df.rename(columns={'Sex':'Gender'})
test_df.rename(columns={'Sex':'Gender'},inplace=True)
combined_df.rename(columns={'Sex':'Gender'},inplace=True)
#Replacing female with 1 and male with 0

cleanup_nums = {"Gender":     {"female": 1, "male": 0}}
train_df.replace(cleanup_nums, inplace=True)

train_df.head()
test_df.replace(cleanup_nums, inplace=True)

test_df.head()
combined_df.replace(cleanup_nums, inplace=True)

combined_df.head()
train_df['Embarked'].count()
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
train_df['Embarked']=train_df['Embarked'].fillna(freq_port)
train_df['Embarked'].count()
train_df
test_df['Fare'].count()
freq_fare = test_df.Fare.dropna().mode()[0]
freq_fare
test_df['Fare']=train_df['Fare'].fillna(freq_fare)
test_df['Embarked'].count()
train_df['Fare'].count()
r=[-0.01,7.911,14.4541,31.01,512.4]

#r= pd.IntervalIndex.from_tuples([(-0.001, 7.91), (7.911, 14.454), (14.455, 31),(31.01,512.4)])

g=[0,1,2,3]

train_df['Fare']=pd.cut(train_df['Fare'],bins=r,labels=g)
train_df['Fare'].head(10)
train_df[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df['Age'].isnull().sum()
!pip install -U scikit-learn
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=10)

train_df[['Age','Survived']] = imputer.fit_transform(train_df[['Age','Survived']])

train_df['Age'].isnull().sum()