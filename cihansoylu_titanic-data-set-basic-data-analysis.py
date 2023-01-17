import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
data = pd.read_csv('../input/train.csv')
data.head()
# Which features has missing values?
data.apply(pd.isna).apply(np.any)
#What are the unique values?
print(data.Parch.unique())
print(data.Sex.unique()) 
print(data.SibSp.unique()) 
print(data.Pclass.unique()) 
print(data.Cabin.unique()) 
print(data.Embarked.unique())
#Clean the data

#Categorize the age feature after filling the missing values
def categorize_ages(df):
    df.Age = df.Age.fillna(-.5)
    bins = (-1, 0, 5, 12, 18, 30, 55, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
    df.Age = pd.cut(df.Age, bins, labels=group_names)  
    return df

#Simplify cabin number to a single letter after filling the missing values
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')  #fill NaN with N
    df.Cabin = df.Cabin.apply(lambda x: x[0]) #Replace with the first letter
    return df

#Drop irrelevant features
def drop_features(df):   
    return df.drop(['Ticket', 'Name'], axis=1)

#Categorize the fares after filling the missing values
def categorize_fares(df):
    df.Fare = df.Fare.fillna(-1)
    bins = (-2,0, 8, 15, 31, 52, 75, 93, 115, 135, 1000)
    group_names = ["unknown", "quartile_0", "quartile_1", "quartile_2", "quartile_3", "quartile_4", "quartile_5", 
                   "quartile_6", "quartile_7", "quartile_8"]
    df.Fare = pd.cut(df.Fare, bins, labels=group_names)
    return df

def simplfy_embarked(df):
    df.Embarked = df.Embarked.fillna('N')
    return df

def transform_features(df):
    df = categorize_ages(df)
    df = simplify_cabins(df)
    df = categorize_fares(df)
    df = drop_features(df)
    df = simplfy_embarked(df)
    return df
transform_features(data).head()
# How much effect does the gender have on survival?
print(data.Survived.value_counts())
print(data.Sex.value_counts())
print(data.groupby(['Sex'])['Survived'].value_counts(1))

# We see that only 1 in 5 male survived meanwhile 3 in 4 females survived. 
# Hence for a female the probability of survival is %74 and for a male it is %18. 
# Thus the gender is a big factor in determining if someone will survive or not. 


sns.barplot(x="Sex", y="Survived", data=data);
sns.factorplot(x="Sex", hue="Survived", data=data, kind='count');
#Do a similar analysis using other features.
print('How much effect do the Pclass, Age, SibSp, Parch, Fare, Cabin, Embarked have on the chances of survival?')

print(data.groupby(['Pclass'])['Survived'].value_counts(1))
print('This shows that the probability of survivals for the 1st, 2nd and 3rd class are %62, %47 and %24, respectively.') 
sns.factorplot(x="Pclass", hue="Survived", data=data, kind='count');

print(data.groupby(['Pclass', 'Sex'])['Survived'].value_counts())
#print(data.groupby(['Pclass', 'Sex']).get_group((1, 'female'))['Survived'].value_counts(1))
print('We see that the probability of survival for 1st and 2nd class females are %96 and %92 which is really high.')
sns.barplot(x="Pclass", y="Survived", hue ='Sex', data=data);

#Only 9 females in the 1st and the 2nd class did not survive. Let's find these data points.
data[(data.Sex == 'female') & (data.Survived == 0) & ((data.Pclass == 1)|(data.Pclass == 2))]
print(data.groupby(['Age'])['Survived'].value_counts(1))
print('The chances of survival for a baby is %70.')

sns.barplot(y="Age", x="Survived", hue='Pclass', data=data);
print(data.groupby(['SibSp'])['Survived'].value_counts())
print('Since the numbers are low, it is hard to make any conlusion on the effect of SibSp on survival. \
       However having 1 or 2 sibling/spouse increases the chances of survival.')

sns.factorplot(x="SibSp", hue="Survived", data=data, kind = 'count');

print(data.groupby(['Parch'])['Survived'].value_counts())
print('Same conclusion as SibSp.')

sns.factorplot(x="Parch", hue="Survived", data=data, kind = 'count');

print(data.groupby(['Fare'])['Survived'].value_counts())
print('The chances of survival increases as the fare increases')
sns.factorplot(y="Fare", hue="Survived", data=data, kind = 'count');
print(data.groupby(['Cabin'])['Survived'].value_counts())
print('The chances of survival increases as the fare increases')
sns.barplot(x="Cabin", y="Survived", data=data);

