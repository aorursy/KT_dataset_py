import os
print(os.listdir("../input"))
# manipulating dataframes
import pandas as pd
import numpy as np
# visualizing libraries
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
titanic = pd.read_csv('../input/train.csv')
titanic.head()
titanic.shape
titanic.dtypes
#summary statistics
titanic.describe()
#plots to identify any patterns or insights
titanic.hist(figsize=(10,10))
#New Feature:Title
def title(x):
    if 'Mr.' in x:
        return 'Mr'
    elif 'Mrs.' in x:
        return 'Mrs'
    elif 'Master' in x:
        return 'Master'
    elif 'Miss.' in x:
        return 'Miss'
    else:
        return 'Other'
# feature for the title of each person
titanic['Title'] = titanic['Name'].apply(title)

titanic['Title'].value_counts()
#New Feature: Mother
def mother(df):
    if df['Sex'] == 'female' and df['Parch'] > 0 and df['Age'] > 18 and df['Title'] != 'Miss':
        return 'Mother'
    else:
        return 'Not Mother'
# feature mother for each person
titanic['Mother'] = titanic.apply(mother, axis=1)

titanic.Mother.value_counts()
# Added in 2 new features
titanic.head(3)
titanic.isnull().sum() / len(titanic)
#Data Imputation: Age
# Plot Distribution of Age (Missing)
plt.subplot(1, 2, 1)
titanic['Age'].hist(bins=20, figsize=(15,6), edgecolor='white')
plt.xlabel('Age', fontsize=12)
plt.title('Distribution of Age (Missing)', fontsize=18)

# Plot Distribution of Age (Imputed Mean)
plt.subplot(1, 2, 2)
mean_age = pd.DataFrame(titanic['Age'].fillna(titanic.Age.mean()))
mean_age['Age'].hist(bins=20, figsize=(15,6), edgecolor='white', color='r')
plt.xlabel('Age)', fontsize=12)
plt.title('Distribution of Age (Imputed Mean)', fontsize=18)

plt.show()    
 
    
    
    
#Data Imputation: Age by Passenger Title
titanic.groupby(['Title'])['Age'].median()
titanic.boxplot(column='Age',by='Title') #Mean Age is different per title
plt.ylabel('Age')
# Fill in the missing age with the median of their Titles
titanic['Age'].fillna(titanic.groupby(["Title"])["Age"].transform(np.median),inplace=True)
#Plot the Age Distribution
titanic['Age'].hist(bins=20, figsize=(15,6), edgecolor='white')
plt.xlabel('Age', fontsize=12)
plt.title('Distribution of Age (Imputed by Title)', fontsize=18)
#Data Imputation: Embarked
titanic.Embarked.value_counts() 
#Impute missing 'Embarked' variable with the most frequent value: (S)
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].value_counts().index[0])
#Data Imputation: Cabin
titanic.Cabin.isnull().sum() / len(titanic)
#Drop Cabin Feature
titanic.drop(columns=['Cabin'], inplace=True)
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1
titanic.FamilySize.value_counts()
titanic.FamilySize.hist()
plt.title('FamilySize Distribution for Titanic Passengers', size=15)
plt.xlabel("Family Size")
plt.ylabel('Frequency')
titanic['IsAlone'] = titanic['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
titanic.IsAlone.value_counts()
titanic.IsAlone.hist()
plt.title('IsAlone Distribution for Titanic Passengers', size=15)
plt.xlabel("IsAlone")
plt.ylabel('Frequency')
# Examine the Age Distribution
#Binning helps solve the skewness problem.
titanic.Age.hist(bins=25)
plt.title('Age Distribution for Titanic Passengers', size=15)
plt.xlabel("Age")
plt.ylabel('Frequency')
# Fixed Width Binning (Kid, Teen, Adult, Elderly)
#With domain knowledge, we can safely bin our passengers into different age groups.
bins = [0,12,17,60,150]
labels = ["kid","teen","adult","elderly"]
titanic['AgeGroup'] = pd.cut(titanic.Age,bins=bins,labels=labels)
titanic[['Age','AgeGroup']].head(10)
#Quantile Binning uses the quantiles of our feature
titanic.Age.describe()
#Create a quantile list
quantile_list = [0, .25, .5, .75, 1.]
quantiles = titanic['Age'].quantile(quantile_list)
quantiles
titanic['age_quantile_range'] = pd.qcut(titanic.Age, 4)
titanic['age_quantile_label'] = pd.qcut(titanic.Age, 4, labels=[0.25, 0.5, 0.75, 1])
titanic[['Age','age_quantile_range','age_quantile_label']].head()
# Plot Fare Price Distribution
plt.subplot(1, 2, 1)
(titanic['Fare']).plot.hist(bins=15, figsize=(15, 6), edgecolor = 'white')
plt.xlabel('Fare Price', fontsize=12)
plt.title('BEFORE', fontsize=24)

#Plot Log Fare Price Distribution
plt.subplot(1, 2, 2)
np.log(titanic['Fare']+1).plot.hist(bins=15,figsize=(15,6), edgecolor='white', color='r')
plt.xlabel('log(Fare Price+1)', fontsize=12)
plt.title('AFTER', fontsize=24)

plt.show()
import sklearn.preprocessing as preproc
df_scale = titanic[['Fare']]
#Min-Max Scaling
df_scale['Min-Max'] = preproc.minmax_scale(titanic[['Fare']])
#Standardization
df_scale['Standardization'] = preproc.StandardScaler().fit_transform(titanic[['Fare']])
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
fig.tight_layout()

# Plot Original Price
df_scale['Fare'].hist(ax=ax1, bins=50)
ax1.tick_params(labelsize=14)
ax1.set_xlabel("Original Price", fontsize=10)
ax1.set_ylabel("Frequency", fontsize=14)

# Plot Min-Max Scaling on Price
df_scale['Min-Max'].hist(ax=ax2, bins=50, color='r')
ax2.tick_params(labelsize=14)
ax2.set_xlabel("Min-Max Price", fontsize=10)

# Plot Standardized Scaling on Price
df_scale['Standardization'].hist(ax=ax3, bins=50, color='g')
ax3.tick_params(labelsize=14)
ax3.set_xlabel("Standarized Price", fontsize=10)
titanic_cat = titanic.select_dtypes(include=['object','category'])
titanic_cat.head()
#Encode our 'AgeGroup' Category
age_group = np.unique(titanic_cat['AgeGroup'])
age_group
#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
genre_labels = le.fit_transform(titanic['AgeGroup'])
titanic_cat['AgeGroup_LE'] = genre_labels
titanic_cat[['AgeGroup','AgeGroup_LE']].head(20)
# Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[['kid','teen','adult','elderly']])
X = [['adult'], ['teen'], ['kid'], ['elderly'], ['adult']]
AgeGroup_OE = enc.fit_transform(X)
AgeGroup_OE
#Get the original data back
age_ord_map = {'kid': 1, 'teen': 2, 'adult': 3, 'elderly': 4}
titanic_cat['AgeGroup_OE'] = titanic_cat['AgeGroup'].map(age_ord_map)
titanic_cat[['AgeGroup','AgeGroup_OE']].head(20)
age = pd.DataFrame(['Kid','Teen','Adult','Elderly'], columns=['AgeGroup'])
age_dummy_features = pd.get_dummies(age['AgeGroup'])
pd.concat([age, age_dummy_features], axis=1)
#Apply Dummy Encoding to 'AgeGroup' feature
titanic_dummyage = pd.get_dummies(titanic_cat['AgeGroup'])
pd.concat([titanic_cat, titanic_dummyage], axis=1)