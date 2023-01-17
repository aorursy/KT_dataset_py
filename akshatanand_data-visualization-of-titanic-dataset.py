#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("../input/titanicdataset-traincsv/train.csv")
import warnings
warnings.filterwarnings('ignore')
train.info()
train.head()
train.describe()
train.isnull()
###Who were the passengers on the titanic? (What age, gender, class etc)

###Gender Plot
sns.factorplot('Sex',data=train,kind='count')

### Shows more male passengers than female 
### Class plot
sns.factorplot('Pclass',data=train,kind='count')
## More passengers are from class Three. Now lets find the gender ration among the classes

sns.factorplot('Pclass',data=train,hue='Sex',kind='count')

lm = sns.lmplot(x = 'Age', y = 'Fare', data = train, hue = 'Sex', fit_reg=False)


lm.set(title = 'Fare & Age')


axes = lm.axes
axes[0,0].set_ylim(-5,)
axes[0,0].set_xlim(-5,85)

df = train.Fare.sort_values(ascending = False)
df


binsVal = np.arange(0,600,10)
binsVal


plt.hist(df, bins = binsVal)


plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Fare Payed Histrogram')

plt.show()
train.Survived.sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)

train['Age'].hist(bins=30,color='darkred',alpha=0.3)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30,color='green')
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
train.head()
