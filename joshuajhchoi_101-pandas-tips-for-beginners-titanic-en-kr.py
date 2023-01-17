import numpy as np

import pandas as pd



test = pd.read_csv('../input/titanic/test.csv')

train = pd.read_csv('../input/titanic/train.csv')
train.head()
train.head(10)
train
train.tail()
train.tail(10)
train[101:111]
train.describe()
train.info()
train.shape
train.dtypes
len(train)
train.columns
train.columns[3]
train.columns[3:5]
train['Name']
train.isnull().any()
train.isna().any()
train.isnull().sum()
train.isna().sum()
train.loc[9]
train['Name']
train.loc[train.Age.values < 1]
train.loc[9,['Name']]
train.iloc[11:21]
train.iloc[21]
pd.set_option("display.max_columns", 8)

train.head()
train.sort_values('Fare', ascending=False)
train.sort_values('Fare', ascending=True)
train.sort_values('Fare', ascending=True)[101:105]
train.sort_values(['Fare', 'Survived', 'Pclass'], ascending=[False, False, False])
train[train['Fare'] > 80]
train.sort_values(by=['Pclass'], axis=0)

train.reindex(sorted(train.columns), axis=1)
train.sort_index(axis=1)
train[(train.Age >= 30) & (train.Sex == 'female')]
train[train.Embarked.isin(['C'])]
train[train.Cabin.isin(['C50', 'C85'])]
train[train.Ticket.isin(['113572']) & train.Parch.isin([0])]
(train.Fare == 80.0).sum()
(train.Fare > 80.0).sum()
(train.Fare > 80.0).any()
(train.Fare > 80.0).mean()
sample = pd.DataFrame()
temp = train.copy()



temp.head()
temp['New']  = 0



temp.head()
temp['New']  = temp['Age']



temp.head()
y_train = train['Survived'] 



y_train.head()
sample1 = train[0:3] 



sample1.head()
sample2 = train[['Pclass', 'Name', 'Sex']]  # Make sure there are two square backets 대괄호 두 개 임을 명심하세요.



sample2.head()
sample3 = train[['Pclass', 'Name', 'Sex']] [1:3]



sample3.head()
sample4 = train.loc[(train.Survived.values == 1 )]



sample4.head()
sample5 = train.loc[((train.Survived.values == 1 ) & (train.Pclass.values == 3 ) & (train.Sex.values == 'male' ))]



sample5.head()
data = pd.concat((train, test),sort=True)



data.head()
for i in data:

    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')



data.head()
for i in data:

    data['Initial1']=data.Name.str.extract('([A-Za-z])\.')



data.head()
for i in data:

    data['LastName']=data.Name.str.extract('^([A-Za-z]+)')



data.head()
for i in data:

    data['Alphabet1']=data.Name.str.extract('^([A-Za-z])')



data.head()
data['Initick'] = data.Ticket.str.split()



data.head()
data['Aiphabets'] = data['Name'].apply(lambda x: [item for elem in [y.split() for y in x] for item in elem])



data.head()
data['FirstLastName'] = data['Name'].str.split(', ').str[::-1].str.join(' ')



data.head()
data['Initick'] = 0

for s in data:

    data['Initick']=temp.Ticket.str.extract('^([A-Za-z]+)') # And this will make "number only" tickets become null values.



data = data.reset_index(drop=True) # We neeed this line otherwise we will get "cannot reindex from a duplicate axis"error  

    

for s in data:

    data.loc[(data.Initick.isnull()),'Initick']= data['Ticket']



data.head()
train['Age_Range']=pd.qcut(train['Age'],8)



train.head()
train['Age_Range'].value_counts()
train['Age_Cut']=pd.cut(train['Age'],8)



train.head()
train['Age_Cut'].value_counts()
train.Parch.value_counts()
top3=train.Parch.value_counts().nlargest(3).index

print (top3)
data['NewParch'] = 3 # let something more than 3 be 3 (others)



data.loc[(data.Parch.values == 0),'NewParch']= 0

data.loc[(data.Parch.values == 1),'NewParch']= 1

data.loc[(data.Parch.values == 2),'NewParch']= 2



data[11:30]
frequencies = data["SibSp"].value_counts(normalize = True)

frequencies
threshold = 0.015

small_categories = frequencies[frequencies < threshold].index

small_categories
data['NewSibSp'] = 5 # let something more than 5 be 5 (others)



data.loc[(data.SibSp.values == 0),'NewSibSp']= 0

data.loc[(data.SibSp.values == 1),'NewSibSp']= 1

data.loc[(data.SibSp.values == 2),'NewSibSp']= 2

data.loc[(data.SibSp.values == 3),'NewSibSp']= 3

data.loc[(data.SibSp.values == 4),'NewSibSp']= 4



data[11:30]
train.groupby(['Pclass','Parch'])['Pclass'].count()
train.groupby(['Pclass','Age'])['Survived'].mean()
train.groupby(['Pclass','Survived'])['Pclass'].count().to_frame().style.background_gradient(cmap='summer_r')
train.groupby(['Pclass'])['Age'].mean().to_frame().style.background_gradient(cmap='summer_r')
pd.crosstab(train.Survived,train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
data.Age.value_counts()
top5 = train.Age.value_counts().nlargest(5).index

top5
data['Agroup'] = 1



data.loc[(data.Age.values < 24.0),'Agroup']= 0

data.loc[(data.Age.values > 30.0),'Agroup']= 2



data.head()
data['Age_Cuts'] = pd.cut(data.Age, bins=[0, 1, 3, 10, 18, 65, 99], labels=['Baby', 'Todler', 'Kid', 'Teens',  'adult', 'elderly'])



data.head()
data.insert(3, 'New1', 0)



data.head()
data.loc[(data['Initial']=='Dona')]
data.loc[(data['Initial']=='Dona'),'Initial']= 'Mrs'
data.groupby('Initial')['Age'].mean()
data['Age'] = data.groupby('Initial')['Age'].apply(lambda x: x.fillna(x.mean()))



data[31:50]
data.Sex.values
data['Gender12'] = data.Sex.map({'male': 1, 'female':2})



data.head()
data['Gender']= data['Sex']



for n in range(1,4):

  data.loc[(data['Sex'] == 'male') & (data['Pclass'] == n),'Gender']= 'm'+str(n)

  data.loc[(data['Sex'] == 'female') & (data['Pclass'] == n),'Gender']= 'w'+str(n)



data.loc[(data['Gender'] == 'm3'),'Gender']= 'm2'

data.loc[(data['Gender'] == 'w3'),'Gender']= 'w2'

data.loc[(data['Age'] <= 1.0),'Gender']= 'baby'

data.loc[(data['Age'] > 75.0),'Gender']= 'old'



data.head()
data['NumName']= 0



for i in temp:

    data['NumName'] = data['LastName'].factorize()[0]

    

data.head()
temp = pd.DataFrame()



def change(a):

    temp[a] = data[a]



change('Age')

change('Gender12')

change('Initial')

change('Pclass')



temp.head()
temp.columns = ['Age', 'Sex', 'Name', 'Pclass']



temp.head()
temp.columns = temp.columns.str.lower()



temp.head()
temp['longname']=data['Name']



temp.head()
temp['longname']= temp.longname.str.replace(' ', '_')



temp.head()
data.Ticket.dtype



data.astype({'Age':'int'})
data.Gender.value_counts()
data['Gender'].replace(['m2','w2', 'm1', 'w1', 'baby', 'old'],[1, 2, 3, 4, 5, 6 ],inplace=True)



data.Gender.value_counts()
data.columns
data.drop('Gender12' , axis=1)



data.head() # It will display the dropped columns but they have been dropped / 실제로 drop되어도 보임
a = data.groupby('Age_Cuts')



a.get_group('adult')
import pandas as pd

import numpy as np



# For Visualization / 시각화

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('seaborn-whitegrid')

import missingno



import missingno

missingno.matrix(data, figsize = (15,8))
corr = data.corr()

# 마스크 셋업

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# 그래프 셋업

plt.figure(figsize=(14, 8))

# 그래프 타이틀

plt.title('Overall Correlation of Titanic Features', fontsize=18)

#  Co-relation 매트릭스 런칭

sns.heatmap(corr, mask=mask, annot=False,cmap='RdYlGn', linewidths=0.2, annot_kws={'size':20})

plt.show()
f,ax=plt.subplots(1,2,figsize=(15,6))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
def bag(a,b,c,d):

  f,ax=plt.subplots(1,2,figsize=(20,8))

  train[[a,b]].groupby([a]).mean().plot.bar(ax=ax[0])

  ax[0].set_title(c)

  sns.countplot(a,hue=b,data=train,ax=ax[1])

  ax[1].set_title(d)

  plt.show()



bag('Sex','Survived','Survived per Sex','Sex Survived vs Not Survived')  
def survpct(a):

  return data.groupby(a).Survived.mean()



survpct('Initial')
print('Oldest Passenger was ',data['Age'].max(),'Years')

print('Youngest Passenger was ',data['Age'].min(),'Years')

print('Average Age on the ship was ',int(data['Age'].mean()),'Years')
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
data.columns
data['Age_Range']=pd.qcut(data['Age'],10)

def groupmean(a,b):

  return data.groupby([a])[b].mean().to_frame().style.background_gradient(cmap='summer_r')



groupmean('Age_Range', 'Age')
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])

ax[0].set_title('Fares in Pclass 1')

sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])

ax[1].set_title('Fares in Pclass 2')

sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()
sns.factorplot('Embarked','Survived',data=data)

fig=plt.gcf()

fig.set_size_inches(5,3)

plt.show()
f,ax=plt.subplots(2,2,figsize=(20,15))

sns.countplot('Embarked',data=data,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
data.reset_index().rename(columns={"Initial": "Salutation"})

X_test = test

X_train = train
ntrain = train.shape[0]

ntest = test.shape[0]
temp.head()
df = pd.DataFrame()

df['Age'] = temp['age']

df['Sex'] =  temp['sex']

df['Pclass'] = temp['pclass']
from sklearn.preprocessing import LabelEncoder



df_enc = df.apply(LabelEncoder().fit_transform)

                          

df_enc.head()
from sklearn.preprocessing import OneHotEncoder



one_hot_cols = df.columns.tolist()

df_enc = pd.get_dummies(df, columns=one_hot_cols)



df_enc.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_train = scaler.fit_transform(df_enc)

X_test = scaler.transform(df_enc)
# This one make an error but when you are all done using the following format to make a final submission form



# predictions = grid_soft.predict(X_test)