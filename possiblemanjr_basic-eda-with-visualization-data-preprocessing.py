import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline
train = pd.read_csv('/kaggle/input/kakr-4th-competition/train.csv')

test = pd.read_csv('/kaggle/input/kakr-4th-competition/test.csv')

sample_submission = pd.read_csv('/kaggle/input/kakr-4th-competition/sample_submission.csv')
train.shape, test.shape, sample_submission.shape
train.info()
for col, values in train.iteritems():

    num_uniques = values.nunique()

    print ('{name}: {num_unique}'.format(name=col, num_unique=num_uniques))

    print (values.unique())

    print ('\n')
train.describe(include='O')
### for visualization

train['income_cat'] = train.income.map({'>50K': 1, '<=50K': 0})
f,ax=plt.subplots(1,2,figsize=(18,8))

train['income'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('income')

ax[0].set_ylabel('')

sns.countplot('income',data=train,ax=ax[1])

ax[1].set_title('income')

plt.show()
train['workclass'].value_counts()
train.groupby(['workclass','income'])['income'].count()
f,ax=plt.subplots(1,2,figsize=(20,8))

train[['workclass','income_cat']].groupby(['workclass']).mean().plot.bar(ax=ax[0])

ax[0].set_title('income_cat vs workclass')



chart = sns.countplot(x = 'workclass',hue='income_cat',data=train, ax=ax[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,

                  horizontalalignment='right',

                  fontweight='light',

                  fontsize='large')

ax[1].set_title('workclass : High Income vs Low Income')

plt.show()
pd.crosstab(train.workclass,train.income,margins=True).style.background_gradient(cmap='summer_r')
train['education'].value_counts()
train.groupby(['education','income'])['income'].count()
f,ax=plt.subplots(1,2,figsize=(20,8))

train[['education','income_cat']].groupby(['education']).mean().plot.bar(ax=ax[0])

ax[0].set_title('income_cat vs education')



chart = sns.countplot(x = 'education',hue='income_cat',data=train, ax=ax[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,

                  horizontalalignment='right',

                  fontweight='light',

                  fontsize='large')

ax[1].set_title('education : High Income vs Low Income')

plt.show()
pd.crosstab(train.education,train.income,margins=True).style.background_gradient(cmap='summer_r')
train['marital_status'].value_counts()
train.groupby(['marital_status','income'])['income'].count()
f,ax=plt.subplots(1,2,figsize=(20,8))

train[['marital_status','income_cat']].groupby(['marital_status']).mean().plot.bar(ax=ax[0])

ax[0].set_title('income_cat vs marital_status')



chart = sns.countplot(x = 'marital_status',hue='income_cat',data=train, ax=ax[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,

                  horizontalalignment='right',

                  fontweight='light',

                  fontsize='large')

ax[1].set_title('marital_status : High Income vs Low Income')

plt.show()
pd.crosstab(train.marital_status,train.income,margins=True).style.background_gradient(cmap='summer_r')



train['occupation'].value_counts()

train.groupby(['occupation','income'])['income'].count()
f,ax=plt.subplots(1,2,figsize=(20,8))

train[['occupation','income_cat']].groupby(['occupation']).mean().plot.bar(ax=ax[0])

ax[0].set_title('income_cat vs occupation')



chart = sns.countplot(x = 'occupation',hue='income_cat',data=train, ax=ax[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,

                  horizontalalignment='right',

                  fontweight='light',

                  fontsize='large')

ax[1].set_title('occupation : High Income vs Low Income')

plt.show()
pd.crosstab(train.occupation,train.income,margins=True).style.background_gradient(cmap='summer_r')
train['relationship'].value_counts()
train.groupby(['relationship','income'])['income'].count()
f,ax=plt.subplots(1,2,figsize=(20,8))

train[['relationship','income_cat']].groupby(['relationship']).mean().plot.bar(ax=ax[0])

ax[0].set_title('income_cat vs relationship')



chart = sns.countplot(x = 'relationship',hue='income_cat',data=train, ax=ax[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,

                  horizontalalignment='right',

                  fontweight='light',

                  fontsize='large')

ax[1].set_title('relationship : High Income vs Low Income')

plt.show()
pd.crosstab(train.relationship,train.income,margins=True).style.background_gradient(cmap='summer_r')
train['race'].value_counts()

train.groupby(['race','income'])['income'].count()


f,ax=plt.subplots(1,2,figsize=(20,8))

train[['race','income_cat']].groupby(['race']).mean().plot.bar(ax=ax[0])

ax[0].set_title('income_cat vs race')



chart = sns.countplot(x = 'race',hue='income_cat',data=train, ax=ax[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,

                  horizontalalignment='right',

                  fontweight='light',

                  fontsize='large')

ax[1].set_title('race : High Income vs Low Income')

plt.show()

pd.crosstab(train.race,train.income,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

train[['sex','income_cat']].groupby(['sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('High Income vs Low Income')

sns.countplot('sex',hue='income_cat',data=train,ax=ax[1])

ax[1].set_title('sex: High Income vs Low Income')

plt.show()
train['native_country'].value_counts()
train.groupby(['native_country','income'])['income'].count()
f,ax=plt.subplots(1,2,figsize=(20,8))

train[['native_country','income_cat']].groupby(['native_country']).mean().plot.bar(ax=ax[0])

ax[0].set_title('income_cat vs native_country')



chart = sns.countplot(x = 'native_country',hue='income_cat',data=train, ax=ax[1])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,

                  horizontalalignment='right',

                  fontweight='light',

                  fontsize='large')

ax[1].set_title('native_country : High Income vs Low Income')

plt.show()

pd.crosstab(train.native_country,train.income,margins=True).style.background_gradient(cmap='summer_r')

print('Highest age is:',train['age'].max(), 'years_old')

print('Lowest age is:',train['age'].min(), 'years_old')

print('Average age is:',train['age'].mean(), 'years_old')
age_high_zero_died = train[(train["age"] > 0) & (train["income_cat"] == 0)]

age_high_zero_surv = train[(train["age"] > 0) & (train["income_cat"] == 1)]





plt.figure(figsize=(20,8))





sns.distplot(age_high_zero_surv["age"], bins=24, color='g')

sns.distplot(age_high_zero_died["age"], bins=24, color='r')

plt.title("Distribuition and density by age",fontsize=20)

plt.xlabel("age",fontsize=15)

plt.ylabel("Distribuition High Income and Low Income",fontsize=15)

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("race","age", hue="income_cat", data=train,split=True,ax=ax[0])

ax[0].set_title('race and age vs income')

ax[0].set_yticks(range(0,100,10))

sns.violinplot("workclass","age", hue="income_cat", data=train,split=True,ax=ax[1])

ax[1].set_title('workclass and age vs income_cat')

ax[1].set_yticks(range(0,100,10))

plt.show()
print('Highest fnlwgt is:',train['fnlwgt'].max())

print('Lowest fnlwgt is:',train['fnlwgt'].min())

print('Average fnlwgt is:',train['fnlwgt'].mean())
fnlwgt_high_zero_died = train[(train["fnlwgt"] > 0) & (train["income_cat"] == 0)]

fnlwgt_high_zero_surv = train[(train["fnlwgt"] > 0) & (train["income_cat"] == 1)]





plt.figure(figsize=(20,8))





sns.distplot(fnlwgt_high_zero_died["fnlwgt"], bins=24, color='g')

sns.distplot(fnlwgt_high_zero_surv["fnlwgt"], bins=24, color='r')

plt.title("Distribuition and density by fnlwgt",fontsize=20)

plt.xlabel("fnlwgt",fontsize=15)

plt.ylabel("Distribuition High Income and Low Income",fontsize=15)

plt.show()
print('Highest capital_gain is:',train['capital_gain'].max() , '$')

print('Lowest capital_gain is:',train['capital_gain'].min(), '$')

print('Average capital_gain is:',train['capital_gain'].mean(), '$')
capital_gain_high_zero_died = train[(train["capital_gain"] > 0) & (train["income_cat"] == 0)]

capital_gain_high_zero_surv = train[(train["capital_gain"] > 0) & (train["income_cat"] == 1)]





plt.figure(figsize=(20,8))





sns.distplot(capital_gain_high_zero_died["capital_gain"], bins=24, color='g')

sns.distplot(capital_gain_high_zero_surv["capital_gain"], bins=24, color='r')

plt.title("Distribuition and density by capital_gain",fontsize=20)

plt.xlabel("capital_gain",fontsize=15)

plt.ylabel("Distribuition High Income and Low Income",fontsize=15)

plt.show()
print('Highest capital_loss is:',train['capital_loss'].max() , '$')

print('Lowest capital_loss is:',train['capital_loss'].min(), '$')

print('Average capital_loss is:',train['capital_loss'].mean(), '$')
capital_loss_high_zero_died = train[(train["capital_loss"] > 0) & (train["income_cat"] == 0)]

capital_loss_high_zero_surv = train[(train["capital_loss"] > 0) & (train["income_cat"] == 1)]





plt.figure(figsize=(20,8))





sns.distplot(capital_loss_high_zero_died["capital_loss"], bins=24, color='g')

sns.distplot(capital_loss_high_zero_surv["capital_loss"], bins=24, color='r')

plt.title("Distribuition and density by capital_loss",fontsize=20)

plt.xlabel("capital_loss",fontsize=15)

plt.ylabel("Distribuition High Income and Low Income",fontsize=15)

plt.show()
print('Highest hours_per_week is:',train['hours_per_week'].max() , 'hours')

print('Lowest hours_per_week is:',train['hours_per_week'].min(), 'hours')

print('Average hours_per_week is:',train['hours_per_week'].mean(), 'hours')
hours_high_zero_died = train[(train["hours_per_week"] > 0) & (train["income_cat"] == 0)]

hours_high_zero_surv = train[(train["hours_per_week"] > 0) & (train["income_cat"] == 1)]





plt.figure(figsize=(20,8))



sns.distplot(capital_loss_high_zero_died["hours_per_week"], bins=24, color='g')

sns.distplot(capital_loss_high_zero_surv["hours_per_week"], bins=24, color='r')

plt.title("Distribuition and density by hours_per_week",fontsize=20)

plt.xlabel("hours_per_week",fontsize=15)

plt.ylabel("Distribuition High Income and Low Income",fontsize=15)

plt.show()
train.info()
## Label Encoding

## Categorize category features



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for df in [train,test]:

    df['workclass'] = le.fit_transform(df['workclass'].astype(str))

    df['marital_status'] = le.fit_transform(df['marital_status'].astype(str))

    df['education'] = le.fit_transform(df['education'].astype(str))

    df['occupation'] = le.fit_transform(df['occupation'].astype(str))

    df['race'] = le.fit_transform(df['race'].astype(str))

    df['sex'] = le.fit_transform(df['sex'].astype(str))

    df['relationship'] = le.fit_transform(df['native_country'].astype(str))

    df['native_country'] = le.fit_transform(df['native_country'].astype(str))

    

#   cat_features = ['workclass','marital_status','education','occupation','race','sex','relationship','native_country']





# for i in enumerate (cat_features):

#    ca = i[1]

#    train[ca] = train[ca].astype('category')

train['income'] = train.income.map({'>50K': 1, '<=50K': 0})
for df in [train, test]:

    df.drop('education_num', axis= 1 , inplace  = True)
train.drop('income_cat', axis= 1 , inplace  = True)
train.info()
########## Export

train.to_pickle('train.pkl')

test.to_pickle('test.pkl')