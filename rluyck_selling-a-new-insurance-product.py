import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

sns.set_context('notebook')

%matplotlib inline
df = pd.read_csv('/kaggle/input/insurance-company/Customer_data.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df['TARGET'] = df['TARGET'].map({'Y': 1, 'N': 0})
ax = sns.countplot(x="TARGET", data=df)

plt.title('TARGET')



total = len(df['TARGET'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
df_X = df.drop('TARGET', axis=1)
cm = df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(cm, annot=True, cmap='coolwarm')
cm_X = (df_X).corr()

plt.figure(figsize=(12, 8))

sns.heatmap(cm_X, annot=True, cmap='coolwarm')
# https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant



X_ = add_constant(df)
pd.Series([variance_inflation_factor(X_.values, i)

           for i in range(X_.shape[1])],

          index=X_.columns)



# VIF of 5 or 10 and above indicates a multicollinearity problem.

# If there is perfect correlation, then VIF = infinity.
# Distribution

plt.figure(figsize=(8, 5))

sns.distplot(df['ID'])
df['ID'].value_counts().head(5)
df[df['ID'].duplicated(keep=False)]
print(str(np.round((3008/len(df))*100, decimals=3)) +

      '% of the samples have a duplicate ID')
df[df['ID'] == 306]
df['turnover_A'].nunique()
# Double check: 11008 unique values and we have 14016 samples

14016-11008

# Conclussion: 3008 duplicate records in the dataset
df.shape
# remove duplicate records

# there is no timestamp record, so does not matter which of the two duplicate id rows I remove

df = df.drop_duplicates(subset=['ID', 'turnover_A'], keep='first')
df.shape
ax = sns.countplot(x="TARGET", data=df)

plt.title('TARGET')



total = len(df['TARGET'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
plt.figure(figsize=(8, 5))

ax = sns.countplot(x="loyalty", data=df)

plt.title('loyalty')



total = len(df['loyalty'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
# Correlation

plt.figure(figsize=(8, 5))

cm = df.corr()

cm.nlargest(14, 'loyalty')['loyalty'].plot(kind='bar')
sns.heatmap((df[['loyalty', 'LOR']]).corr(), annot=True)
# 99 = unclassified (beige lines)

plt.figure(figsize=(8, 5))

sns.heatmap(df[['loyalty']])
classified_loyalty = df[df['loyalty'] < 99]
plt.figure(figsize=(14, 5))

ax = sns.countplot(x='LOR', hue='loyalty',data=classified_loyalty)

plt.title('LOR per classified loyalty')

plt.show()
plt.figure(figsize=(14, 5))

ax = sns.countplot(x="LOR", data=classified_loyalty)

plt.title('LOR for classified loyalty <99')



total = len(classified_loyalty['LOR'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
unclassified_loyalty = df[df['loyalty'] == 99]
plt.figure(figsize=(15, 5))

ax = sns.countplot(x="LOR", data=unclassified_loyalty)

plt.title('LOR for loyalty 99')



total = len(unclassified_loyalty['LOR'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
df.shape
df = df.drop('loyalty', axis=1)
df.shape
plt.figure(figsize=(8, 6))

sns.distplot(df['ID'])

plt.title('ID distribution')

# After removing duplicate ID's, majority still has low ID
df_check_under = df[(df['ID'] < 20000)]

print(len(df_check_under)/len(df))
df_check_above = df[(df['ID'] > 20000)]

print(len(df_check_above)/len(df))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(

    2, 2, figsize=(12, 8), sharey=True,sharex=True)



sns.distplot(df_check_under[df_check_under['TARGET'] == 0]

             ['ID'], ax=ax1).set_title('ID = <20000; target = 0')

sns.distplot(df_check_under[df_check_under['TARGET'] == 1]

             ['ID'], ax=ax2,color='orange').set_title('ID = <20000; target = 1')

sns.distplot(df_check_above[df_check_above['TARGET'] == 0]

             ['ID'], ax=ax3).set_title('ID = >20000; target = 0')

sns.distplot(df_check_above[df_check_above['TARGET'] == 1]

             ['ID'], ax=ax4,color='orange').set_title('ID = >20000; target = 1')

fig.tight_layout()
fig = plt.figure(figsize=(14, 5))

sns.distplot(df['age'],bins=80)

plt.xticks(np.arange(min(df['age']), max(df['age'])+1, 5))

plt.title('Raw data age distribution')

plt.show()
import scipy.stats as sp



print("Mode: "+str(df['age'].mode()[0]))

print("Median: "+str(df['age'].median()))

print("Mean: "+str(np.round(df['age'].mean(), decimals=2)))

print("Skew: "+str(np.round(sp.skew(df['age']), decimals=2)))

print("Kurtosis: "+str(np.round(sp.kurtosis(df['age']), decimals=2)))

print("Min: "+str(df['age'].min()))

print("Max: "+str(df['age'].max()))

print("Range: "+str((df['age'].max())-(df['age'].min())))
age_log = np.log(df['age'])
fig = plt.figure(figsize=(14, 5))

sns.distplot(age_log,bins=80)

plt.xticks(np.arange(min(age_log), max(age_log)+1, 1))

plt.title('Log of age distribution')

plt.show()
fig = plt.figure(figsize=(16, 6))

sns_plot = sns.distplot(df['age'], hist=False, rug=True).set_title('Age')

sns_plot = sns.distplot(df[df['TARGET'] == 0]['age'], hist=False, rug=True)

sns_plot = sns.distplot(df[df['TARGET'] == 1]['age'], hist=False, rug=True)

fig.legend(labels=['Combined',

                   'TARGET = 0',

                   'TARGET = 1'])

plt.title('Raw age distribution per target value and combined ')

plt.xticks(np.arange(min(df['age']), max(df['age'])+1, 2))

plt.show()

fig.tight_layout()
fig = plt.figure(figsize=(16, 6))

sns_plot = sns.distplot(df[df['TARGET'] == 0]['age'], hist=False, rug=True)

sns_plot = sns.distplot(df[df['TARGET'] == 1]['age'], hist=False, rug=True)

fig.legend(labels=['TARGET = 0',

                   'TARGET = 1'])

plt.title('Age distribution per target value after removing outliers')

plt.xticks(np.arange(min(df['age']), max(df['age'])+1, 2))

plt.show()
plt.figure(figsize=(12, 6))

ax = sns.boxplot(x="age", y="TARGET", data=df, orient="h")
df.shape
df = df.drop('city', axis=1)
df.shape
plt.figure(figsize=(14, 5))

ax = sns.countplot(df['LOR'])

plt.title('Length of relationship in years distribution')



total = len(df['LOR'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
plt.figure(figsize=(14, 5))

ax = sns.countplot(df['lor_M'])

plt.title('Length of relationship in months distribution')



total = len(df['lor_M'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
df.shape
df = df.drop(['lor_M'], axis=1)
df.shape
plt.figure(figsize=(14, 5))

ax = sns.countplot(x="prod_A", data=df)

plt.title('Distribution of  prod_A')



total = len(df['prod_A'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
fig = plt.figure(figsize=(14, 5))

sns_plot = sns.distplot(df[df['prod_A'] == 0]['age'], hist=False, rug=True)

sns_plot = sns.distplot(df[df['prod_A'] == 1]['age'], hist=False, rug=True)

fig.legend(labels=['prod_A_0',

                   'prod_A_1'])

plt.xticks(np.arange(min(df['age']), max(df['age'])+1, 2))

plt.show()
plt.figure(figsize=(14, 5))

sns.boxplot(x="age", y="prod_A", data=df, orient="h")
plt.figure(figsize=(14, 5))

ax = sns.countplot(x="type_A", data=df)

plt.title('Distribution of  type_A')



total = len(df['type_A'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
df.shape
df = df.drop('type_A', axis=1)
df.shape
plt.figure(figsize=(14, 5))

ax = sns.countplot(x="prod_B", data=df)

plt.title('Distribution of  prod_B')



total = len(df['prod_B'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
plt.figure(figsize=(14, 5))

ax = sns.countplot(x="type_B", data=df)

plt.title('Distribution of  type_B')



total = len(df['type_B'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
df.shape
# holds quasi the same distribution as prod_B, remove

df = df.drop('type_B', axis=1)
df.shape
plt.figure(figsize=(14, 5))

sns.distplot(df['turnover_A'])
print("Mode: "+str(np.round(df['turnover_A'].mode()[0], decimals=2)))

print("Median: "+str(np.round(df['turnover_A'].median(), decimals=2)))

print("Mean: "+str(np.round(df['turnover_A'].mean(), decimals=2)))

print("Skew: "+str(np.round(sp.skew(df['turnover_A']), decimals=2)))

print("Kurtosis: "+str(np.round(sp.kurtosis(df['turnover_A']), decimals=2)))

print("Min: "+str(np.round(df['turnover_A'].min(), decimals=2)))

print("Max: "+str(np.round(df['turnover_A'].max(), decimals=2)))
df['turnover_A'].describe()
# plot 40 highest values to check if valid outliers

plt.figure(figsize=(14, 5))

df['turnover_A'].sort_values(ascending=False).head(40).plot(kind='bar')

plt.xlabel('index')

plt.ylabel('turnover_A')

plt.title('Inspect highest turnover_A values')

plt.show()
df_turnover_A_reduced_plot = df.drop(df[df['turnover_A'] > 400].index)
print("Mode: "+str(np.round(df_turnover_A_reduced_plot['turnover_A'].mode()[0], decimals=2)))

print("Median: "+str(np.round(df_turnover_A_reduced_plot['turnover_A'].median(), decimals=2)))

print("Mean: "+str(np.round(df_turnover_A_reduced_plot['turnover_A'].mean(), decimals=2)))

print("Skew: "+str(np.round(sp.skew(df_turnover_A_reduced_plot['turnover_A']), decimals=2)))

print("Kurtosis: "+str(np.round(sp.kurtosis(df_turnover_A_reduced_plot['turnover_A']), decimals=2)))

print("Min: "+str(np.round(df_turnover_A_reduced_plot['turnover_A'].min(), decimals=2)))

print("Max: "+str(np.round(df_turnover_A_reduced_plot['turnover_A'].max(), decimals=2)))
df_turnover_A_reduced_plot['turnover_A'].describe()
plt.figure(figsize=(14, 5))

sns.distplot(df_turnover_A_reduced_plot['turnover_A'])

plt.show()
df = df.drop(df[df['turnover_A'] > 400].index)
plt.figure(figsize=(14, 5))

sns.distplot(df['turnover_B'])

plt.show()
print("Mode: "+str(np.round(df['turnover_B'].mode()[0], decimals=2)))

print("Median: "+str(np.round(df['turnover_B'].median(), decimals=2)))

print("Mean: "+str(np.round(df['turnover_B'].mean(), decimals=2)))

print("Skew: "+str(np.round(sp.skew(df['turnover_B']), decimals=2)))

print("Kurtosis: "+str(np.round(sp.kurtosis(df['turnover_B']), decimals=2)))

print("Min: "+str(np.round(df['turnover_B'].min(), decimals=2)))

print("Max: "+str(np.round(df['turnover_B'].max(), decimals=2)))
df['turnover_B'].describe()
# plot 40 highest values to check if outliers are valid

plt.figure(figsize=(14, 5))

df['turnover_B'].sort_values(ascending=False).head(40).plot(kind='bar')

plt.xlabel('index')

plt.ylabel('turnover_B')

plt.title('Inspect highest turnover_B values')

plt.show()
df_TO_A_B_reduced_plot = df_turnover_A_reduced_plot.drop(

    df_turnover_A_reduced_plot[df_turnover_A_reduced_plot['turnover_B'] >= 260].index)
print("Mode: "+str(np.round(df_TO_A_B_reduced_plot['turnover_B'].mode()[0], decimals=2)))

print("Median: "+str(np.round(df_TO_A_B_reduced_plot['turnover_B'].median(), decimals=2)))

print("Mean: "+str(np.round(df_TO_A_B_reduced_plot['turnover_B'].mean(), decimals=2)))

print("Skew: "+str(np.round(sp.skew(df_TO_A_B_reduced_plot['turnover_B']), decimals=2)))

print("Kurtosis: "+str(np.round(sp.kurtosis(df_TO_A_B_reduced_plot['turnover_B']), decimals=2)))

print("Min: "+str(np.round(df_TO_A_B_reduced_plot['turnover_B'].min(), decimals=2)))

print("Max: "+str(np.round(df_TO_A_B_reduced_plot['turnover_B'].max(), decimals=2)))
df_TO_A_B_reduced_plot['turnover_B'].describe()
fig = plt.figure(figsize=(14, 5))

sns_plot = sns.distplot(df_TO_A_B_reduced_plot['turnover_A'])

sns_plot = sns.distplot(df_TO_A_B_reduced_plot['turnover_B'])

fig.legend(labels=['turnover_A', 'turnover_B'])

plt.xlabel('Turnover_B and Turnover_A')

plt.show()

fig.tight_layout()
df.shape
df_TO_A_B_reduced_plot.shape
df = df.drop(df[df['turnover_B'] >= 260].index)
df.shape
df.shape
df['contract'].value_counts()
df = df.drop(['contract'], axis=1)
df.shape
sns.heatmap(df[['age_P', 'age']].corr(), annot=True)
df.shape
df = df.drop('age_P', axis=1)
df.shape
g = sns.PairGrid(df, hue='TARGET', corner=True)

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)

g = g.add_legend()
df.head()
target_zero = df[df['TARGET'] == 0]

target_one = df[df['TARGET'] == 1]
fig = plt.figure(figsize=(14, 4))

ax = sns.countplot(x="TARGET", data=df)

total = len(df['TARGET'])



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 3,

            '{:1.3f}%'.format(100*(height/total)), ha="center")
fig = plt.figure(figsize=(14, 5))

sns.distplot(target_one['age'], color='orange')

sns.distplot(target_zero['age'], color='blue')

fig.legend(labels=['TARGET = 1', 'TARGET = 0'])

plt.show()

fig.tight_layout()
fig = plt.figure(figsize=(14, 4))

ax = sns.boxplot(x="age", y="TARGET", data=df, orient="h")
fig = plt.figure(figsize=(14, 4))

ax = df[df['TARGET'] == 1]['age'].value_counts().head(10).plot(kind='bar',color='orange')

total = len(df[df['TARGET'] == 1]['age'])



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 0.5,

            '{:1.3f}%'.format(100*(height/total)), ha="center")



plt.title('Top 10 ages buying the new product')

plt.xlabel('age')

plt.ylabel('count')

plt.show()
fig = plt.figure(figsize=(14, 4))

ax = df[df['TARGET'] == 0]['age'].value_counts().head(10).plot(kind='bar')

total = len(df[df['TARGET'] == 0]['age'])



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 1,

            '{:1.3f}%'.format(100*(height/total)), ha="center")



plt.title('Top 10 ages not buying the new product')

plt.xlabel('age')

plt.ylabel('count')

plt.show()
fig = plt.figure(figsize=(14, 4))

ax = sns.countplot(df['LOR'], hue=df['TARGET'])

total = len(df['LOR'])



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 3,

            '{:1.3f}%'.format(100*(height/total)), ha="center")



plt.title('Length of relationship vs TARGET')

plt.show()
fig = plt.figure(figsize=(14, 4))

ax = sns.countplot(df['TARGET'], hue=df['prod_B'])

total = len(df['TARGET'])



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 3,

            '{:1.3f}%'.format(100*(height/total)), ha="center")



plt.title('Prod_B vs TARGET')

plt.show()
fig = plt.figure(figsize=(14, 4))

sns.distplot(target_one['turnover_B'], color='orange')

sns.distplot(target_zero['turnover_B'], color='blue')

fig.legend(labels=['TARGET = 1', 'TARGET = 0'])



plt.title('Turnover_B distribution per target value')

plt.show()
fig = plt.figure(figsize=(14, 4))

ax = sns.boxplot(x="turnover_B", y="TARGET",

                 data=df, orient="h")

plt.title('turnover_B distribution vs TARGET')

plt.show()
fig = plt.figure(figsize=(14, 4))

ax = sns.countplot(df['TARGET'], hue=df['prod_A'])

total = len(df['TARGET'])



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 3,

            '{:1.3f}%'.format(100*(height/total)), ha="center")



plt.title('Prod_A vs TARGET')

plt.show()
fig = plt.figure(figsize=(14, 4))

sns.distplot(target_one['turnover_A'], color='orange')

sns.distplot(target_zero['turnover_A'], color='blue')

fig.legend(labels=['TARGET = 1', 'TARGET = 0'])

plt.title('Turnover_A distribution per target value')

plt.show()
fig = plt.figure(figsize=(14, 4))

ax = sns.boxplot(x="turnover_A", y="TARGET",

                 data=df, orient="h")



plt.title('Turnover_A distribution per target value')

plt.show()
fig = plt.figure(figsize=(12, 5))

sns.scatterplot(x='ID', y='age', data=df, hue='TARGET')

fig.tight_layout()

fig = plt.figure(figsize=(16, 4))

sns.distplot(target_one['ID'], color='orange')

sns.distplot(target_zero['ID'], color='blue')

fig.legend(labels=['TARGET = 1', 'TARGET = 0'])

plt.xticks(np.arange(min(df['ID'])-1, max(df['ID']), 10000))

plt.show()
df_lower_id = df[(df['ID'] < 20000)]

df_higher_id = df[(df['ID'] > 20000)]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(

    2, 2, figsize=(12, 8), sharey=True,sharex=True)



ax1 = sns.distplot(df_lower_id[df_lower_id['TARGET'] == 0]['ID'], ax=ax1).set_title(

    'ID = <20000; target = 0', fontweight="bold", size=15)

ax2 = sns.distplot(df_lower_id[df_lower_id['TARGET'] == 1]['ID'], color='orange', ax=ax2).set_title(

    'ID = <20000; target = 1', fontweight="bold", size=15)

ax3 = sns.distplot(df_higher_id[df_higher_id['TARGET'] == 0]['ID'], ax=ax3).set_title(

    'ID = >20000; target = 0', fontweight="bold", size=15)

ax4 = sns.distplot(df_higher_id[df_higher_id['TARGET'] == 1]['ID'], color='orange', ax=ax4).set_title(

    'ID = >20000; target = 1', fontweight="bold", size=15)

fig.tight_layout()
sns.jointplot(x='ID', y='age', data=df_lower_id,

              color='blue', kind='kde')
sns.jointplot(x='ID', y='age', data=df_higher_id,

              color='blue', kind='kde')
prod_A_zero = df[df['prod_A'] == 0]

prod_A_one = df[df['prod_A'] == 1]
fig = plt.figure(figsize=(14, 4))

ax = sns.countplot(df['prod_A'], hue=df['TARGET'])

total = len(df['prod_A'])



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 3,

            '{:1.3f}%'.format(100*(height/total)), ha="center")



plt.title('Prod_A vs TARGET')

plt.show()
fig = plt.figure(figsize=(14, 5))

sns.distplot(prod_A_one['age'], color='orange',bins=44)

sns.distplot(prod_A_zero['age'], color='blue',bins=44)

fig.legend(labels=['prod_A=1', 'prod_A=0'])

fig.tight_layout()
fig = plt.figure(figsize=(14, 5))

sns.distplot(prod_A_one['turnover_B'], color='orange')

sns.distplot(prod_A_zero['turnover_B'], color='blue')

fig.legend(labels=['prod_A=1', 'prod_A=0'])

fig.tight_layout()
fig = plt.figure(figsize=(14, 5))

sns.distplot(prod_A_one['turnover_A'], color='orange')

sns.distplot(prod_A_zero['turnover_A'], color='blue')

fig.legend(labels=['prod_A=1', 'prod_A=0'])

fig.tight_layout()
fig=plt.figure(figsize=(14, 5))

sns.distplot(prod_A_one['ID'], color='orange')

sns.distplot(prod_A_zero['ID'], color='blue')

fig.legend(labels=['prod_A=1', 'prod_A=0'])

fig.tight_layout()
fig = plt.figure(figsize=(14, 4))

ax = sns.countplot(df['prod_B'], hue=df['TARGET'])

total = len(df['prod_B'])



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2., height + 3,

            '{:1.3f}%'.format(100*(height/total)), ha="center")



plt.title('Prod_B vs TARGET')

plt.show()
prod_B_zero = df[df['prod_B'] == 0]

prod_B_one = df[df['prod_B'] == 1]
fig = plt.figure(figsize=(14, 5))

sns.distplot(prod_B_one['age'], color='orange',bins=44)

sns.distplot(prod_B_zero['age'], color='blue',bins=44)

fig.legend(labels=['prod_B=1', 'prod_B=0'])

fig.tight_layout()
fig = plt.figure(figsize=(14, 5))

sns.distplot(prod_B_one['turnover_B'], color='orange')

sns.distplot(prod_B_zero['turnover_B'], color='blue')

fig.legend(labels=['prod_B=1', 'prod_B=0'])

fig.tight_layout()
fig = plt.figure(figsize=(14, 5))

sns.distplot(prod_B_one['turnover_A'], color='orange')

sns.distplot(prod_B_zero['turnover_A'], color='blue')

fig.legend(labels=['prod_B=1', 'prod_B=0'])

fig.tight_layout()
fig = plt.figure(figsize=(14, 5))

sns.distplot(prod_B_one['ID'], color='orange')

sns.distplot(prod_B_zero['ID'], color='blue')

fig.legend(labels=['prod_B=1', 'prod_B=0'])

fig.tight_layout()
sns.boxplot(x='TARGET', y='turnover_A', data=df)
sns.jointplot(x='turnover_A', y='age',

              data=df, color='blue', kind='kde')
sns.jointplot(x='turnover_A', y='LOR',

              data=df, color='blue', kind='kde')
sns.jointplot(x='turnover_A', y='turnover_B',

              data=df, color='blue', kind='kde')
sns.jointplot(x='turnover_A', y='ID', data=df,

              color='blue', kind='kde')
sns.jointplot(x='turnover_A', y='prod_B',

              data=df, color='blue', kind='kde')
sns.boxplot(x='TARGET', y='turnover_B', data=df)
df[df['TARGET'] == 1]['turnover_B'].median()
df[df['TARGET'] == 0]['turnover_B'].median()
sns.jointplot(x='turnover_A', y='age',

              data=df, color='blue', kind='kde')
sns.jointplot(x='turnover_A', y='LOR',

              data=df, color='blue', kind='kde')
sns.jointplot(x='turnover_A', y='turnover_B',

              data=df, color='blue', kind='kde')
sns.jointplot(x='turnover_A', y='ID', data=df,

              color='blue', kind='kde')
sns.jointplot(x='turnover_A', y='prod_B',

              data=df, color='blue', kind='kde')
plt.figure(figsize=(14, 5))

ax = sns.countplot(x="prod_B", data=df, hue=df['prod_A'])

plt.title('Relationship prod_A and prod_B')



total = len(df['prod_B'])



for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
sns.heatmap(df[['prod_B', 'prod_A']

                             ].corr(), annot=True, cmap='coolwarm')
fig, axs = plt.subplots(nrows=3, figsize=(15, 10))



sns.boxplot(x="age", y="TARGET", ax=axs[0],data=df, orient="h").set_title('TARGET', fontweight="bold", size=15)

sns.boxplot(x="age", y="prod_A", ax=axs[1],data=df, orient="h").set_title('prod_A', fontweight="bold", size=15)

sns.boxplot(x="age", y="prod_B", ax=axs[2],data=df, orient="h").set_title('prod_B', fontweight="bold", size=15)

plt.tight_layout()
print("Median age of people not buying the target product is " +  

    str(df[df['TARGET'] == 0]['age'].median()))

print("Median age of people buying the target product is " +

      str(df[df['TARGET'] == 1]['age'].median()))

print('-'*30)

print("Median age of people not buying prod_A is " +

      str(df[df['prod_A'] == 0]['age'].median()))

print("Median age of people buying prod_A is " +

      str(df[df['prod_A'] == 1]['age'].median()))

print('-'*30)

print("Median age of people not buying prod_B is " +

      str(df[df['prod_B'] == 0]['age'].median()))

print("Median age of people buying prod_B is " +

      str(df[df['prod_B'] == 1]['age'].median()))
plt.figure(figsize=(14, 5))

sns.boxplot(x="TARGET", y="age", hue="prod_A",

            data=df, linewidth=2.5, orient='H')
plt.figure(figsize=(14, 5))

sns.boxplot(x="TARGET", y="age", hue="prod_B",

            data=df, linewidth=2.5, orient='H')
g = sns.catplot(x="age", y="TARGET",

                hue="LOR", col="prod_A",

                data=df, kind="box",

                height=10, aspect=.7, orient='h')
g = sns.catplot(x="age", y="TARGET",

                hue="LOR", col="prod_B",

                data=df, kind="box",

                height=10, aspect=.7, orient='h')
cm_combo = df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(cm_combo, annot=True, cmap='coolwarm')
cm_X_combo = (df.drop('TARGET', axis=1)).corr()

plt.figure(figsize=(12, 8))

sns.heatmap(cm_X_combo, annot=True, cmap='coolwarm')
# https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

X_combo = add_constant(df)
pd.Series([variance_inflation_factor(X_combo.values, i)

           for i in range(X_combo.shape[1])],

          index=X_combo.columns)



# VIF of 5 or 10 and above indicates a multicollinearity problem.

# If there is perfect correlation, then VIF = infinity.
df_combo = df.drop('ID', axis=1)
df_combo_1_LOR_0 = df_combo.drop('LOR', axis=1)

df_combo_L0_prod_A_0 = df_combo_1_LOR_0.drop('prod_A', axis=1)

df_combo_L0_prod_B_0 = df_combo_1_LOR_0.drop('prod_B', axis=1)

df_combo_L0_prod_A_B_0 = df_combo_1_LOR_0.drop(['prod_A', 'prod_B'], axis=1)
df_combo_1_LOR_1 = df_combo

df_combo_L1_prod_A_0 = df_combo.drop('prod_A', axis=1)

df_combo_L1_prod_B_0 = df_combo.drop('prod_B', axis=1)

df_combo_L1_prod_A_B_0 = df_combo.drop(['prod_A', 'prod_B'], axis=1)
cm_df_combo_1_LOR_0 = df_combo_1_LOR_0.corr()

cm_df_combo_L0_prod_A_0 = df_combo_L0_prod_A_0.corr()

cm_df_combo_L0_prod_B_0 = df_combo_L0_prod_B_0.corr()

cm_df_combo_L0_prod_A_B_0 = df_combo_L0_prod_A_B_0.corr()



cm_df_combo_1_LOR_1 = df_combo_1_LOR_1.corr()

cm_df_combo_L1_prod_A_0 = df_combo_L1_prod_A_0.corr()

cm_df_combo_L1_prod_B_0 = df_combo_L1_prod_B_0.corr()

cm_df_combo_L1_prod_A_B_0 = df_combo_L1_prod_A_B_0.corr()



fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)

      ) = plt.subplots(4, 2, figsize=(15, 20), sharey=True)



sns.heatmap(cm_df_combo_1_LOR_0, annot=True, cmap='coolwarm', ax=ax1).set_title(

    'cm_df_combo_1_LOR_0', fontweight="bold", size=15)

sns.heatmap(cm_df_combo_L0_prod_A_0, annot=True, cmap='coolwarm', ax=ax2).set_title(

    'cm_df_combo_L0_prod_A_0', fontweight="bold", size=15)

sns.heatmap(cm_df_combo_L0_prod_B_0, annot=True, cmap='coolwarm', ax=ax3).set_title(

    'cm_df_combo_L0_prod_B_0', fontweight="bold", size=15)

sns.heatmap(cm_df_combo_L0_prod_A_B_0, cmap='coolwarm', annot=True, ax=ax4).set_title(

    'cm_df_combo_L0_prod_A_B_0', fontweight="bold", size=15)

sns.heatmap(cm_df_combo_1_LOR_1, annot=True, cmap='coolwarm', ax=ax5).set_title(

    'cm_df_combo_1_LOR_1', fontweight="bold", size=15)

sns.heatmap(cm_df_combo_L1_prod_A_0, annot=True, cmap='coolwarm', ax=ax6).set_title(

    'cm_df_combo_L1_prod_A_0', fontweight="bold", size=15)

sns.heatmap(cm_df_combo_L1_prod_B_0, annot=True, cmap='coolwarm', ax=ax7).set_title(

    'cm_df_combo_L1_prod_B_0', fontweight="bold", size=15)

sns.heatmap(cm_df_combo_L1_prod_A_B_0, annot=True, cmap='coolwarm', ax=ax8).set_title(

    'cm_df_combo_L1_prod_A_B_0', fontweight="bold", size=15)



fig.tight_layout()
fig = plt.figure(figsize=(14, 5))

ax = sns.countplot(x="TARGET", data=df_combo_L0_prod_B_0)

plt.title('TARGET df_combo_L0_prod_B_0')

total = len(df_combo_L0_prod_B_0['TARGET'])

for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
fig = plt.figure(figsize=(14, 5))

ax = df_combo_L0_prod_B_0.corr()['TARGET'][1:].sort_values().plot(kind='bar')

plt.title('TARGET df_combo_L0_prod_B_0')

total = len(df_combo_L0_prod_B_0.corr()['TARGET'][1:].sort_values())

for p in ax.patches:

    height = p.get_height()



    ax.text(p.get_x()+p.get_width()/2.,

            height + 0,

            '{:1.3f}%'.format(100*(height/total)),

            ha="center")
df_alpha_B = df_combo_L0_prod_B_0
df_alpha_B['age']=np.log(df_alpha_B['age'])
X = df_alpha_B.drop('TARGET', axis=1).values

y = df_alpha_B['TARGET'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = logmodel.predict(X_train)
print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))
plot_confusion_matrix(logmodel, X_train, y_train)
test_predictions = logmodel.predict(X_test)
print(classification_report(y_test, test_predictions))
# Remember:

# 72.674% didn't buy the new product

# 27.326% bought ..
print(confusion_matrix(y_test, test_predictions))
# TN FN

# FP TP



# -TP times the model predicts correctly that a customer will buy TARGET

# -FP times the model predicts icorrectly that a customer will buy TARGET

# -FN times the model predicts incorrectly that a customer won't buy TARGET, while he did buy it

# -TN times the model predicts correctly that a customer does not buy TARGET
plot_confusion_matrix(logmodel, X_test, y_test)
sns.countplot(df_alpha_B['TARGET'])
df_alpha_B.shape
X = df_alpha_B.drop('TARGET', axis=1).values

y = df_alpha_B['TARGET'].values
from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy

oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform

X_over, y_over = oversample.fit_resample(X, y)
from collections import Counter
# summarize class distribution

print(Counter(y))
# summarize class distribution

print(Counter(y_over))
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)



sns.countplot(df_alpha_B['TARGET'], ax=ax1).set_title('Before')

sns.countplot((y_over), ax=ax2).set_title('After')
X = X_over

y = y_over
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression



logmodel_over = LogisticRegression()

logmodel_over.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = logmodel_over.predict(X_train)
print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))
plot_confusion_matrix(logmodel_over, X_train, y_train)
test_predictions = logmodel_over.predict(X_test)
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
plot_confusion_matrix(logmodel_over, X_test, y_test)
sns.countplot(df_alpha_B['TARGET'])
df_alpha_B.shape
X = df_alpha_B.drop('TARGET', axis=1).values

y = df_alpha_B['TARGET'].values
from imblearn.under_sampling import RandomUnderSampler
# define undersampling strategy

oversample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform

X_over, y_over = oversample.fit_resample(X, y)
from collections import Counter
# summarize class distribution

print(Counter(y))
# summarize class distribution

print(Counter(y_over))
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)



sns.countplot(df_alpha_B['TARGET'], ax=ax1).set_title('Before')

sns.countplot((y_over), ax=ax2).set_title('After')
X = X_over

y = y_over
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression



logmodel_under = LogisticRegression()

logmodel_under.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = logmodel_under.predict(X_train)
print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))
plot_confusion_matrix(logmodel_under, X_train, y_train)
test_predictions = logmodel_under.predict(X_test)
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
plot_confusion_matrix(logmodel_under, X_test, y_test)
X = df_alpha_B.drop('TARGET', axis=1).values

y = df_alpha_B['TARGET'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = svc_model.predict(X_train)
print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))
plot_confusion_matrix(svc_model, X_train, y_train)
test_predictions = svc_model.predict(X_test)
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
plot_confusion_matrix(svc_model, X_test, y_test)
sns.countplot(df_alpha_B['TARGET'])
df_alpha_B.shape
X = df_alpha_B.drop('TARGET', axis=1).values

y = df_alpha_B['TARGET'].values
from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy

oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform

X_over, y_over = oversample.fit_resample(X, y)
from collections import Counter
# summarize class distribution

print(Counter(y))
# summarize class distribution

print(Counter(y_over))
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)



sns.countplot(df_alpha_B['TARGET'], ax=ax1).set_title('Before')

sns.countplot((y_over), ax=ax2).set_title('After')
X = X_over

y = y_over
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.svm import SVC
SVM_over = SVC()
SVM_over.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = SVM_over.predict(X_train)
print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))
plot_confusion_matrix(SVM_over, X_train, y_train)
test_predictions = SVM_over.predict(X_test)
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
plot_confusion_matrix(SVM_over, X_test, y_test)
sns.countplot(df_alpha_B['TARGET'])
df_alpha_B.shape
X = df_alpha_B.drop('TARGET', axis=1).values

y = df_alpha_B['TARGET'].values
from imblearn.under_sampling import RandomUnderSampler
# define undersampling strategy

oversample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform

X_over, y_over = oversample.fit_resample(X, y)
from collections import Counter
# summarize class distribution

print(Counter(y))
# summarize class distribution

print(Counter(y_over))
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)



sns.countplot(df_alpha_B['TARGET'], ax=ax1).set_title('Before')

sns.countplot((y_over), ax=ax2).set_title('After')
X = X_over

y = y_over
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.svm import SVC
SVM_under = SVC()
SVM_under.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = SVM_under.predict(X_train)
print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))
plot_confusion_matrix(SVM_under, X_train, y_train)
test_predictions = SVM_under.predict(X_test)
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
plot_confusion_matrix(SVM_under, X_test, y_test)
# pip install -U imbalanced-learn

# https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/

# check version number

import imblearn

print(imblearn.__version__)

sns.countplot(df_alpha_B['TARGET'])
df_alpha_B.shape
X = df_alpha_B.drop('TARGET', axis=1).values

y = df_alpha_B['TARGET'].values
from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy

oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform

X_over, y_over = oversample.fit_resample(X, y)
from collections import Counter
# summarize class distribution

print(Counter(y))
# summarize class distribution

print(Counter(y_over))
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)



sns.countplot(df_alpha_B['TARGET'], ax=ax1).set_title('Before')

sns.countplot((y_over), ax=ax2).set_title('After')
X = X_over

y = y_over
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
RFC_over = RandomForestClassifier(n_estimators=600)
RFC_over.fit(X_train,y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = RFC_over.predict(X_train)
print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))
plot_confusion_matrix(RFC_over, X_train, y_train)
test_predictions = RFC_over.predict(X_test)
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
plot_confusion_matrix(RFC_over, X_test, y_test)
sns.countplot(df_alpha_B['TARGET'])
df_alpha_B.shape
X = df_alpha_B.drop('TARGET', axis=1).values

y = df_alpha_B['TARGET'].values
from imblearn.under_sampling import RandomUnderSampler
# define undersampling strategy

oversample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform

X_over, y_over = oversample.fit_resample(X, y)
from collections import Counter
# summarize class distribution

print(Counter(y))
# summarize class distribution

print(Counter(y_over))
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)



sns.countplot(df_alpha_B['TARGET'], ax=ax1).set_title('Before')

sns.countplot((y_over), ax=ax2).set_title('After')
X = X_over

y = y_over
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
RFC_under = RandomForestClassifier(n_estimators=600)
RFC_under.fit(X_train,y_train)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = RFC_under.predict(X_train)
print(classification_report(y_train, training_predictions))
print(confusion_matrix(y_train, training_predictions))
plot_confusion_matrix(RFC_under, X_train, y_train)
test_predictions = RFC_under.predict(X_test)
print(classification_report(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
plot_confusion_matrix(RFC_under, X_test, y_test)
from sklearn.metrics import f1_score, recall_score
original_dataset = pd.read_csv('/kaggle/input/insurance-company/Customer_data.csv')
original_dataset = original_dataset.drop(['loyalty', 'ID', 'city',

                                          'LOR', 'prod_A', 'type_A',

                                          'type_B', 'contract', 'age_P','lor_M'],axis=1)
original_dataset.columns
original_dataset['age']=np.log(original_dataset['age'])
X_test = original_dataset.drop('TARGET', axis=1).values

y_test = original_dataset['TARGET'].values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)
print(type(X_test))

print(len(X_test))

print(X_test.shape)
print(type(y_test))

print(len(y_test))

print(y_test.shape)
y_test
df_y_test = pd.DataFrame(data=y_test, columns=["true_values"])
def replace_yn(target):

    for t in target:

        if t == 'Y':

            return int(1)

        else:

            return int(0)





df_y_test['true_values'] = df_y_test['true_values'].apply(replace_yn)
df_y_test.values
y_test = df_y_test.values
test_predictions_logmodel_over = logmodel_over.predict(X_test)
print('1. The F-1 score of the model {}\n'.format(f1_score(y_test, test_predictions_logmodel_over, average='macro')))

print('2. The recall score of the model {}\n'.format(recall_score(y_test, test_predictions_logmodel_over, average='macro')))

print('3. Classification report \n {} \n'.format(classification_report(y_test, test_predictions_logmodel_over)))

print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_test, test_predictions_logmodel_over)))
plot_confusion_matrix(logmodel_over, X_test, y_test,normalize='true')
plot_confusion_matrix(logmodel_over, X_test, y_test)
test_predictions_logmodel_under = logmodel_under.predict(X_test)
print('1. The F-1 score of the model {}\n'.format(f1_score(y_test, test_predictions_logmodel_under, average='macro')))

print('2. The recall score of the model {}\n'.format(recall_score(y_test, test_predictions_logmodel_under, average='macro')))

print('3. Classification report \n {} \n'.format(classification_report(y_test, test_predictions_logmodel_under)))

print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_test, test_predictions_logmodel_under)))
plot_confusion_matrix(logmodel_under, X_test, y_test,normalize='true')
plot_confusion_matrix(logmodel_under, X_test, y_test)
test_predictions_svm_over = SVM_over.predict(X_test)
print('1. The F-1 score of the model {}\n'.format(f1_score(y_test, test_predictions_svm_over, average='macro')))

print('2. The recall score of the model {}\n'.format(recall_score(y_test, test_predictions_svm_over, average='macro')))

print('3. Classification report \n {} \n'.format(classification_report(y_test, test_predictions_svm_over)))

print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_test, test_predictions_svm_over)))
plot_confusion_matrix(SVM_over, X_test, y_test,normalize='true')
plot_confusion_matrix(SVM_over, X_test, y_test)
test_predictions_svm_under = SVM_under.predict(X_test)
print('1. The F-1 score of the model {}\n'.format(f1_score(y_test, test_predictions_svm_under, average='macro')))

print('2. The recall score of the model {}\n'.format(recall_score(y_test, test_predictions_svm_under, average='macro')))

print('3. Classification report \n {} \n'.format(classification_report(y_test, test_predictions_svm_under)))

print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_test, test_predictions_svm_under)))
plot_confusion_matrix(SVM_under, X_test, y_test,normalize='true')
plot_confusion_matrix(SVM_under, X_test, y_test)
test_predictions_RFC_over = RFC_over.predict(X_test)
print('1. The F-1 score of the model {}\n'.format(f1_score(y_test, test_predictions_RFC_over, average='macro')))

print('2. The recall score of the model {}\n'.format(recall_score(y_test, test_predictions_RFC_over, average='macro')))

print('3. Classification report \n {} \n'.format(classification_report(y_test, test_predictions_RFC_over)))

print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_test, test_predictions_RFC_over)))
plot_confusion_matrix(RFC_over, X_test, y_test,normalize='true')
plot_confusion_matrix(RFC_over, X_test, y_test)
test_predictions_RFC_under = RFC_under.predict(X_test)
print('1. The F-1 score of the model {}\n'.format(f1_score(y_test, test_predictions_RFC_under, average='macro')))

print('2. The recall score of the model {}\n'.format(recall_score(y_test, test_predictions_RFC_under, average='macro')))

print('3. Classification report \n {} \n'.format(classification_report(y_test, test_predictions_RFC_under)))

print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_test, test_predictions_RFC_under)))
plot_confusion_matrix(RFC_under, X_test, y_test,normalize='true')
plot_confusion_matrix(RFC_under, X_test, y_test)