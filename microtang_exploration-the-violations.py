# Importing libraries

import re

import warnings

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

pd.options.display.max_columns = 50

%matplotlib inline

warnings.filterwarnings("ignore")

sns.set_style("darkgrid")
# Importing the dataset

df = pd.read_csv('../input/Traffic_Violations.csv', low_memory=False)

# Giving the dimension information

print('Dataframe dimensions:', df.shape)

#____________________________________________________________

# Giving some infos on columns types and number of null values

tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))

tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)

                         .T.rename(index={0:'null values (%)'}))

tab_info
# Let's see what we have

df.head()
# Counting the violations by State

df = df[df.State.notnull()]

aggResult = df.groupby(by=['State'])['Fatal'].agg({'Count': np.size})

aggResult['Count'] = aggResult.Count.astype(int)

aggResult = aggResult.sort_values(by = 'Count', ascending=False)

aggResult = aggResult.reset_index()

aggResult.head(20)
# Visualization the result

fig = plt.figure(figsize=(15,23))

x = aggResult['Count']

y = len(aggResult.index) - aggResult.index #swap high and low

labels = aggResult['State']



plt.scatter(x, y, color='g', label = 'Violation Count')

plt.yticks(y, labels)



plt.xlabel('Numbers')

plt.ylabel('State')

plt.title('Violation Count by State')

plt.legend()

plt.show()
# Counting the violations by Maker

df = df.dropna(axis = 0, how='any')

aggResult = df.groupby(by=['Make'])['Fatal'].agg({'Count': np.size})

aggResult['Count'] = aggResult.Count.astype(int)

aggResult = aggResult.sort_values(by = 'Count', ascending=False)

aggResult = aggResult[aggResult['Count'] >500]

aggResult = aggResult.reset_index()

aggResult.head(20)
# Visualization the result

fig = plt.figure(figsize=(15,20))

x = aggResult['Count']

y = len(aggResult.index) - aggResult.index 

labels = aggResult['Make']



plt.scatter(x, y, color='g', label = 'Violation Count')

plt.yticks(y, labels)



plt.xlabel('Number')

plt.ylabel('Maker Name')

plt.title('Violation Count by Maker')

plt.legend()

plt.show()
# Counting the violations by Model

aggResult = df.groupby(by=['Model'])['Fatal'].agg({'Count': np.size})

aggResult['Count'] = aggResult.Count.astype(int)

aggResult = aggResult.sort_values(by = 'Count', ascending=False)

aggResult = aggResult[aggResult['Count'] >1000]

aggResult = aggResult.reset_index()

aggResult.head(20)
# Visualization the results

fig = plt.figure(figsize=(15,23))

x = aggResult['Count']

y = len(aggResult.index) - aggResult.index

labels = aggResult['Model']



plt.scatter(x, y, color='g', label = 'Violation Count')

plt.yticks(y, labels)



plt.xlabel('Number')

plt.ylabel('Model')

plt.title('Violation Count by model')

plt.legend()

plt.show()
# Let's see the kinds of violations

f, axarr = plt.subplots(2, 2, figsize=(16, 16))



f.subplots_adjust(hspace=0.5)



sns.countplot(df['Gender'], ax=axarr[0][0], color='salmon')

axarr[0][0].set_title("Gender", fontsize=14)



sns.countplot(df['Race'], ax=axarr[0][1], color='salmon')

axarr[0][1].set_title("Race", fontsize=14)



sns.countplot(df['Article'], ax=axarr[1][0], color='salmon')

axarr[1][0].set_title("Article", fontsize=14)



sns.countplot(df['Violation Type'], ax=axarr[1][1], color='salmon')

axarr[1][1].set_title("Violation Type", fontsize=14)



sns.despine()
# Let's see the Charge distribution

df['Charge'] = df['Charge'].apply(lambda x: re.findall(r"\d+\.?\d*", x)[1]).astype(float)

kde_kwargs = {'color': 'crimson', 'shade': True}

vis1 = sns.kdeplot(df['Charge'], **kde_kwargs)
# Let's see the violations kinds

f, axarr = plt.subplots(5, 2, figsize=(20, 20))



f.subplots_adjust(hspace=0.5)



sns.countplot(df['Accident'], ax=axarr[0][0], color='salmon')

axarr[0][0].set_title("Accident", fontsize=14)



sns.countplot(df['Belts'], ax=axarr[0][1], color='salmon')

axarr[0][1].set_title("Belts", fontsize=14)



sns.countplot(df['Personal Injury'], ax=axarr[1][0], color='salmon')

axarr[1][0].set_title("Personal Injury", fontsize=14)



sns.countplot(df['Property Damage'], ax=axarr[1][1], color='salmon')

axarr[1][1].set_title("Property Damage", fontsize=14)



sns.countplot(df['Fatal'], ax=axarr[2][0], color='salmon')

axarr[2][0].set_title("Fatal", fontsize=14)



sns.countplot(df['Commercial License'], ax=axarr[2][1], color='salmon')

axarr[2][1].set_title("Commercial License", fontsize=14)



sns.countplot(df['HAZMAT'], ax=axarr[3][0], color='salmon')

axarr[3][0].set_title("HAZMAT", fontsize=14)



sns.countplot(df['Commercial Vehicle'], ax=axarr[3][1], color='salmon')

axarr[3][1].set_title("Commercial Vehicle", fontsize=14)



sns.countplot(df['Alcohol'], ax=axarr[4][0], color='salmon')

axarr[4][0].set_title("Alcohol", fontsize=14)



sns.countplot(df['Work Zone'], ax=axarr[4][1], color='salmon')

axarr[4][1].set_title("Work Zone", fontsize=14)



sns.despine()