#importing necessary libraries



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

from matplotlib import patches
#setting up path for the data directory and reading csv files

path = Path('../input/av-healthcare-analytics-ii/healthcare')

train = pd.read_csv(path/'train_data.csv')

data_info = pd.read_csv(path/'train_data_dictionary.csv', index_col = 'Column')

test = pd.read_csv(path/'test_data.csv')
data_info
train.head()
def get_info(parameter):

    print(data_info.loc[parameter]['Description'])



get_info('Hospital_region_code')
train['Stay'] = train['Stay'].apply(lambda x: x.replace('More than 100 Days', '100+'))
test.head()
print(f"shape of training data is {train.shape}.")

print(f'shape of testing data is {test.shape}.')

percent = (test.shape[0]/train.shape[0])*100

print(f"tesing data is {percent:.2f}% of the training data.")
train.info()
train.nunique()
train.describe()
def bar_chart(parameter, figsize=(8,8)):

    target_counts = train[parameter].value_counts()

    target_perc = target_counts.div(target_counts.sum(), axis=0)

    plt.figure(figsize=figsize)

    ax = sns.barplot(x=target_counts.index.values, y=target_counts.values, order=target_counts.index)

    plt.xticks(rotation=90)

    plt.xlabel(f'{parameter}', fontsize=12)

    plt.ylabel('# of occurances', fontsize=12)



    rects = ax.patches

    labels = np.round(target_perc.values*100, 2)

    for rect, label in zip(rects, labels):

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2, height + 5, f'{label}%', ha='center', va='bottom')
bar_chart('Stay')
bar_chart('Department')
bar_chart('Available Extra Rooms in Hospital', figsize=(20, 4))
get_info('Ward_Type')

bar_chart('Ward_Type')
bar_chart('Age')
bar_chart('Type of Admission')
bar_chart('Severity of Illness')
bar_chart('Bed Grade')
bar_chart('Hospital_region_code')
plt.figure(figsize=(12,4))

sns.distplot(train['Admission_Deposit'], kde=False, bins=50)
order = ['0-10', '11-20', '21-30', '31-40', 

         '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']
sns.catplot(x="Stay", hue="Bed Grade", kind="count",

            palette="pastel", edgecolor=".6",

            data=train, height=5, aspect=2, order=order)

plt.xticks(rotation=90);
sns.catplot(x="Stay", hue="Severity of Illness", kind="count",

            palette="colorblind", edgecolor=".9",

            data=train, height=5, aspect=2, order=order)

plt.xticks(rotation=90);
sns.catplot(x='Stay', hue="Type of Admission", kind="count",

            palette="colorblind", edgecolor=".6",

            data=train, height=5, aspect=2, order=order)

plt.xticks(rotation=90);
sns.catplot(x='Stay', hue="Hospital_region_code", kind="count",

            palette="colorblind", edgecolor=".6",

            data=train, height=5, aspect=2, order=order)

plt.xticks(rotation=90);
plt.figure(figsize=(14,8))

sns.boxplot(y='Visitors with Patient',x='Stay', hue="Severity of Illness", order=order, data=train);
plt.figure(figsize=(14,8))

sns.boxplot(x='Stay',y='Admission_Deposit', order=order, data=train);
sns.catplot(x='Stay', hue="Ward_Facility_Code", kind="count",

            palette="colorblind", edgecolor=".6",

            data=train, height=5, aspect=2, order=order)

plt.xticks(rotation=90);