import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
# setting up default plotting parameters

%matplotlib inline



plt.rcParams['figure.figsize'] = [20.0, 7.0]

plt.rcParams.update({'font.size': 22,})



sns.set_palette('viridis')

sns.set_style('white')

sns.set_context('talk', font_scale=0.8)
train_raw = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test_raw = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')



print(train_raw.shape, test_raw.shape)

train_raw.head()
train_raw.info()
# check for missing values

# from https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(train_raw).head(10)
# using seaborns countplot to show distribution of questions in dataset

fig, ax = plt.subplots()

g = sns.countplot(train_raw.target, palette='viridis')

g.set_xticklabels(['0', '1'])

g.set_yticklabels([])



# function to show values on bars

def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.0f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)

show_values_on_bars(ax)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Distribution of Target', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
binary_features = train_raw[['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']].copy()

binary_features.head()
def encode_binary_features(df):

    df['bin_3'] = df['bin_3'].replace({'T':1, 'F':0})

    df['bin_4'] = df['bin_4'].replace({'Y':1, 'N':0})

    return df



binary_features = encode_binary_features(binary_features)

binary_features.head()
# using seaborns countplot to show distribution of binary features

fig, ax = plt.subplots()

ax = plt.subplot(231)

g = sns.countplot(binary_features.bin_0, palette='viridis')

g.set_xticklabels(['0', '1'])

g.set_yticklabels([])

plt.title('Distribution of bin_0', fontsize=20)

show_values_on_bars(ax)

plt.xlabel('')

plt.ylabel('')



ax = plt.subplot(232)

g = sns.countplot(binary_features.bin_1, palette='viridis')

g.set_xticklabels(['0', '1'])

g.set_yticklabels([])

plt.title('Distribution of bin_1', fontsize=20)

show_values_on_bars(ax)

plt.xlabel('')

plt.ylabel('')



ax = plt.subplot(233)

g = sns.countplot(binary_features.bin_2, palette='viridis')

g.set_xticklabels(['0', '1'])

g.set_yticklabels([])

plt.title('Distribution of bin_2', fontsize=20)

show_values_on_bars(ax)

plt.xlabel('')

plt.ylabel('')



ax = plt.subplot(234)

g = sns.countplot(binary_features.bin_3, palette='viridis')

g.set_xticklabels(['0', '1'])

g.set_yticklabels([])

plt.title('Distribution of bin_3', fontsize=20)

show_values_on_bars(ax)

plt.xlabel('')

plt.ylabel('')



ax = plt.subplot(235)

g = sns.countplot(binary_features.bin_4, palette='viridis')

g.set_xticklabels(['0', '1'])

g.set_yticklabels([])

plt.title('Distribution of bin_5', fontsize=20)

show_values_on_bars(ax)

plt.xlabel('')

plt.ylabel('')



plt.subplots_adjust(hspace = 0.3, wspace=.3)

sns.despine(left=True, bottom=True)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
from sklearn.preprocessing import LabelEncoder



#label encoder can't handle missing values

ordinal_features['ord_1'] = ordinal_features['ord_1'].fillna('None')



# Label encode ord_1 feature

label_encoder = LabelEncoder()

ordinal_features['ord_1'] = label_encoder.fit_transform(ordinal_features['ord_1'])



# Print sample of dataset

ordinal_features.head()
ordinal_features = train_raw[['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']].copy()

ordinal_features.head()
def encode_ordinal_features(df):

    ord_1 = {'Novice':1, 

            'Contributor':2, 

            'Expert':4, 

            'Master':5, 

            'Grandmaster':6}

    df['ord_1'] = df.ord_1.map(ord_1)

    ord_2 = {'Freezing':1, 

            'Cold':2, 

            'Warm':3, 

            'Hot':4, 

            'Boiling Hot':5, 

            'Lava Hot':6}

    df['ord_2'] = df.ord_2.map(ord_2)

    return df





ordinal_features = encode_ordinal_features(ordinal_features)

ordinal_features.head()
# start over with a clean copy from our raw data

ordinal_features = train_raw[['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']].copy()

ordinal_features.head()
nominal_features = pd.get_dummies(nominal_features, drop_first=True)

nominal_features.head()
nominal_features = train_raw[['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9']].copy()

nominal_features.head()