import os



import pandas as pd



import seaborn as sns



import matplotlib.pyplot as plt



from tqdm import tqdm_notebook



import numpy as np



import warnings



import scipy
warnings.filterwarnings("ignore")



%matplotlib inline



pd.options.display.max_rows = 10000



pd.options.display.max_colwidth = 1000
print(os.listdir("../input"))
def calculate_h0_rejected(df1, df2, alpha=0.05):

    features = ['var_{}'.format(feature_number) for feature_number in range(200)]

    p_values = np.array(

        [

            scipy.stats.mannwhitneyu(

                df1[feature],

                df2[feature]

            )[1] for feature in tqdm_notebook(features)

        ])

    h0_rejected_hypotheses = p_values < alpha

    return pd.DataFrame(

        {

            'p_values': p_values,

            'h0_rejected_hypotheses': h0_rejected_hypotheses

        },

        index=features

    )
def calculate_statistics_distributions(df1, df2, names):

    statistics = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

    

    df1_features_stats = pd.melt(

        df1.describe().T.reset_index(),

        id_vars=['index'],

        value_vars=statistics

    )

    df1_features_stats['data_part'] = names[0]



    df2_features_stats = pd.melt(

        df2.describe().T.reset_index(),

        id_vars=['index'],

        value_vars=statistics

    )

    df2_features_stats['data_part'] = names[1]

    

    df1_df2_features_stats = pd.concat(

        [

            df1_features_stats,

            df2_features_stats

        ],

        ignore_index=True

    )

    

    return df1_df2_features_stats
train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)
train[train.columns.difference(['target'])].describe().T
train.head().T
train.info(verbose=True, null_counts=True)
test.describe().T
test.head().T
test.info(verbose=True, null_counts=True)
train_test_features_stats = calculate_statistics_distributions(

    train[train.columns.difference(['target'])],

    test,

    ['train', 'test']

)
train_test_features_stats.head()
plt.figure(figsize=(20, 10))

sns.boxplot(x='variable', y='value', hue='data_part', data=train_test_features_stats)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x='data_part', y='value', hue='variable', data=train_test_features_stats)

plt.show()
pos_neg_features_stats = calculate_statistics_distributions(

    train[train.target == 1][train.columns.difference(['target'])],

    train[train.target == 0][train.columns.difference(['target'])],

    ['positive', 'negative']

)
pos_neg_features_stats.head()
plt.figure(figsize=(20, 10))

sns.boxplot(x='variable', y='value', hue='data_part', data=pos_neg_features_stats)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x='data_part', y='value', hue='variable', data=pos_neg_features_stats)

plt.show()
neg_test_features_stats = calculate_statistics_distributions(

    train[train.target == 0][train.columns.difference(['target'])],

    test,

    ['negative', 'test']

)
neg_test_features_stats.head()
plt.figure(figsize=(20, 10))

sns.boxplot(x='variable', y='value', hue='data_part', data=neg_test_features_stats)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x='data_part', y='value', hue='variable', data=neg_test_features_stats)

plt.show()
pos_test_features_stats = calculate_statistics_distributions(

    train[train.target == 1][train.columns.difference(['target'])],

    test,

    ['positive', 'test']

)
pos_test_features_stats.head()
plt.figure(figsize=(20, 10))

sns.boxplot(x='variable', y='value', hue='data_part', data=pos_test_features_stats)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x='data_part', y='value', hue='variable', data=pos_test_features_stats)

plt.show()
alpha = 0.001
train_test_H0_rejected = calculate_h0_rejected(train, test, alpha)
plt.figure(figsize=(20, 10))

sns.countplot(x=train_test_H0_rejected.h0_rejected_hypotheses)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x=train_test_H0_rejected.p_values)

plt.grid(True)

plt.show()
neg_pos_H0_rejected = calculate_h0_rejected(train[train.target == 0], train[train.target == 1], alpha)
plt.figure(figsize=(20, 10))

sns.countplot(x=neg_pos_H0_rejected.h0_rejected_hypotheses)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x=neg_pos_H0_rejected.p_values)

plt.grid(True)

plt.show()
neg_test_H0_rejected = calculate_h0_rejected(train[train.target == 0], test, alpha)
plt.figure(figsize=(20, 10))

sns.countplot(x=neg_test_H0_rejected.h0_rejected_hypotheses)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x=neg_test_H0_rejected.p_values)

plt.grid(True)

plt.show()
pos_test_H0_rejected = calculate_h0_rejected(train[train.target == 1], test, alpha)
plt.figure(figsize=(20, 10))

sns.countplot(x=pos_test_H0_rejected.h0_rejected_hypotheses)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x=pos_test_H0_rejected.p_values)

plt.grid(True)

plt.show()