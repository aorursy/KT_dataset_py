%%bash



ls -l
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
train_trans = pd.read_csv('../input/train_transaction.csv')

test_trans = pd.read_csv('../input/test_transaction.csv')
SEED = 5000

train_trans.sample(20, random_state=SEED)
train_trans.shape
len(train_trans['TransactionID'].unique())
fig = plt.figure(figsize=(10,7))

sns.countplot(train_trans['isFraud'])
train_trans_stats = train_trans.describe(include='all')
train_trans_stats.loc['max', 'TransactionDT']
train_trans_stats.loc['min', 'TransactionDT']
train_trans_stats.loc['max', 'TransactionDT'] / (60*60*24) / 30
train_trans_stats.loc['min', 'TransactionDT'] / (60*60*24)
train_trans_stats.loc['na'] = train_trans.shape[0] - train_trans_stats.loc['count']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (15,20))



for ax, feature in zip(axes.flatten(), ['card' + str(x) for x in range(1,7)]):

    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)

    ax.set(title=feature.upper())
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,7), sharey=True)



for ax, card_type in zip(axes.flatten(), ['credit', 'debit']):

    ax.bar(train_trans[train_trans['card6'] == card_type]['isFraud'].value_counts().index,

           train_trans[train_trans['card6'] == card_type]['isFraud'].value_counts().values/\

           train_trans[train_trans['card6'] == card_type]['isFraud'].value_counts().sum())

    ax.set(title=card_type.upper())
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,7), sharey=True)



for ax, card_type in zip(axes.flatten(), ['visa', 'mastercard']):

    ax.bar(train_trans[train_trans['card4'] == card_type]['isFraud'].value_counts().index,

           train_trans[train_trans['card4'] == card_type]['isFraud'].value_counts().values/\

           train_trans[train_trans['card4'] == card_type]['isFraud'].value_counts().sum())

    ax.set(title=card_type.upper())
fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (15,7))



for ax, feature in zip(axes.flatten(), ['ProductCD', 'addr1', 'addr2', 'dist1','dist2']):

    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)

    ax.set(title=feature.upper())
p_emails = pd.DataFrame(data={'email_domains':train_trans['P_emaildomain'].value_counts().index,

                      'email_counts':train_trans['P_emaildomain'].value_counts().values})



r_emails = pd.DataFrame(data={'email_domains':train_trans['R_emaildomain'].value_counts().index,

                      'email_counts':train_trans['R_emaildomain'].value_counts().values})
fig = plt.figure(figsize=(15,10))



sns.set(style="whitegrid")



ax = sns.barplot(x='email_counts', y='email_domains', data=p_emails)



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 250000), ylabel="",

       xlabel="'P' email domains (purchaser?)")

sns.despine(left=True, bottom=True)
fig = plt.figure(figsize=(15,10))



sns.set(style="whitegrid")



ax = sns.barplot(x='email_counts', y='email_domains', data=r_emails)



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 250000), ylabel="",

       xlabel="'R' email domains (recipient?)")

sns.despine(left=True, bottom=True)
fig, axes = plt.subplots(nrows=30, ncols=2, figsize = (15,120), sharey=True)



for ax, email_domain in zip(axes.flatten(), p_emails['email_domains']):

    ax.bar(train_trans[train_trans['P_emaildomain'] == email_domain]['isFraud'].value_counts().index,

           train_trans[train_trans['P_emaildomain'] == email_domain]['isFraud'].value_counts().values/\

           train_trans[train_trans['P_emaildomain'] == email_domain]['isFraud'].value_counts().sum())

    ax.set(title=email_domain.upper())
useful_p_domains = ['protonmail.com', 'mail.com', 'outlook.es', 'aim.com']
p_train_emails = p_emails[p_emails['email_domains'].isin(useful_p_domains)]

p_train_emails['email_counts %'] = (p_train_emails['email_counts']*100)/len(train_trans)

p_train_emails
p_test_emails = pd.DataFrame(data={'email_domains':test_trans['P_emaildomain'].value_counts().index,

                      'email_counts':test_trans['P_emaildomain'].value_counts().values})



p_test_emails = p_test_emails[p_test_emails['email_domains'].isin(useful_p_domains)]

p_test_emails['email_counts %'] = (p_test_emails['email_counts']*100)/len(test_trans)

p_test_emails
fig, axes = plt.subplots(nrows=30, ncols=2, figsize = (15,120), sharey=True)



for ax, email_domain in zip(axes.flatten(), r_emails['email_domains']):

    ax.bar(train_trans[train_trans['R_emaildomain'] == email_domain]['isFraud'].value_counts().index,

           train_trans[train_trans['R_emaildomain'] == email_domain]['isFraud'].value_counts().values/\

           train_trans[train_trans['R_emaildomain'] == email_domain]['isFraud'].value_counts().sum())

    ax.set(title=email_domain.upper())
useful_r_domains = ['protonmail.com', 'mail.com', 'outlook.es', 'outlook.com', 'netzero.net']

r_train_emails = r_emails[r_emails['email_domains'].isin(useful_r_domains)]

r_train_emails['email_counts %'] = (r_train_emails['email_counts']*100)/len(train_trans)

r_train_emails
r_test_emails = pd.DataFrame(data={'email_domains':test_trans['R_emaildomain'].value_counts().index,

                      'email_counts':test_trans['R_emaildomain'].value_counts().values})



r_test_emails = r_test_emails[r_test_emails['email_domains'].isin(useful_r_domains)]

r_test_emails['email_counts %'] = (r_test_emails['email_counts']*100)/len(test_trans)

r_test_emails
plt.figure(figsize=(15,9))

sns.distplot(train_trans['TransactionAmt'])
plt.figure(figsize=(15,9))

sns.boxenplot(train_trans['TransactionAmt'])
plt.figure(figsize=(15,9))

sns.countplot(train_trans[train_trans['TransactionAmt'] > 30000]['isFraud'])
train_trans[train_trans['TransactionAmt'] > 30000]
plt.figure(figsize=(15,9))

sns.distplot(test_trans['TransactionAmt'])
plt.figure(figsize=(15,9))

sns.boxenplot(test_trans['TransactionAmt'])
train_trans['TransactionAmt'].describe()
test_trans['TransactionAmt'].describe()
plt.figure(figsize=(15,9))

sns.distplot(train_trans[train_trans['TransactionAmt'] < 30000]['TransactionAmt'])
plt.figure(figsize=(15,9))

sns.boxenplot(train_trans[train_trans['TransactionAmt'] < 30000]['TransactionAmt'])
train_trans[train_trans['TransactionAmt'] > 125]['isFraud'].value_counts()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,7), sharey=True)



axes = axes.flatten()



axes[0].bar(train_trans[train_trans['TransactionAmt'] > 125]['isFraud'].value_counts().index,

            train_trans[train_trans['TransactionAmt'] > 125]['isFraud'].value_counts().values/\

            train_trans[train_trans['TransactionAmt'] > 125]['isFraud'].value_counts().sum())

axes[0].set(title='Transaction amounts > 75th percentile'.upper(),

            ylabel = 'Normalized count',

            xlabel = 'isFraud')



axes[1].bar(train_trans[train_trans['TransactionAmt'] < 125]['isFraud'].value_counts().index,

            train_trans[train_trans['TransactionAmt'] < 125]['isFraud'].value_counts().values/\

            train_trans[train_trans['TransactionAmt'] < 125]['isFraud'].value_counts().sum())

axes[1].set(title='Transaction amounts < 75th percentile'.upper(),

            ylabel = 'Normalized count',

            xlabel = 'isFraud')
fig = plt.figure(figsize=(15,5))

ax = sns.barplot(x=train_trans_stats.columns, y=train_trans_stats.loc['na'])
fig = plt.figure(figsize=(15,10))

ax = sns.barplot(x=train_trans_stats.loc[:, (train_trans_stats.loc['na'] > 0) & (train_trans_stats.loc['na'] < 1500)].columns,

                 y=train_trans_stats.loc[:, (train_trans_stats.loc['na'] > 0) & (train_trans_stats.loc['na'] < 1500)].loc['na'])

'No. of featues without NaNs: {}'.format(len(train_trans_stats.loc[:, train_trans_stats.loc['na'] == 0].columns))
fig = plt.figure(figsize=(15,10))

ax = sns.barplot(x=train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].columns,

                 y=train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].loc['na'])
train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].columns.values
v_features = train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].columns.values[9:].tolist()
len(v_features)
fig = plt.figure(figsize=(15,10))

ax = sns.barplot(x=v_features[:30],

                 y=train_trans_stats.loc[:, v_features[:30]].loc['na'])
train_trans['V138'].dtype
plt.figure(figsize=(15,10))

ax = sns.violinplot(data=train_trans.loc[:, v_features])
train_trans.loc[:, v_features].head(10)
fig, axes = plt.subplots(nrows=36, ncols=4, figsize = (15,120))



for ax, feature in zip(axes.flatten(), v_features):

    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)

    ax.set(title=feature.upper())
d_features = train_trans_stats.loc[:, train_trans_stats.loc['na'] > 450000].columns.values[2:8].tolist()
d_features
import matplotlib as mpl

import importlib

importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (15,10))



for ax, feature in zip(axes.flatten(), d_features):

    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)

    ax.set(title=feature.upper())
fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (15,10))



for ax, feature in zip(axes.flatten(), [feature for feature in d_features if feature not in 'D9']):

    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)

    ax.set(title=feature.upper())
train_trans_stats[d_features]
fig = plt.figure(figsize=(15,10))

ax = sns.barplot(x=train_trans_stats.columns[:60], y=train_trans_stats.loc['na'][:60])
fig, axes = plt.subplots(nrows=2, ncols=4, figsize = (15,7))



for ax, feature in zip(axes.flatten(),

                       train_trans.iloc[:, :60].loc[:, train_trans_stats.loc['na'][:60] > 500000].columns):

    

    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)

    ax.set(title=feature.upper())
len([column for column in train_trans.columns.tolist() if column.startswith('D')])
fig, axes = plt.subplots(nrows=4, ncols=4, figsize = (15,14))



for ax, feature in zip(axes.flatten(),

                       [column for column in train_trans.columns.tolist() if column.startswith('D')]):

    

    ax.bar(train_trans[feature].value_counts().index, train_trans[feature].value_counts().values)

    ax.set(title=feature.upper())
train_trans_stats.loc[:, [column for column in train_trans.columns.tolist() if column.startswith('D')]]
correlations = train_trans.corr()['isFraud'].sort_values()
neg_corrs = correlations.head(10)

pos_corrs = correlations.tail(10)
corrs = pos_corrs.append(neg_corrs)
corrs
train_trans[corrs.index].corr()
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(train_trans[corrs.index].corr(),  cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})