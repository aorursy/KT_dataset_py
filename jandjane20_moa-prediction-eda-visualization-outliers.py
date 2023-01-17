%matplotlib inline



import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams['figure.figsize'] = (8, 5)

plt.rcParams['font.size'] = 16
df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

df_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
print('# of NaNs in train dataset:', df.isna().values.sum())

print('# of NaNs in test dataset:', df_test.isna().values.sum())
BAR_WIDTH = 0.2
percentage = df.cp_type.value_counts() / len(df) * 100

plt.bar(np.arange(len(percentage)), percentage.values, width=BAR_WIDTH, label='Train')



percentage = df_test.cp_type.value_counts() / len(df_test) * 100

plt.bar(np.arange(len(percentage)) + BAR_WIDTH, percentage.values, width=BAR_WIDTH, label='Test')



plt.xticks(np.arange(len(percentage)) + BAR_WIDTH / 2, percentage.index)

plt.legend()

plt.xlabel('Perturbation (cp_type) column value')

plt.ylabel('% of values')

plt.title('cp_type')

plt.show()
percentage = df.cp_time.value_counts() / len(df) * 100

plt.bar(np.arange(len(percentage)), percentage.values, width=BAR_WIDTH, label='Train')



percentage = df_test.cp_time.value_counts() / len(df_test) * 100

plt.bar(np.arange(len(percentage)) + BAR_WIDTH, percentage.values, width=BAR_WIDTH, label='Test')



plt.xticks(np.arange(len(percentage)) + BAR_WIDTH / 2, percentage.index.astype(str) + ' hours')

plt.xlim(-0.5, 3.3)

plt.legend()

plt.xlabel('Treatment duration (cp_time) column value')

plt.ylabel('% of values')

plt.title('cp_time')

plt.show()
percentage = df.cp_dose.value_counts() / len(df) * 100

plt.bar(np.arange(len(percentage)), percentage.values, width=BAR_WIDTH, label='Train')



percentage = df_test.cp_dose.value_counts() / len(df_test) * 100

plt.bar(np.arange(len(percentage)) + BAR_WIDTH, percentage.values, width=BAR_WIDTH, label='Test')



plt.xticks(np.arange(len(percentage)) + BAR_WIDTH / 2, percentage.index.astype(str) + ' hours')

plt.xlim(-0.5, 2)

plt.legend()

plt.xlabel('Treatment dose (cp_dose) column value')

plt.ylabel('% of values')

plt.title('cp_dose')

plt.show()
gene_cols = [f'g-{i}' for i in range(772)]

df[gene_cols].values.min(), df[gene_cols].values.max()
gene_columns_sample = df[gene_cols].sample(9, axis=1, random_state=42)

gene_columns_sample.describe()
fig, axs = plt.subplots(3, 3, figsize=(15, 6), constrained_layout=True)

for i, col in enumerate(gene_columns_sample):

    sns.distplot(df[col], ax=axs[i // 3, i % 3], label='Train')

    sns.distplot(df_test[col], ax=axs[i // 3, i % 3], label='Test')

    axs[i // 3, i % 3].set_title(col)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 10.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

plt.legend()

plt.show()
sns.distplot(df[gene_cols].mean(), kde=False, bins=75)

plt.title('Gene features mean distribution')

plt.show()
gene_cols_with_high_mean = np.argsort(df[gene_cols].mean())[-3:]

gene_cols_with_low_mean = np.argsort(df[gene_cols].mean())[:3]



fig, axs = plt.subplots(2, 3, figsize=(15, 5), constrained_layout=True)

for i, col_number in enumerate(gene_cols_with_high_mean):

    col_name = f'g-{col_number}'

    sns.distplot(df[col_name], ax=axs[0, i], label='Train')

    sns.distplot(df_test[col_name], ax=axs[0, i], label='Test')

    axs[0, i].set_title(col_name)

for i, col_number in enumerate(gene_cols_with_low_mean):

    col_name = f'g-{col_number}'

    sns.distplot(df[col_name], ax=axs[1, i], label='Train')

    sns.distplot(df_test[col_name], ax=axs[1, i], label='Test')

    axs[1, i].set_title(col_name)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 10.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

fig.suptitle('Gene features with the highest (first row) and the lowest (second row) mean')

plt.legend()

plt.show()
sns.distplot(df[gene_cols].std(), kde=False, bins=75)

plt.title('Gene features std distribution')

plt.show()
gene_cols_with_high_std = np.argsort(df[gene_cols].std())[-3:]

gene_cols_with_low_std = np.argsort(df[gene_cols].std())[:3]



fig, axs = plt.subplots(2, 3, figsize=(15, 5), constrained_layout=True)

for i, col_number in enumerate(gene_cols_with_high_std):

    col_name = f'g-{col_number}'

    sns.distplot(df[col_name], ax=axs[0, i], label='Train')

    sns.distplot(df_test[col_name], ax=axs[0, i], label='Test')

    axs[0, i].set_title(col_name)

for i, col_number in enumerate(gene_cols_with_low_std):

    col_name = f'g-{col_number}'

    sns.distplot(df[col_name], ax=axs[1, i], label='Train')

    sns.distplot(df_test[col_name], ax=axs[1, i], label='Test')

    axs[1, i].set_title(col_name)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 10.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

fig.suptitle('Gene features with the highest (first row) and the lowest (second row) std')

plt.legend()

plt.show()
plt.figure(figsize=(8, 7))

sns.heatmap(df[gene_cols[:50]].corr())

plt.title('Pairwise correlations of the first 50 gene features')

plt.show()
correlations = df[gene_cols].corr()

plt.figure(figsize=(8, 7))

sns.heatmap(correlations)

plt.title('Pairwise correlations of gene features')

plt.show()
sns.distplot(correlations.abs().values.flatten(), kde=False)

plt.title('Gene features pairwise (absolute) correlation coefficients distribution')

plt.show()
correlations_np = correlations.values

correlations_np[np.arange(len(gene_cols)), np.arange(len(gene_cols))] = np.NaN
max_corr = np.nanmax(correlations_np)

i, j = np.where(correlations_np == max_corr)[0]

i, j = f'g-{i}', f'g-{j}'

print(f'Two features with the highest pairwise correlation in the train dataset: {i}, {j}')

print(f'Correlation coefficient on train data:', max_corr)

print(f'Correlation coefficient on test data:', df_test[[i, j]].corr().values[0][1])

plt.scatter(df[i], df[j], alpha=0.05)

plt.xlabel(i)

plt.ylabel(j)

plt.title('Two features with the highest correlation')

plt.show()
min_corr = np.nanmin(correlations_np)

i, j = np.where(correlations_np == min_corr)[0]

i, j = f'g-{i}', f'g-{j}'

print(f'Two features with the lowest pairwise correlation in the train dataset: {i}, {j}')

print(f'Correlation coefficient on train data:', min_corr)

print(f'Correlation coefficient on test data:', df_test[[i, j]].corr().values[0][1])

plt.scatter(df[i], df[j], alpha=0.05)

plt.xlabel(i)

plt.ylabel(j)

plt.title('Two features with the lowest correlation')

plt.show()
cell_viability_cols = [f'c-{i}' for i in range(100)]
viability_columns_sample = df[cell_viability_cols].sample(9, axis=1)

viability_columns_sample.describe()
fig, axs = plt.subplots(3, 3, figsize=(15, 6), constrained_layout=True)

for i, col in enumerate(viability_columns_sample):

    sns.distplot(df[col], ax=axs[i // 3, i % 3], label='Train')

    sns.distplot(df_test[col], ax=axs[i // 3, i % 3], label='Test')

    axs[i // 3, i % 3].set_title(col)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 6.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

plt.legend()

plt.show()
sns.distplot(df[cell_viability_cols].mean(), kde=False, bins=15)

plt.title('Cell viability features mean distribution')

plt.show()
gene_cols_with_high_mean = np.argsort(df[cell_viability_cols].mean())[-3:]

gene_cols_with_low_mean = np.argsort(df[cell_viability_cols].mean())[:3]



fig, axs = plt.subplots(2, 3, figsize=(15, 5), constrained_layout=True)

for i, col_number in enumerate(gene_cols_with_high_mean):

    col_name = f'c-{col_number}'

    sns.distplot(df[col_name], ax=axs[0, i], label='Train')

    sns.distplot(df_test[col_name], ax=axs[0, i], label='Test')

    axs[0, i].set_title(col_name)

for i, col_number in enumerate(gene_cols_with_low_mean):

    col_name = f'c-{col_number}'

    sns.distplot(df[col_name], ax=axs[1, i], label='Train')

    sns.distplot(df_test[col_name], ax=axs[1, i], label='Test')

    axs[1, i].set_title(col_name)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 6.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

fig.suptitle('Cell viability features with the highest (first row) and the lowest (second row) mean')

plt.legend()

plt.show()
sns.distplot(df[cell_viability_cols].std(), kde=False, bins=15)

plt.title('Cell viability features std distribution')

plt.show()
gene_cols_with_high_std = np.argsort(df[cell_viability_cols].std())[-3:]

gene_cols_with_low_std = np.argsort(df[cell_viability_cols].std())[:3]



fig, axs = plt.subplots(2, 3, figsize=(15, 5), constrained_layout=True)

for i, col_number in enumerate(gene_cols_with_high_std):

    col_name = f'c-{col_number}'

    sns.distplot(df[col_name], ax=axs[0, i], label='Train')

    sns.distplot(df_test[col_name], ax=axs[0, i], label='Test')

    axs[0, i].set_title(col_name)

for i, col_number in enumerate(gene_cols_with_low_std):

    col_name = f'c-{col_number}'

    sns.distplot(df[col_name], ax=axs[1, i], label='Train')

    sns.distplot(df_test[col_name], ax=axs[1, i], label='Test')

    axs[1, i].set_title(col_name)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 6.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

fig.suptitle('Cell viability features with the highest (first row) and the lowest (second row) std')

plt.legend()

plt.show()
correlations = df[cell_viability_cols].corr()

plt.figure(figsize=(8, 7))

sns.heatmap(correlations)

plt.title('Pairwise correlations of cell viability features')

plt.show()
sns.distplot(correlations.abs().values.flatten(), kde=False)

plt.title('Cell viability features pairwise (absolute) correlation coefficients distribution')

plt.show()
col = 'c-0'

print(df[df[col] < -9][cell_viability_cols[1:]].values.mean(), df[df[col] >= -9][cell_viability_cols[1:]].values.mean())

sns.distplot(df[df[col] < -9][cell_viability_cols[1:]].values, label='c-1 - c-99 distribution when c-0 < -9')

sns.distplot(df[df[col] >= -9][cell_viability_cols[1:]].values, label='c-1 - c-99 distribution when c-0 >= -9')

plt.title('Cell viability distribution conditioned on c-0')

plt.legend()

plt.show()
print(df.loc[(df[cell_viability_cols] < -9).any(axis=1), cell_viability_cols].values.mean(), 

      df.loc[(df[cell_viability_cols] >= -9).all(axis=1), cell_viability_cols].values.mean())

sns.distplot(df.loc[(df[cell_viability_cols] < -9).any(axis=1), cell_viability_cols], label='any of cell viability features < -9')

sns.distplot(df.loc[(df[cell_viability_cols] >= -9).all(axis=1), cell_viability_cols], label='all of cell viability features >= -9')

plt.title('Conditioned cell viability features distribution')

plt.legend()

plt.show()
print('Percentage of rows containing a value < -9:', (df[cell_viability_cols] < -9).any(axis=1).sum() / len(df) * 100)
correlations = df.loc[(df[cell_viability_cols] >= -9).all(axis=1), cell_viability_cols].corr()

plt.figure(figsize=(8, 7))

sns.heatmap(correlations)

plt.title('Pairwise correlations of cell viability features, outliers excluded')

plt.show()
targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
targets.head()
targets.shape
target_cols = targets.columns[1:]
(targets[target_cols].mean() * 100).plot.hist(bins=50)

plt.title('% of true labels in target distribution')

plt.xlabel('% of true labels')

plt.show()
frequent_targets = (targets[target_cols].mean() * 100).sort_values()[-20:].index
(targets[frequent_targets].mean() * 100).sort_values().plot.bar()

plt.title('Most frequent targets')

plt.ylabel('% of true labels')

plt.show()
vc = targets[target_cols].sum(axis=1).value_counts()

plt.title('# of true labels per row distribution')

plt.ylabel('# of rows')

plt.xlabel('# of true targets per row')

plt.bar(vc.index, vc.values)

plt.show()
counts = np.zeros((len(frequent_targets), len(frequent_targets)))

for i, col1 in enumerate(frequent_targets):

    for j, col2 in enumerate(frequent_targets):

        if i != j:

            counts[i, j] = len(targets[(targets[col1] == 1) & (targets[col2] == 1)]) / len(targets[targets[col1] == 1]) 
plt.figure(figsize=(12, 10))

sns.heatmap(counts, annot=True, fmt=".1f", annot_kws={"size": 14})

plt.xticks(np.arange(len(frequent_targets)), frequent_targets, rotation=90)

plt.yticks(np.arange(len(frequent_targets)), frequent_targets, rotation=0)

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(14, 6), constrained_layout=True)

for i, col in enumerate(gene_columns_sample.columns[:4]):

    sns.distplot(df[df.cp_time == 24][col], label='24 hours', kde=False, ax=axs[i // 2, i % 2])

    sns.distplot(df[df.cp_time == 48][col], label='48 hours', kde=False, ax=axs[i // 2, i % 2])

    sns.distplot(df[df.cp_time == 72][col], label='72 hours', kde=False, ax=axs[i // 2, i % 2])

    axs[i // 2, i % 2].set_title(col)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 10.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

fig.suptitle('Gene features distributions conditioned on cp_time')

plt.legend()

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(14, 6), constrained_layout=True)

for i, col in enumerate(gene_columns_sample.columns[:4]):

    sns.distplot(df[df.cp_dose == 'D1'][col], label='Low dose', kde=False, ax=axs[i // 2, i % 2])

    sns.distplot(df[df.cp_dose == 'D2'][col], label='High dose', kde=False, ax=axs[i // 2, i % 2])

    axs[i // 2, i % 2].set_title(col)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 10.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

fig.suptitle('Gene features distributions conditioned on cp_dose')

plt.legend()

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(14, 6), constrained_layout=True)

for i, col in enumerate(viability_columns_sample.columns[:4]):

    sns.distplot(df[df.cp_time == 24][col], label='24 hours', kde=False, ax=axs[i // 2, i % 2])

    sns.distplot(df[df.cp_time == 48][col], label='48 hours', kde=False, ax=axs[i // 2, i % 2])

    sns.distplot(df[df.cp_time == 72][col], label='72 hours', kde=False, ax=axs[i // 2, i % 2])

    axs[i // 2, i % 2].set_title(col)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 10.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

fig.suptitle('Cell viability features distributions conditioned on cp_time')

plt.legend()

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(14, 6), constrained_layout=True)

for i, col in enumerate(viability_columns_sample.columns[:4]):

    sns.distplot(df[df.cp_dose == 'D1'][col], label='Low dose', kde=False, ax=axs[i // 2, i % 2])

    sns.distplot(df[df.cp_dose == 'D2'][col], label='High dose', kde=False, ax=axs[i // 2, i % 2])

    axs[i // 2, i % 2].set_title(col)

for ax in axs.flat:

    ax.set(xlabel='', ylabel='')

    ax.set_xlim(-10.5, 10.5)

    ax.label_outer()

    ax.title.set_fontsize(12)

fig.suptitle('Cell viability features distributions conditioned on cp_dose')

plt.legend()

plt.show()
correlations = df[gene_cols + cell_viability_cols].corr()
plt.figure(figsize=(16, 7))

sns.heatmap(correlations.loc[cell_viability_cols, gene_cols])

plt.title('Pairwise correlations of gene and cell viabilty features')

plt.show()
plt.figure(figsize=(8, 7))

sns.heatmap(correlations.loc[cell_viability_cols[:50], gene_cols[:50]])

plt.title('Closer look: pairwise correlations of gene and cell viabilty features')

plt.show()
sns.distplot(correlations.loc[cell_viability_cols, gene_cols].abs().values.flatten(), kde=False, bins=200)

plt.title('Gene - cell viability (absolute) correlation coefficients distribution')

plt.show()
correlations = df.loc[(df[cell_viability_cols] >= -9).all(axis=1), gene_cols + cell_viability_cols].corr()
plt.figure(figsize=(16, 7))

sns.heatmap(correlations.loc[cell_viability_cols, gene_cols])

plt.title('Pairwise correlations of gene and cell viabilty features, outliers excluded')

plt.show()
plt.figure(figsize=(8, 7))

sns.heatmap(correlations.loc[cell_viability_cols[:50], gene_cols[:50]])

plt.title('Closer look: pairwise correlations of gene and cell viabilty features, outliers excluded')

plt.show()
sns.distplot(correlations.loc[cell_viability_cols, gene_cols].abs().values.flatten(), kde=False, bins=200)

plt.title('Gene - cell viability (absolute) correlation coefficients distribution, outliers excluded')

plt.show()
targets.loc[df.cp_type == 'ctl_vehicle', target_cols].sum().sum()
EPS = 10 ** -5

REL_ERROR_THRESHOLD = 0.3

columns = ['cp_dose', 'cp_time']



t = targets[target_cols].sum()

target_cols_more_than_once = t[t > 1].index  # we don't want to look at targets with only one true label



for col in columns:

    col_unique_values = df[col].unique()

    target_averages = []

    

    # Step 1: compute mean targets scores

    for val in col_unique_values:

        target_averages.append(targets.loc[df[col] == val, target_cols_more_than_once].mean())

     

    # Step 2: compute relative differences

    for i in range(len(col_unique_values)):

        for j in range(i + 1, len(col_unique_values)):

            rel_diff = abs(target_averages[i] - target_averages[j]) / (pd.concat([target_averages[i], target_averages[j]], axis=1).max(axis=1) + EPS)

            if rel_diff.max() < REL_ERROR_THRESHOLD:

                continue

            print(col_unique_values[i], col_unique_values[j])  # Step 3: output feature values and targets with high relative difference

            for target in rel_diff[rel_diff >= REL_ERROR_THRESHOLD].index:

                print(target, target_averages[i][target], target_averages[i][target])
correlations = pd.concat([df[gene_cols + cell_viability_cols], targets[target_cols]], axis=1).corr()
plt.figure(figsize=(16, 7))

sns.heatmap(correlations.loc[target_cols, gene_cols])

plt.title('Gene features correlations with targets')

plt.show()
plt.figure(figsize=(16, 7))

sns.heatmap(correlations.loc[target_cols, cell_viability_cols])

plt.title('Cell viability features correlations with targets')

plt.show()
correlations.loc[target_cols, cell_viability_cols].mean(axis=1).sort_values()
correlations.loc[target_cols, gene_cols].std(axis=1).sort_values(ascending=False)
for target in ['proteasome_inhibitor', 'nfkb_inhibitor']:

    print(f'Percentage of true labels for {target} target among outliers %.2f' % (targets.loc[(df[cell_viability_cols] < -9).any(axis=1), target].mean() * 100))

    print(f'Percentage of true labels for {target} target among non-outliers %.2f' % (targets.loc[(df[cell_viability_cols] >= -9).all(axis=1), target].mean() * 100))

    print('-' * 20)