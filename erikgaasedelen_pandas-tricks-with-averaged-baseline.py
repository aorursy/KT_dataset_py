import pandas as pd
def rsna_to_pivot(df, sub_type_name='HemType'):

    """Convert RSNA data frame to pivoted table with

    each subtype as a binary encoded column."""

    df2 = df.copy()

    ids, sub_types = zip(*df['ID'].str.rsplit('_', n=1).values)

    df2.loc[:, 'ID'] = ids

    df2.loc[:, sub_type_name] = sub_types

    return df2.pivot(index='ID', columns=sub_type_name, values='Label')



def pivot_to_rsna(df, sub_type_name='HemType'):

    """Converted pivoted table back to RSNA spec for submission."""

    df2 = df.copy()

    df2 = df2.reset_index()

    unpivot_vars = df2.columns[1:]

    df2 = pd.melt(df2, id_vars='ID', value_vars=unpivot_vars, var_name=sub_type_name, value_name='Label')

    df2['ID'] = df2['ID'].str.cat(df2[sub_type_name], sep='_')

    df2.drop(columns=sub_type_name, inplace=True)

    return df2
sample_sub = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')

train = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

#First step is to get rid of duplicated entries. Luckily they are consistent.
# Let's check that that is true just to be sure.

# If there were any groups that were not consistent,

# the set of labels should be more than 0 and 1, therefore we are safe.

set(train.groupby('ID').mean()['Label'].values)
# Actually remove duplicates

train = train.groupby('ID').first().reset_index()

train.head()
print(train.shape) # Notice all the rows

train.head()
split_series = train['ID'].str.rsplit('_', n=1)

split_series.head()
ids, sub_types = zip(*train['ID'].str.rsplit('_', n=1).values)

train.loc[:, 'ID'] = ids

train.loc[:, 'HemType'] = sub_types # We are using HemType as our column name for our sub_types

train.head()
train = train.pivot(index='ID', columns='HemType', values='Label')

print(train.shape)

train.head()
# Yay! That's what I like to see. Let's grab some stats real quick.

# We can save these for later.

train.mean()
train = train.reset_index()
unpivot_vars = train.columns[1:] # Here we need the names of categories so we can push them back in the ID

train = pd.melt(train, id_vars='ID', value_vars=unpivot_vars, var_name='HemType', value_name='Label')

train.head()
train['ID'] = train['ID'].str.cat(train['HemType'], sep='_')

train.drop(columns='HemType', inplace=True)

train.head()
# The only additions I"m adding is copying dataframes so we don't accidentally change data we want to keep.



def rsna_to_pivot(df, sub_type_name='HemType'):

    """Convert RSNA data frame to pivoted table with

    each subtype as a binary encoded column."""

    df2 = df.copy()

    ids, sub_types = zip(*df['ID'].str.rsplit('_', n=1).values)

    df2.loc[:, 'ID'] = ids

    df2.loc[:, sub_type_name] = sub_types

    return df2.pivot(index='ID', columns=sub_type_name, values='Label')



def pivot_to_rsna(df, sub_type_name='HemType'):

    """Converted pivoted table back to RSNA spec for submission."""

    df2 = df.copy()

    df2 = df2.reset_index()

    unpivot_vars = df2.columns[1:]

    df2 = pd.melt(df2, id_vars='ID', value_vars=unpivot_vars, var_name=sub_type_name, value_name='Label')

    df2['ID'] = df2['ID'].str.cat(df2[sub_type_name], sep='_')

    df2.drop(columns=sub_type_name, inplace=True)

    return df2
# Just prep work we did before

sample_sub = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')

train = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

train = train.groupby('ID').first().reset_index()
# Easy. Abstraction makes life great.

train_pivot = rsna_to_pivot(train)

sample_sub_pivot = rsna_to_pivot(sample_sub)
train_pivot.head()
sample_sub_pivot.head()
# We did this before.

averages = train_pivot.mean()

averages
# Go through the averages and deliver them to the columns of the submission.

for label, value in averages.items():

    sample_sub_pivot.loc[:, label] = value
sample_sub_pivot.head()
# We pivoted and now let's melt this back on in.

submission = pivot_to_rsna(sample_sub_pivot)

submission.head()
# What's easier than this?

submission.to_csv('submission.csv', index=False)