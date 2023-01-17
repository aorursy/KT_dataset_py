with open('../input/creditapprovaldataset/crx.txt', 'r') as file:

    print(file.read())
import pandas as pd



print('Environment ready!')
col_names = [('A'+str(i)) for i in range(1,17)]

df = pd.read_csv('../input/creditapprovaldataset/crx.data', names=col_names, header=None)



print(f'Dataset Shape: {df.shape}')

df
print(f'Number of Columns: {len(df.columns)}')

df.columns
df.info()
df.columns
missing_df = pd.DataFrame()



for idx in range(len(df)):

    if df.iloc[idx].isin(['?']).any():

        missing_df = missing_df.append(df.iloc[idx], ignore_index=False)

        

print(f'Number of records with missing values: {missing_df.shape}')

missing_df.head()
for col in missing_df.columns:

    tally = df[col].isin(['?']).sum()

    if tally != 0:

        print(f'{col}: {tally}')
import numpy as np

df = df.replace(to_replace='?', value=np.nan)



print(f'Shape: {df.shape}')

print(f'Missing values: {df.isna().sum()}')

df.head()
df.info()
df.head()
df[['A2', 'A14']] = df[['A2', 'A14']].astype('float64')

df.dtypes
df.describe()
df = df.replace({'+':1, '-':0})

print(df.shape)

df.head()
cols_one_unique = [col for col in df.columns if df[col].nunique() <= 1 ]



print(f'These are the features with only one unique value: {cols_one_unique}')
for col in df.columns:

    print(col, ': ', df[col].nunique())
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.pairplot(data=df, vars=df.columns)

plt.show()
import math

numerical_vars = [feat for feat in df.select_dtypes(exclude='object').columns.to_list()]



n_cols=3

n_rows = math.ceil(len(numerical_vars)/n_cols)

i=0



fig,ax=plt.subplots(n_rows, n_cols, figsize=(20,20))



for var in numerical_vars:

    i += 1

    plt.subplot(n_rows, n_cols, i)

    plt.title(var)

    sns.boxplot(x=df.A16, y=df[var])



plt.show()
n_cols = 3

n_rows = math.ceil(len(numerical_vars)/n_cols)

i=0



fig,ax = plt.subplots(n_rows, n_cols, figsize=(20,20))



for var in numerical_vars:

    i+=1

    plt.subplot(n_rows, n_cols, i)

    plt.title(var)

    sns.swarmplot(x=df.A16, y=df[var])



plt.show()
denied_df = df.loc[df.A16 == 0]

approved_df = df.loc[df.A16 == 1]



n_cols = 3

n_rows = math.ceil(len(numerical_vars)/n_cols)

i=0



fig,ax = plt.subplots(n_rows, n_cols, figsize=(20,10))



for var in numerical_vars:

    i+=1

    plt.subplot(n_rows, n_cols, i)

    plt.title(var)

    sns.distplot(a = denied_df[var], hist=False, label='denied', kde_kws={'bw':1.5}, color="#5982C5")

    sns.distplot(a = approved_df[var], hist=False, label='approved', kde_kws={'bw':1.5}, color="#FB3523")



plt.show()
!pip install yellowbrick
from yellowbrick.target import FeatureCorrelation

from yellowbrick.classifier import ClassBalance, ClassificationReport, ConfusionMatrix, DiscriminationThreshold

from yellowbrick.features import JointPlotVisualizer, PCADecomposition, RadViz, Rank1D, Rank2D



print('Visual EDA ready!')
# I usually implement copy() to preserver the original dataframe with which I can always revert back to when

# there's a mistake in cleaning.



# I need to drop the rows with None as the visualizers used below can't handle NA

df = df.dropna()

X = df.drop('A16', axis=1, inplace=False)

y = df[['A16']].astype('int')



print(f'Feature Variables: {X.shape}')

print(f'Target Variable: {y.shape}')



X.head()
feat_matrix = X.values

target_vector = y.values.flatten()



print(f'Features: {feat_matrix.shape}')

print(f'Target/Response: {target_vector.shape}')



feat_vars = X.columns.to_list()

target_var = y.columns.to_list()

print(f'Feature Variables: {feat_vars}')

print(f'Target Variable: {target_var}')



target_balance = ClassBalance(labels=['denied', 'approved'])

target_balance.fit(target_vector)

target_balance.show()
numerical_vars = [feat for feat in X.select_dtypes(exclude='object').columns.to_list()]



feature_correlation = FeatureCorrelation(method='mutual_info-classification',

                                        feature_names=numerical_vars,

                                        sort=True)



feature_correlation.fit(X[numerical_vars], y[target_var].values.flatten())

feature_correlation.show()
import matplotlib.pyplot as plt

%matplotlib inline



design_matrix = X[numerical_vars].values

target_vector = y.values.flatten()



rad_viz = RadViz(classes=['denied', 'approved'], features=numerical_vars, colormap='winter')



plt.figure(figsize=(10,10))

rad_viz.fit(design_matrix, target_vector)

rad_viz.show()
rank_1D = Rank1D(algorithm='shapiro', features=numerical_vars, color='goldenrod')

rank_1D.fit_transform_show(design_matrix, target_vector)
rand_2D = Rank2D(algorithm='pearson', features=numerical_vars, colormap='bwr')

plt.figure(figsize=(10,10))

rand_2D.fit_transform_show(design_matrix, target_vector)
import seaborn as sns



plt.figure(figsize=(10,10))

ax = sns.heatmap(abs(df.drop('A16', axis=1).corr()),

                 vmin=0, vmax=1, center=0.5,

                 cmap='RdYlGn_r',

                 square=True,

                 annot=True,

                 annot_kws={"size": 12})



ax.set_xticklabels(ax.get_xticklabels(),

                  rotation=45,

                  horizontalalignment='right')
colors = np.array(['red' if yi else 'blue' for yi in target_vector])



plt.figure(figsize=(10,7))

pca = PCADecomposition(scale=True, proj_features=True)

pca.fit_transform_show(design_matrix, target_vector, colors=colors)
categorical_vars = [feat for feat in X.select_dtypes(include='object').columns.to_list()]

categorical_vars
n_cols = 4

n_rows = math.ceil(len(categorical_vars)/n_cols)

i = 0



fig,axs = plt.subplots(n_rows, n_cols, figsize=(20,10))



for var in categorical_vars:

    i += 1

    plt.subplot(n_rows,n_cols,i)

    plt.title(var)

    sns.barplot(x=df[var].value_counts().index, y=df[var].value_counts().values)



plt.show()