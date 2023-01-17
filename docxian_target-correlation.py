import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read targets file for training data

df_train_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
df_train_scored.shape
df_train_scored.head()
targets = df_train_scored.columns[1:207] # skip id

n_targets = len(targets)
print('Number of targets:', n_targets)
# calc correlation matrix for all (numeric) columns

cor_targets = df_train_scored.corr(method='pearson')

cor_targets
# plot correlations (due to symmetry all pairs except the diagonal appear twice!)

plt.rcParams["figure.figsize"]=(12,12)

plt.matshow(cor_targets)

plt.colorbar()

plt.show()
# extract highest absolute correlations



my_threshold = 0.25



count = 1

for i in range(n_targets):

    var_i = targets[i]

    for j in range(n_targets):

        if (i>j):

            var_j = targets[j]            

            cor_x = df_train_scored[var_i].corr(df_train_scored[var_j])

            if (abs(cor_x) > my_threshold):

                print(count, ': corr(',var_i,',',var_j,') = ', np.round(cor_x,4))

                count = count + 1

        
# 14 : corr( proteasome_inhibitor , nfkb_inhibitor ) =  0.9213

pd.crosstab(df_train_scored.proteasome_inhibitor, df_train_scored.nfkb_inhibitor)
# 11 : corr( pdgfr_inhibitor , kit_inhibitor ) =  0.9156

pd.crosstab(df_train_scored.pdgfr_inhibitor, df_train_scored.kit_inhibitor)
# 4 : corr( kit_inhibitor , flt3_inhibitor ) =  0.7581

pd.crosstab(df_train_scored.kit_inhibitor, df_train_scored.flt3_inhibitor)
# 10 : corr( pdgfr_inhibitor , flt3_inhibitor ) =  0.7051

pd.crosstab(df_train_scored.pdgfr_inhibitor, df_train_scored.flt3_inhibitor)
# and another one with a relatively low correlation

# 7 : corr( nrf2_activator , bcl_inhibitor ) =  0.2533

pd.crosstab(df_train_scored.nrf2_activator, df_train_scored.bcl_inhibitor)
# and finally an example with correlation close zo zero

print('corr = ', df_train_scored.acat_inhibitor.corr(df_train_scored.acetylcholine_receptor_agonist))

pd.crosstab(df_train_scored.acat_inhibitor, df_train_scored.acetylcholine_receptor_agonist)
df_train_scored['multiplicity'] = df_train_scored.iloc[:,1:207].sum(axis=1)

df_train_scored.multiplicity.value_counts()
plt.rcParams["figure.figsize"]=(7,4)

df_train_scored.multiplicity.value_counts().plot(kind='bar')

plt.grid()

plt.show()
# look e. g. at the 6 rows having 7 synchronous 1's

demo = df_train_scored[df_train_scored.multiplicity==7]

demo
# remove multiplicity column first

df_train_scored = df_train_scored.drop(columns=['multiplicity'])

# calc means

target_means = df_train_scored.mean()
# and plot

plt.rcParams["figure.figsize"]=(8,36)

sns.barplot(y=target_means.index, x=target_means.values)

plt.grid()

plt.show()