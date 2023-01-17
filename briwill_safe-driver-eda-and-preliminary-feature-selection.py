import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

from IPython.display import display

%matplotlib inline



#This keeps the "middle" columns from being omitted when wide dataframes are being displayed

pd.options.display.max_columns = None



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#customers = pd.read_csv('../input/train.csv', na_values='-1')

customers = pd.read_csv('../input/train.csv')
customers.head()
for colname in customers.columns:

    print (colname, customers[colname].dtype)
bin_cols = ['target']

cat_cols = []

cont_cols = []



for colname in customers.columns:

        if 'bin' in colname:

            bin_cols.append(colname)

        elif 'cat' in colname:

            cat_cols.append(colname)

        else:

            cont_cols.append(colname)
print (len(customers))



customers[bin_cols].describe()
customers[bin_cols].corr()
# Adding features to the list with a correlation magnitude of 0.01 or greater. Tiny, I know...

potential_features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_16_bin', 'ps_ind_17_bin']
customers[cat_cols].describe()
customers[['target', 'ps_car_08_cat']].corr()
# remove the binary-like feature

cat_cols = list(set(cat_cols) - set(['ps_car_08_cat']))

# add it to the previous binary features list

bin_cols += ['ps_car_08_cat']

# add it to the potential features list

potential_features += ['ps_car_08_cat']
#sort by correlation to target and save correlation dataframe

bin_corr = customers[bin_cols].corr().sort_values(['target'], ascending=0) 

#reorder the dataframe columns so we have the nice symmetry of the 1 correlations down the diagonal

bin_corr = bin_corr[list(bin_corr.index.values)]

bin_corr
plt.figure(figsize=(18, 14))

sns.heatmap(bin_corr, cmap="YlGnBu", annot=True, fmt='03.2f')
for col in cat_cols:

    sns.countplot(x=col, hue="target", data=customers)

    plt.show()
for col in cat_cols:

    cont_table = pd.crosstab(customers['target'], customers[col])

    print ("Feature:", col, "P-value:", stats.chi2_contingency(observed= cont_table)[1])
cat_cols = list(set(cat_cols) - set(['ps_car_10_cat']))



potential_features += cat_cols
cont_cols = list(set(cont_cols) - set(['id']) - set(['target']))



customers[cont_cols].describe()


plt.figure(figsize=(18, 14))

plt.xticks(rotation=90)

sns.boxplot(data=customers[cont_cols])
print ("Missing Feature Counts:")

print ("ps_reg_03: ", customers['ps_reg_03'].value_counts().loc[-1])

print ("ps_car_12: ", customers['ps_car_12'].value_counts().loc[-1])

print ("ps_car_14: ", customers['ps_car_14'].value_counts().loc[-1])

print ("ps_car_11: ", customers['ps_car_11'].value_counts().loc[-1])
plt.figure(figsize=(18, 14))

sns.heatmap(customers[(customers['ps_reg_03'] != -1) & (customers['ps_car_14'] != -1)][cont_cols].corr(), cmap="YlGnBu", annot=True, fmt='03.2f')
temp = customers[(customers['ps_reg_03'] != -1) & (customers['ps_car_14'] != -1)][cont_cols]

sns.regplot(data=temp, x='ps_reg_02', y='ps_reg_03')

plt.show()

sns.regplot(data=temp, x='ps_car_13', y='ps_car_14')

plt.show()

sns.regplot(data=temp, x='ps_car_12', y='ps_car_14')

plt.show()
lst = []



for col in cont_cols:

    if (col == 'ps_reg_03' or col == 'ps_car_14'):

        slope, intercept, r_value, p_value, std_err = stats.linregress(customers[customers[col] != -1][col], customers[customers[col] != -1]['target'])

        lst.append([col, slope, p_value])              

    else:

        slope, intercept, r_value, p_value, std_err = stats.linregress(customers[col], customers['target'])

        lst.append([col, slope, p_value])

        

cont_features = pd.DataFrame(lst, columns=['Feature', 'Slope', 'P-value'])

cont_features.sort_values(['P-value'], inplace=True)

                   

cont_features
#Add features with p-value over .01 to potential features list

potential_features += list(cont_features[cont_features['P-value'] < .01]['Feature'])
print ("Count: ", len(potential_features))

potential_features