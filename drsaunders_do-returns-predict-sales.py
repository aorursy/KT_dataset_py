import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import time 

import seaborn as sns

from sklearn import cross_validation
# This bit of memory-saving code is from Eric Couto, https://www.kaggle.com/ericcouto/grupo-bimbo-inventory-demand/using-82-less-memory/code

types = {'Semana':np.uint8, 'Agencia_ID':np.uint16, 'Canal_ID':np.uint8,

         'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 'Producto_ID':np.uint16,'Venta_uni_hoy':np.uint32,

         'Dev_uni_proxima':np.uint32, 'Demanda_uni_equil':np.uint32 }

train = pd.read_csv('../input/train.csv', usecols=types.keys(), dtype=types)

print(train.info(memory_usage=True))

new_names = ["week","depot","channel","route","client","prod","sales_units","return_units_next_week","demand"]

train.columns = new_names
def plot_weekly_samples(w, h, pt_sales, pt_returns):

    plt.figure(figsize=[12,12])

    ind = np.arange(len(pt_sales.columns))

    width = 0.3

    for i in range(len(pt_sales)):

        plt.subplot(h,w,i+1)

        plt.bar(left=ind, height=pt_sales.iloc[i], width=width,linewidth=4,color='b',edgecolor='b')

        plt.bar(left=ind+width+0.2, height=pt_returns.iloc[i], width=width,linewidth=4,color='r',edgecolor='r')

        plt.title(pt_sales.iloc[i].name)

        plt.xlim([0,7])

        plt.xticks(ind+width+0.1,pt_sales.columns)

    plt.subplots_adjust(hspace = 0.7, left = 0.05, right = 0.95, bottom = 0.05, top=0.95)
# Number of rows and columns to have in the output

w = 4

h = 8

sample_size = w*h



randseed = 7



# Select a random sample of sample_size rows from the training data

dummy, sample_rows = cross_validation.train_test_split(range(len(train)), test_size=sample_size, train_size=0, random_state=randseed)

selected = train.iloc[sample_rows][['client','route','prod']]



# For each unique client, route, and product combo from those rows, look up all the weeks of data

# associated with them (this will also result in averaging over multiple depots, for example, if there

# are different entries for them)

selected_train = pd.DataFrame()

for i,t in selected.iterrows():

    selected_train = pd.concat([selected_train, train.loc[(train.loc[:,'client']==t.client) & (train.loc[:,'route'] == t.route) & (train.loc[:,'prod'] == t.loc['prod']),:]])



# Make pivot tables for sales and returns with weeks for columns and samples for rows

pt_sales = selected_train.pivot_table(values='sales_units',index=['client','route','prod'],columns=['week'],aggfunc=np.mean)

pt_returns = selected_train.pivot_table(values='return_units_next_week',index=['client','route','prod'],columns=['week'],aggfunc=np.mean)



# Plot these sample product histories in a grid (arbitrarily)

plot_weekly_samples(w, h, pt_sales, pt_returns)
print("Training entries that have any return units at all: %.1f%%" % (100*np.mean(train.return_units_next_week > 0)))
with_returns_indices = np.nonzero(train['return_units_next_week'] >0)[0]

dummy, sample_indices = cross_validation.train_test_split(range(len(with_returns_indices)), test_size=sample_size, train_size=0, random_state=randseed)

sample_rows = with_returns_indices[sample_indices]



selected = train.iloc[sample_rows][['client','route','prod']]



selected_train = pd.DataFrame()

for i,t in selected.iterrows():

    selected_train = pd.concat([selected_train, train.loc[(train.loc[:,'client']==t.client) & (train.loc[:,'route'] == t.route) & (train.loc[:,'prod'] == t.loc['prod']),:]])



pt_sales = selected_train.pivot_table(values='sales_units',index=['client','route','prod'],columns=['week'],aggfunc=np.mean)

pt_returns = selected_train.pivot_table(values='return_units_next_week',index=['client','route','prod'],columns=['week'],aggfunc=np.mean)



plot_weekly_samples(w, h, pt_sales, pt_returns)
with_returns_indices = np.nonzero(train['return_units_next_week'] > 2)[0]

dummy, sample_indices = cross_validation.train_test_split(range(len(with_returns_indices)), test_size=sample_size, train_size=0, random_state=randseed)

sample_rows = with_returns_indices[sample_indices]



selected = train.iloc[sample_rows][['client','route','prod']]



selected_train = pd.DataFrame()

for i,t in selected.iterrows():

    selected_train = pd.concat([selected_train, train.loc[(train.loc[:,'client']==t.client) & (train.loc[:,'route'] == t.route) & (train.loc[:,'prod'] == t.loc['prod']),:]])



pt_sales = selected_train.pivot_table(values='sales_units',index=['client','route','prod'],columns=['week'],aggfunc=np.mean)

pt_returns = selected_train.pivot_table(values='return_units_next_week',index=['client','route','prod'],columns=['week'],aggfunc=np.mean)



plot_weekly_samples(w, h, pt_sales, pt_returns)
# Make a heatmap of sales relative to weeks where returns occurred, based on a sample 

# from the training set



lerandseed=7

n_samples_for_heatmap = 10000

dummy, sample_indices = cross_validation.train_test_split(range(len(with_returns_indices)), test_size=n_samples_for_heatmap, train_size=0, random_state=randseed)

sample_rows = with_returns_indices[sample_indices]

selected = train.iloc[sample_rows][['client','route','prod']]

selected_train = train.loc[train.loc[:,'client'].isin(selected.client) & train.loc[:,'route'].isin(selected.route) & train.loc[:,'prod'].isin(selected.loc[:,'prod']) ,:]



pt_sales = selected_train.pivot_table(values='sales_units',index=['client','route','prod'],columns=['week'],aggfunc=np.mean)

pt_returns = selected_train.pivot_table(values='return_units_next_week',index=['client','route','prod'],columns=['week'],aggfunc=np.mean)



with_one_return = np.nansum(pt_returns.loc[:,3:8]>0,1) == 1

one_return = pt_returns.iloc[with_one_return,:-1].fillna(0).values # Weeks are columns

one_return_sales = pt_sales.iloc[with_one_return,:].fillna(0).values



return_weeks = np.nonzero(one_return)[1]

week_sales = np.zeros([7,7])

for rw in range(6):

    rows_for_week = (return_weeks==rw)

    values = np.nanmean(one_return_sales[rows_for_week],0)

    week_sales[rw,:] = values

week_sales[-1,:] = np.nanmean(one_return_sales,0)

plt.figure()

yticks = [str(a) for a in range(4,10)] + ["Mean for \none-return \nweeks"]

sns.set(font_scale=1.2)

ax = sns.heatmap(week_sales, cmap="Blues", annot=False, xticklabels=range(3,10), yticklabels=yticks)

plt.ylabel('Week return occurred')

plt.xlabel('Week')

cax = plt.gcf().axes[-1]

cax.set_ylabel('Mean units sold for week')

plt.draw()

# Make a heatmap of sales relative to weeks where returns occurred, based on a sample 

# from the training set



one_return_sales_scaled = (one_return_sales-one_return_sales.mean(axis=1, keepdims=True)) / one_return_sales.std(axis=1, keepdims=True)

week_sales = np.zeros([7,7])

for rw in range(6):

    rows_for_week = (return_weeks==rw)

    values = np.nanmean(one_return_sales_scaled[rows_for_week],0)

    week_sales[rw,:] = values

week_sales[-1,:] = np.nanmean(one_return_sales_scaled,0)

plt.figure()

yticks = [str(a) for a in range(4,10)] + ["Mean for \none-return \nweeks"]

sns.set(font_scale=1.2)

ax = sns.heatmap(week_sales, cmap="Greens", annot=False, xticklabels=range(3,10), yticklabels=yticks)

plt.ylabel('Week return occurred')

plt.xlabel('Week')

cax = plt.gcf().axes[-1]

cax.set_ylabel('Mean units sold for week (Normalized)')

plt.draw()