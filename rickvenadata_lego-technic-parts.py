# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import parts inventories and filter out spare piece counts

inv_parts = pd.read_csv('../input/inventory_parts.csv')

inv_parts = inv_parts.loc[inv_parts['is_spare'].isin(['f'])]

inv_parts.head()
parts = pd.read_csv('../input/parts.csv')

parts.head()
# import sets and filter subset of Technic sets

sets = pd.read_csv('../input/sets.csv')

technic_sets = sets.loc[sets['year'].isin([2013,2014,2015,2016,2017]) & sets['theme_id'].isin([1,11,12])]

technic_sets.head()
# plot sets from DataFrame having > 400 parts

import matplotlib.pyplot

technic_sets_plt = technic_sets[['set_num','num_parts']].loc[technic_sets['num_parts'] > 400]

technic_sets_plt.plot(kind="bar")
# import inventory list and perform DataFrame inner join with Technic sets

inventories = pd.read_csv('../input/inventories.csv')

technic_inventories = technic_sets.set_index('set_num').join(inventories[['id','set_num']].set_index('set_num'))

technic_inventories.head()
# show basic stats

technic_inventories.describe()
# join Technic inventories to parts lists

technic_inv_parts = technic_inventories.set_index('id').join(inv_parts.set_index('inventory_id'))

technic_inv_parts.head()
technic_inv_parts.describe()
# total number of pieces if you owned one each of every set

technic_inv_parts['quantity'].sum()
# totals by part if you owned all of these sets

part_sums = technic_inv_parts.groupby('part_num')['quantity'].sum()

part_sums.head()
# add the count of sets each part appears in

parts_sets = technic_inv_parts.groupby('part_num')['name'].count()

parts_sets = pd.DataFrame(parts_sets)

parts_sets = parts_sets.rename(columns={'name': 'setsfoundin'})

parts_matrix = parts_sets.join(part_sums)

include_category = [12,52,40,29,26,25,22,55,51,53,30,46,54,8,45,44]

part_matrix_names = parts_matrix.join(parts.set_index('part_num'))

part_matrix_names = part_matrix_names.loc[part_matrix_names['part_cat_id'].isin(include_category)]

part_matrix_names.head()
part_matrix_names.to_csv('part_matrix_names.csv')
# make a scatterplot

# better illustrates relationship between quantity of pieces and how many sets they appear in 

import seaborn as sea

partscatter = parts_matrix.loc[(parts_matrix['quantity'] > 15) & (parts_matrix['quantity'] < 160)] 

sea.regplot(x=partscatter["quantity"], y=partscatter["setsfoundin"])
# hexbin addresses the overplotting further.

# Color ramp indicates number of data points (Technic parts) in each position on the cart

import matplotlib.pyplot as plt

nbins = 48

#plt.hexbin(x=partscatter["quantity"], y=partscatter["setsfoundin"], gridsize=nbins, cmap=plt.cm.BuGn_r)

plt.hexbin(x=partscatter["quantity"], y=partscatter["setsfoundin"], gridsize=nbins, cmap=plt.cm.PuRd)

plt.rcParams["figure.figsize"] = [12,9]

plt.xlabel('total quantity of part', fontweight='bold', color = 'purple', fontsize='14', horizontalalignment='center')

plt.ylabel('sets part is found in', fontweight='bold', color = 'purple', fontsize='14', horizontalalignment='center')
