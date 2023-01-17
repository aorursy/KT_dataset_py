import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
honey = pd.read_csv('../input/honeyproduction.csv')
honey.head()
print(honey['state'].unique())
print('\n')
print("Number of States in dataset: " + str(len(honey['state'].unique())))
sns.set_style("darkgrid")
honey.groupby('year')['priceperlb'].mean().plot(figsize=(15,4)).set_title('$ Price per lb.')
honey.groupby('year')[['totalprod', 'prodvalue']].sum().plot(figsize=(18,4)).set_title("Total Production (lbs) vs. Production Value ($)")
honey_cor = honey[['totalprod','priceperlb']]
honey_cor.corr()
group= honey.groupby('state')
first_year = honey['year'].min()
last_year= honey['year'].max()

ordered_names = sorted(group['totalprod'].sum().sort_values().index)


fig, axes = plt.subplots(nrows=11, ncols=4, sharex=True, sharey=True, figsize=(18,25))
axes_list = [item for sublist in axes for item in sublist] 

for state in ordered_names:
    selection= group.get_group(state) 
    ax = axes_list.pop(0)
    selection.plot(x='year', y='totalprod', label=state, ax=ax, legend=False)
    ax.set_title(state, fontsize=17)
    ax.tick_params(
        which='both',
        bottom='off',
        left='off',
        right='off',
        top='off',
        labelsize=16 
    )
    ax.grid(linewidth=0.25)
    ax.set_xlim((first_year, last_year))
    ax.set_xlabel("")
    ax.set_xticks((first_year, last_year))
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
for ax in axes_list:
    ax.remove()
    
plt.subplots_adjust(hspace=1)
plt.tight_layout()
pd.pivot_table(honey, index='state', columns='year', values='totalprod', aggfunc=np.sum)
honey.groupby('year')['yieldpercol'].mean().plot(figsize=(15,5)).set_title("Yield Per Colony (lbs)")