# Import relevant packages
import matplotlib.pyplot as plt # For plot configuration
import numpy as np              # For numerical operations
import pandas as pd             # For database management
import seaborn as sns           # For plotting data easily

import warnings
warnings.filterwarnings("ignore")

sns.set()                       # Set seaborn style so plots are nice!
# Read the file with the `.read_csv()` method
df = pd.read_csv('../input/Pokemon.csv')
n_rows, n_cols = df.shape
print('The dataset has {} rows and {} columns.'.format(n_rows, n_cols))
df.head()
columns = df.columns
print('Columns names: {}.'.format(columns.tolist()))
# Method 1: step by step
#    1. Create a copy of the df
#    2. Assign the new column name to the old one.
#    3. Delete the old column.
# Notice jhow this method reorders the columns!
df_slow = df.copy()
df_slow['Num'] = df['#']
to_rename = columns[[2,3,8,9]]
for col in to_rename:
    if '.' in col:
        cola = col.replace('. ', '_')
    else:
        cola = col.replace(' ', '_')
    df_slow[cola] = df[col]
df_slow = df_slow.drop(columns=columns[[0,2,3,8,9]])
print(df_slow.columns)


# Method 2: built-in function
#    1. Create a copy of the df
#    2. Create the mapping
#    3. Apply the function
df_good = df.copy()
mapper = {'#': 'Num'}
mapper.update({col: col.replace(' ','_') if '.' not in col else col.replace('. ', '_') for col in to_rename.tolist()})
df_good.rename(columns=mapper, inplace=True)
print(df_good.columns)
# 1. We can use the built-in method of pandas "unique()"
names = df.Name.unique()

# 2. Print a few names:
print('Some Pokémon names are: '+('"{}", '*3).format(*np.random.choice(names, 3))+
      'and "{}".'.format(*np.random.choice(names,1)))
print('The amount of unique Pokémons is {}.'.format(len(names)))
df_good.describe()
# Let's store these variables since we will use it later on
stats = df_good.columns[4:-3]
sns.pairplot(data=df_good.iloc[:,5:-2], 
             kind='reg')
sns.pairplot(data=df_good.iloc[:,5:-1],       # Take all stats and the Generation column
             diag_kind='kde',                 # Instead of a histogram, plot a KDE
             kind='reg',                      # Plot regression lines as well
             plot_kws=dict(truncate=True),    # Autoadjust the axis to the data
             vars=df_good.iloc[:,5:-2],       # Use only the stats columns for the plots...
             hue='Generation')                # ...and Generation as the separation variable.
for category in ['Type_1', 'Type_2', 'Generation', 'Legendary']:
    print('"{}" has {} missing values. The rest are:'.format(category, df_good[category].isnull().sum()))
    
    # 1.a Simple, built-in method "value_counts()"
    types_simple = df_good[category].value_counts()
    # 1.b MapReduce strategy: group by type and count the instances, keeping only the number (column 'Num')
    types_group = df_good.groupby(category).count()['Num']
    
    # Both yield the same counting *BUT* the ordering is different:
    #    - groupby: alphabetical order.
    #    - value_counts: frequency order.
    # Either way, we can print any of the results (personally I like the frequency order):
    print(types_simple, end='\n\n')
# CLEANING OUTLIERS ---- ongoing
# CATEGORY FIXING ------ todo
# ONE-HOT LABELING ----- todo
# PLOTS! --------------- put them everywhere
def outlier_check(data):
    """
    This function obtains a pandas Series and plots the distribution
    of the variable, along with bars that indicate the upper (or 
    lower) 5% of the data.
    """
    # 1. First computations of maximum, mean and standard deviation
    M = max(data)
    m, s = np.mean(data), np.std(data)
    
    # 2. L(ow) and H(igh) filters.
    L, H = m-2*s, m+2*s
    
    # 3. Plot congiguration
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(5)
    ax.set_ylim([0,0.025])
    ax.set_xlim([0,M])
    ax.set_title('"{}" outlier detection'.format(data.name))
    
    # 3.1 Draw a vertical line in the upper limit and shade it
    ax.vlines(H, 0, 0.025, color='red', linestyle='dashed')
    ax.fill_between(x=[H,M], y1=0.025, color='red', alpha=.05)
    
    # 3.2 Similarly for the lower limit
    ax.vlines(L, 0, 0.025, color='red', linestyle='dashed')
    ax.fill_between(x=[0,L], y1=0.025, color='red', alpha=.05)
    
    # 4. Plot the distribution
    sns.distplot(data, ax=ax)
    
    # 5. Return the indices of the outliers
    return data[(data<L) | (data>H)].index
df_good['Outlier'] = np.zeros((len(df_good),1))
for var in stats:
    df_good.loc[outlier_check(df_good[var]),'Outlier'] = 1
len(df_good.loc[df_good.Outlier==1, :])
legendary = len(df_good[df_good.Legendary==True])
legendary_outliers = len(df_good.loc[(df_good.Outlier==1) & (df_good.Legendary==True), :])
print('There are {} legendary Pokémon, out of those {} have outlier stats.'.format(legendary, legendary_outliers))
normal_outliers = len(df_good.loc[(df_good.Outlier==1) & (df_good.Legendary==False), :])
mega_outliers = len([pokemon for pokemon in df_good.loc[(df_good.Outlier==1) & (df_good.Legendary==False), 'Name'] 
                 if 'Mega' in pokemon])

print('There are {} normal outlier Pokémon, out of those {} are "Mega" versions.'.format(normal_outliers, mega_outliers))





sns.jointplot(df_good.iloc[:,0], df_good.iloc[:,5], kind='reg')




