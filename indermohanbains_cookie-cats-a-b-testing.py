# Importing pandas
import pandas as pd

# Reading in the data
df = pd.read_csv ('../input/cokie-cats/cookie_cats.csv')

# Showing the first few rows
df.head ()
# Counting the number of players in each AB group.
df ['version'].value_counts ()
# This command makes plots appear in the notebook
%matplotlib inline

# Counting the number of players for each number of gamerounds 
plot_df = df.groupby ('sum_gamerounds') ['userid'].count ()

# Plotting the distribution of players that played 0 to 100 game rounds
ax = plot_df.head (100).plot ()
ax.set_xlabel("sum_gamerounds")
ax.set_ylabel("userid")
total = len (df ['retention_1'])
# The % of users that came back the day after they installed
my_df = df ['retention_1'].value_counts ()
print (round (my_df[1]*100/total,2))
# Calculating 1-day retention for each AB-group
ret_group = pd.crosstab (df ['version'], df ['retention_1'])
ret_group.columns = ['False', 'True']
ret_group ['Sum'] = ret_group.sum (axis = 1)

ret_group ['perc_1_ret'] = round (ret_group ['True']*100/ ret_group ['Sum'],2)
ret_group ['perc_1_ret']
# Creating an list with bootstrapped means for each AB-group
boot_1d = []
for i in range(500):
    boot_mean = df.sample (frac = 1, replace = True).groupby ('version')['retention_1'].mean ()
    boot_1d.append(boot_mean)
    
# Transforming the list to a DataFrame
boot_1d = pd.DataFrame (boot_1d)

boot_1d.head ()

# A Kernel Density Estimate plot of the bootstrap distributions
boot_1d.plot (kind = 'kde')
# Adding a column with the % difference between the two AB-groups
boot_1d['diff'] = (boot_1d['gate_30'] - boot_1d['gate_40'])/boot_1d['gate_40']*100

# Ploting the bootstrap % difference
ax = boot_1d['diff'].plot (kind = 'kde')
ax.set_xlabel ('differnce of versions')
# Calculating the probability that 1-day retention is greater when the gate is at level 30
prob = len (boot_1d [boot_1d ['diff'] > 0])/500

# Pretty printing the probability
print (prob)
# Calculating 7-day retention for both AB-groups

# Creating a list with bootstrapped means for each AB-group
boot_7d = []
for i in range(500):
    boot_mean_7 = df.sample (frac = 1, replace = True).groupby ('version')['retention_7'].mean ()
    boot_7d.append(boot_mean_7)
    
# Transforming the list to a DataFrame
boot_7d = pd.DataFrame (boot_7d)

boot_7d.head ()

print (round (boot_7d ['gate_40'].mean (),2))
print (round (boot_7d ['gate_30'].mean (),2))
# Creating a list with bootstrapped means for each AB-group
boot_7d = []
for i in range(500):
    boot_mean = df.sample (frac = 1, replace = True).groupby ('version')['retention_7'].mean ()
    boot_7d.append(boot_mean)
    
# Transforming the list to a DataFrame
boot_7d = pd.DataFrame (boot_7d)

# Adding a column with the % difference between the two AB-groups
boot_7d['diff'] = boot_7d ['gate_30'] - boot_7d ['gate_40']

# Ploting the bootstrap % difference
ax = boot_7d['diff'].plot (kind = 'kde')
ax.set_xlabel("% difference in means")

# Calculating the probability that 7-day retention is greater when the gate is at level 30
prob = len (boot_7d[boot_7d ['diff']>0])/500

# Pretty printing the probability
print (prob)
# So, given the data and the bootstrap analysis
# Should we move the gate from level 30 to level 40 ?
move_to_level_40 = False # True or False ?