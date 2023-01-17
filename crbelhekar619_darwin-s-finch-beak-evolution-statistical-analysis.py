# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
plt.style.use('seaborn')
# Loading the data
finch_beaks_1975 =pd.read_csv("../input/darwins-finches-evolution-dataset/finch_beaks_1975.csv")
finch_beaks_2012 =pd.read_csv("../input/darwins-finches-evolution-dataset/finch_beaks_2012.csv")

finch_beaks_1975.head()
finch_beaks_1975 = finch_beaks_1975.drop(['band'], axis = 'columns')
finch_beaks_1975['year'] = "1975"
finch_beaks_1975.rename(columns={'Beak depth, mm' : 'bdepth','Beak length, mm' : 'blength'},inplace=True)

finch_beaks_1975.head()
finch_beaks_2012 = finch_beaks_2012.drop(['band'], axis = 'columns')
finch_beaks_2012['year'] = "2012"
finch_beaks_2012.head()
finch_beaks_both = pd.concat([finch_beaks_1975,finch_beaks_2012]).reset_index(drop=True)
finch_beaks_both.info()
fortis_f = finch_beaks_both[finch_beaks_both.species == 'fortis'].reset_index(drop=True)
scandens_f = finch_beaks_both[finch_beaks_both.species == 'scandens'].reset_index(drop=True)
markers = {'1975': "s", '2012': "X"}
plt.figure(figsize=(8,6))
sns.scatterplot(x= 'blength', y= 'bdepth', style = 'year', markers=markers, data=fortis_f)
plt.show()
markers = {'1975': "s", '2012': "X"}
plt.figure(figsize=(8,6))
sns.scatterplot(x= 'blength', y= 'bdepth', style = 'year', markers=markers, data=scandens_f)

plt.show()
scandens_1975 = finch_beaks_1975[finch_beaks_1975['species']=='scandens']
scandens_2012 = finch_beaks_2012[finch_beaks_2012['species']=='scandens']

# The depths of scandens beak
scandens_beak_depth_1975=scandens_1975['bdepth'].reset_index(drop=True)
scandens_beak_depth_2012=scandens_2012['bdepth'].reset_index(drop=True)

#The lengths of scandens beak
scandens_beak_length_1975=scandens_1975['blength'].reset_index(drop=True)
scandens_beak_length_2012=scandens_2012['blength'].reset_index(drop=True)
# Create bee swarm plot
sns.swarmplot(x='year', y='bdepth', data=scandens_f)

plt.xlabel('year')
plt.ylabel('beak depth (mm)')

plt.show()
# ECDF calculation functiom
def ecdf(x_data) :
    x = np.sort(x_data)
    y = np.arange(1,len(x)+1) / len(x)

    return x,y

# Compute ECDFs
x_1975, y_1975 = ecdf(scandens_beak_depth_1975)
x_2012, y_2012 = ecdf(scandens_beak_depth_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

_ = plt.margins(0.02)
_ = plt.xlabel('Beak Depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

plt.show()
_ = plt.figure(figsize=(8,6))
_ = sns.lmplot(x='blength', y='bdepth', hue = 'year', data=scandens_f)
_ = plt.xlabel('Beak Length (mm)')
_ = plt.ylabel('Beak Depth (mm)')

plt.show()
# Bootstrap replicate function
def bs_reps(data,func,size=1) :

    bs_rep = np.empty(size)

    for i in range(size) :
        bs_rep[i] = func(np.random.choice(data,size=len(data)))
    return bs_rep
# Compute the difference of the both beak depth
mean_diff = np.mean(scandens_beak_depth_2012) - np.mean(scandens_beak_depth_1975)

# Now bootstrap both the depths using mean function for 10000 samples
bs_rep_1975 = bs_reps(scandens_beak_depth_1975,np.mean,size=10000)
bs_rep_2012 = bs_reps(scandens_beak_depth_2012,np.mean,size=10000)

# Compute the difference of the sample means
bootstrap_rep= bs_rep_2012 - bs_rep_1975

# Compute 95% confidence interval
conf_int = np.percentile(bootstrap_rep, [2.5, 97.5])

# Print the results
print('Difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')
#Shifting the two data sets so that they have the same mean 
combined_mean = np.mean(np.concatenate((scandens_beak_depth_1975,scandens_beak_depth_2012)))

bd_1975_shift = scandens_beak_depth_1975 - np.mean(scandens_beak_depth_1975) + combined_mean
bd_2012_shift = scandens_beak_depth_2012 - np.mean(scandens_beak_depth_2012) + combined_mean

bs_rep_1975_shift = bs_reps(bd_1975_shift,np.mean,size=10000)
bs_rep_2012_shift = bs_reps(bd_2012_shift,np.mean,size=10000)

bs_shifted_mean_diff = bs_rep_2012_shift - bs_rep_1975_shift

#p value
p= np.sum(bs_shifted_mean_diff >= mean_diff) / len(bs_shifted_mean_diff)
print("p-value = ",p)
# Make scatter plot of 1975 & 2012 data
_ = sns.scatterplot(x= 'blength', y= 'bdepth',  data=scandens_1975)
_ = sns.scatterplot(x= 'blength', y= 'bdepth',  data=scandens_2012)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

plt.show()
# Linear regression funcion for pair bootstrap

def bs_pair_linreg(x,y,size=1) :
    indices = np.arange(len(x))
       
    slope_reps = np.empty(size)
    intercept_reps = np.empty(size)

    for i in range(size) :
        bs_indices = np.random.choice(indices,size=len(indices))
        bs_x,bs_y = x[bs_indices],y[bs_indices]
        slope_reps[i],intercept_reps[i] = np.polyfit(bs_x,bs_y,1)

    return slope_reps,intercept_reps
# Compute the linear regressions on the original data
slope_1975,intercept_1975 = np.polyfit(scandens_beak_length_1975,scandens_beak_depth_1975,1)
slope_2012,intercept_2012 = np.polyfit(scandens_beak_length_2012,scandens_beak_depth_2012,1)

# Perform pairs bootstrap for the linear regressions
bs_slope_1975,bs_intercept_1975 = bs_pair_linreg(scandens_beak_length_1975,scandens_beak_depth_1975,1000)
bs_slope_2012,bs_intercept_2012 = bs_pair_linreg(scandens_beak_length_2012,scandens_beak_depth_2012,1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_1975,[2.5,97.5])
slope_conf_int_2012 = np.percentile(bs_slope_2012,[2.5,97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_1975,[2.5,97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_1975,[2.5,97.5])

print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)
# Make scatter plot of 1975 & 2012 data
_ = sns.scatterplot(x= 'blength', y= 'bdepth',  data=scandens_1975)
_ = sns.scatterplot(x= 'blength', y= 'bdepth',  data=scandens_2012)

_ = plt.xlabel('Beak Length (mm)')
_ = plt.ylabel('Beak Depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_1975[i] * x + bs_intercept_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_2012[i] * x + bs_intercept_2012[i],
             linewidth=0.5, alpha=0.2, color='green')

plt.margins(0.001)
plt.show()
# Compute length-to-depth ratios
ratio_1975 = scandens_beak_length_1975 / scandens_beak_depth_1975
ratio_2012 = scandens_beak_length_2012 / scandens_beak_depth_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = bs_reps(ratio_2012, np.mean, size=10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)

_ = plt.figure(figsize=(5,4))
_ = plt.plot(mean_ratio_1975, 1975, 'ro', color = 'b')
_ = plt.plot(mean_ratio_2012, 2012, 'ro', color = 'green')

y5 = np.full((10000), 1975)
_ = plt.plot(bs_replicates_1975, y5)

y2 = np.full((10000), 2012)
_ = plt.plot(bs_replicates_2012, y2)

_ = plt.yticks([1975, 2012])
_ = plt.margins(0.6)

plt.show()