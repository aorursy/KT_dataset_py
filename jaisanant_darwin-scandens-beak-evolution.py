import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading    
finch_beaks_1975=pd.read_csv("../input/finch_beaks_1975.csv")
finch_beaks_2012=pd.read_csv("../input/finch_beaks_2012.csv")

finch_beaks_1975_df=pd.DataFrame(finch_beaks_1975)
finch_beaks_2012_df=pd.DataFrame(finch_beaks_2012)

finch_beaks_1975_gb=finch_beaks_1975_df.groupby(finch_beaks_1975_df['species'])
finch_beaks_1975_scandens=finch_beaks_1975_gb.get_group('scandens').reset_index(drop=True)

finch_beaks_2012_gb=finch_beaks_2012_df.groupby(finch_beaks_2012_df['species'])
finch_beaks_2012_scandens=finch_beaks_2012_gb.get_group('scandens').reset_index(drop=True)

#scandens beak depths
scandens_beak_depth_1975=finch_beaks_1975_scandens['Beak depth, mm']
scandens_beak_depth_2012=finch_beaks_2012_scandens['bdepth']

#scandens beak lengths
scandens_beak_length_1975=finch_beaks_1975_scandens['Beak length, mm']
scandens_beak_length_2012=finch_beaks_2012_scandens['blength']


s_bd_both = pd.concat([finch_beaks_1975_scandens['Beak depth, mm'],finch_beaks_2012_scandens['bdepth']],axis=1).reset_index(drop=True)
s_bd_both.rename(columns={'Beak depth, mm' : '1975','bdepth' : '2012'},inplace=True)

#EDA of beak depths of Darwin's finches
#visualization of beak depth

#1. Swarmplot
sns.set()
_=sns.swarmplot(data=s_bd_both)
_=plt.xlabel('year')
_=plt.ylabel('beak depth (mm)')
'''It is kinda hard to see if there is a clear difference between the 1975 and 2012 data set but it appears 
as though the mean of the 2012 data set might be slightly higher, and it might have a bigger variance.'''

# ecdf calculation functiom
def ecdf(x_data) :
    x=np.sort(x_data)
    y=np.arange(1,len(x)+1) / len(x)

    return x,y

#bootstarp 1D function
def bs_rep_1d(data,func) :
    return func(np.random.choice(data,size=len(data)))

# bottstarp replicate function
def bs_reps(data,func,size=1) :

    bs_rep=np.empty(size)

    for i in range(size) :
        bs_rep[i]=bs_rep_1d(data,func)
    return bs_rep

# linear regression funcion for pair bootstrap

def bs_pair_linreg(x,y,size=1) :
    indices=np.arange(len(x))
       
    slope_reps = np.empty(size)
    intercept_reps = np.empty(size)

    for i in range(size) :
        bs_indices = np.random.choice(indices,size=len(indices))
        bs_x,bs_y = x[bs_indices],y[bs_indices]
        slope_reps[i],intercept_reps[i] = np.polyfit(bs_x,bs_y,1)

    return slope_reps,intercept_reps

#ECDF becasuse it gives better understanding 
x_1975,y_1975 = ecdf(scandens_beak_depth_1975)
x_2012,y_2012 = ecdf(scandens_beak_depth_2012)

_=plt.scatter(x_1975,y_1975)
_=plt.scatter(x_2012,y_2012)

_=plt.xlabel('Beak depth (mm) ' )
_=plt.ylabel('ECDF')
_=plt.legend(('1975','2012'),loc='lower right')
'''The differences is clear in the ECDF. The mean is larger in the 2012 data, and the variance as well'''
#Parameter estimates of beak depths

mean_diff = np.mean(scandens_beak_depth_2012) - np.mean(scandens_beak_depth_1975)

bs_rep_1975 = bs_reps(scandens_beak_depth_1975,np.mean,size=10000)
bs_rep_2012 = bs_reps(scandens_beak_depth_2012,np.mean,size=10000)

bootstrap_rep= bs_rep_2012 - bs_rep_1975

#Confidence Intervl 95%

conf_int=np.percentile(bootstrap_rep,[2.5,97.5])

print('[+] Difference of means in beak depth = ', mean_diff, 'mm')
print('[+] 95% confidence interval = ', conf_int, 'mm')
'''The plot of the ECDF and determination of the confidence interval make it clear that the beaks of Geospiza 
scandens have gotten deeper. But it might be possible that this effect is just due to random chance? what is 
the probability that we would get the observed difference in mean beak depth if the means were the same?'''

#hypothesis test
#shifting the two data sets so that they have the same mean 
combined_mean = np.mean(np.concatenate((scandens_beak_depth_1975,scandens_beak_depth_2012)))

bd_1975_shift = scandens_beak_depth_1975 - np.mean(scandens_beak_depth_1975) + combined_mean
bd_2012_shift = scandens_beak_depth_2012 - np.mean(scandens_beak_depth_2012) + combined_mean

bs_rep_1975_shift = bs_reps(bd_1975_shift,np.mean,size=10000)
bs_rep_2012_shift = bs_reps(bd_2012_shift,np.mean,size=10000)

bs_shifted_mean_diff = bs_rep_2012_shift - bs_rep_1975_shift

#p value

p= np.sum(bs_shifted_mean_diff >= mean_diff) / len(bs_shifted_mean_diff)
print("[+] p-value = ",p)
'''We get a p-value of ~ 0.0038, which suggests that there is a statistically significant difference.In the previous code, we got 
a difference of 0.2 mm between the means. Statistically, Change by 0.2 mm in 37 years is substantial by 
evolutionary standards.'''
# EDA of beak length and depth
plt.figure('Beak depth vs Beak length')
_ = plt.scatter(x=scandens_beak_length_1975,y=scandens_beak_depth_1975,color='blue')
_ = plt.scatter(x=scandens_beak_length_2012,y=scandens_beak_depth_2012,color='red')
_ = plt.xlabel('Beak length(mm)')
_ = plt.ylabel('Beak depth (mm) ')
_=plt.legend(('1975','2012'),loc='upper left')
'''Looking at the plot, we see that beaks got deeper (the red points are higher up in the y-direction),
but not really longer. If anything, they got a bit shorter, since the red dots are to the left of the blue dots. So, it does not
look like the beaks kept the same shape; they became shorter and deeper.'''

# Linear Regression

slope_1975,intercept_1975 = np.polyfit(scandens_beak_length_1975,scandens_beak_depth_1975,1)
slope_2012,intercept_2012 = np.polyfit(scandens_beak_length_2012,scandens_beak_depth_2012,1)

bs_slope_1975,bs_intercept_1975 = bs_pair_linreg(scandens_beak_length_1975,scandens_beak_depth_1975,size=1000)
bs_slope_2012,bs_intercept_2012 = bs_pair_linreg(scandens_beak_length_2012,scandens_beak_depth_2012,size=1000)

slope_conf_int_1975 = np.percentile(bs_slope_1975,[2.5,97.5])
slope_conf_int_2012 = np.percentile(bs_slope_2012,[2.5,97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_1975,[2.5,97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_1975,[2.5,97.5])

print("[+] Slope 1975 : ",slope_1975, 'Confidence interval : ',slope_conf_int_1975)
print("[+] Intercept 1975 : ",intercept_1975, 'Confidence interval : ',intercept_conf_int_1975)
print("[+] Slope 2012 : ",slope_2012, 'Confidence interval : ',slope_conf_int_2012)
print("[+] Intercept 2012 : ",intercept_2012, 'Confidence interval : ',intercept_conf_int_2012)



'''The linear regressions showed information about the beak geometry. The slope was the same in 1975 and 2012,
that means for every millimeter gained in beak length, the birds gained about half a millimeter in depth in 
both years.
However,for the shape of the beak, we want to compare the ratio of beak length to beak depth.'''

ratio_1975 = scandens_beak_length_1975/scandens_beak_depth_1975
ratio_2012 = scandens_beak_length_2012/scandens_beak_depth_2012

mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

ratio_reps_1975 = bs_reps(ratio_1975,np.mean,size=10000)
ratio_reps_2012 = bs_reps(ratio_2012,np.mean,size=10000)

ratio_conf_int_1975 = np.percentile(ratio_reps_1975,[.5,99.5])
ratio_conf_int_2012 = np.percentile(ratio_reps_2012,[.5,99.5])

print('[+] 1975 mean beak length to depth ratio : ',mean_ratio_1975, ' Confidence Interval (99%) : ',ratio_conf_int_1975)
print('[+] 2012 mean beak length to depth ratio : ',mean_ratio_2012, ' Confidence Interval (99%) : ',ratio_conf_int_2012)
'''The mean beak length to depth ratio is ~1.58 in 1975 and 1.47 in 2012. The low end of the 1975 99% 
confidence interval was ~1.56 mm and the high end of the 99% confidence interval in 2012 was ~1.49 mm.
The mean beak length-to-depth ratio decreased by about ~0.1mm, or 7%, from 1975 to 2012. 

Conclusion :The 99% confidence intervals are not even close to overlapping, so this is a real change. 
The beak shape changed.'''