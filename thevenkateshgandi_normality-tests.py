import numpy as np 

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import seaborn as sns
from math import sqrt

n=300 # Sample size

data=np.random.uniform(low=1,high=10,size=n) # Sample

#data=np.random.normal(loc=0,scale=10,size=n)

bin_value=int(sqrt(len(data))) #choosing the number of bins to draw histogram

sns.distplot(data,bins=bin_value);

plt.xlabel("Sample Data",size=15);

plt.ylabel("Density",size=15);

plt.title("Histogram",size=20);

plt.show()
import scipy.stats as stats

stats.probplot(data, dist="norm", plot=plt);
from scipy.stats import shapiro

Shapiro_statistic_value, Shapiro_p_value = shapiro(data)

print(Shapiro_statistic_value, Shapiro_p_value)
standardized_data=(data-np.mean(data))/np.std(data)

ks_statistic,ks_p_value=stats.kstest(standardized_data, 'norm')#Compared with standard normal distribution

ks_statistic,ks_p_value
# By now, we had known the value of maximum distance. If you are intrested in knowing where it is located.

# Consider this code



probs=np.arange(1, n+1)/n # n - sample size

stzd_data=np.sort(standardized_data)

theoretical_values=stats.norm.ppf(probs)

est_probs=np.interp(stzd_data,theoretical_values,probs)

max_id=np.argmax((probs-est_probs)**2)

plt.plot(theoretical_values,probs,label='Standard Normal CDF')

plt.plot(stzd_data,probs,label='Sample EDF')

plt.axvline(stzd_data[max_id], color="red", linestyle="dashed", alpha=0.4)

plt.plot([stzd_data[max_id], stzd_data[max_id]], [probs[max_id], est_probs[max_id]], color="red")

plt.xlabel('X',size=15)

plt.ylabel('CDF(x)',size=15)

plt.title('Illustration of the Kolmogorov–Smirnov statistic',size=15)

plt.legend()

plt.show()
import statsmodels.api as sm

Lilliefors_statistic,Lilliefors_p_value=sm.stats.diagnostic.lilliefors(data,'norm')

Lilliefors_statistic,Lilliefors_p_value
stats.kstest(data,'norm',args=(np.mean(data),np.std(data,ddof=1))) 

#One can obsserve the value of test statistic is same for KS test,Lilliefors test, But the p-values are different.
!pip install scikit-gof
from skgof import cvm_test

CvM_statistic,CvM_pvalue=cvm_test(standardized_data,'norm')

CvM_statistic,CvM_pvalue
AD_result = stats.anderson(data) #The sample data will standardized and compare with N(0,1)

print(AD_result)

print('-'*20)

for i in range(len(AD_result.critical_values)):

    sl, cv = AD_result.significance_level[i], AD_result.critical_values[i]

    if AD_result.statistic < AD_result.critical_values[i]:

        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

    else:

        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
AD_result.statistic,AD_result.significance_level,AD_result.critical_values
# To conduct AD test with unknown mean and variance use this method 

# statsmodels.stats.diagnostic.normal_ad() the estimates for parameters will obtained from sample
import pandas as pd



df=pd.DataFrame([{'SW (n<5000))':Shapiro_p_value,'KS':ks_p_value, 'L':Lilliefors_p_value, 'CvM':CvM_pvalue}])

AD_SL=pd.Series(AD_result.significance_level)

AD_CV=pd.Series(AD_result.critical_values)

AD_statistic=pd.Series([AD_result.statistic]*len(AD_result.critical_values))

AD_statistic

df_AD=pd.DataFrame({'AD_statistic':AD_statistic,'AD_CV':AD_CV, 'AD_SL':AD_SL})



print("AD Test Results")

df_AD.style.apply(lambda x: ["background: green" 

                             if (i>=2 and (x.iloc[0] < x.iloc[1]))

                             else ("background: red" if (i>=2 and (x.iloc[0] >= x.iloc[1])) else "")

                             for i,v in enumerate(x)], axis = 1)



#Sorry if you are reading this in github, the colours are not highlighted(try this in your code editor)
#At alpha=0.01

df.style.apply(lambda x: ["background: green" if v > 0.01 else "background: red" for v in x], axis = 1)
#At alpha=0.05

df.style.apply(lambda x: ["background: green" if v > 0.05 else "background: red" for v in x], axis = 1)
#At alpha=0.1

df.style.apply(lambda x: ["background: green" if v > 0.1 else "background: red" for v in x], axis = 1)