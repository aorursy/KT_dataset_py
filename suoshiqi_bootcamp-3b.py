import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
portreturn = pd.read_csv('../input/returnnew/return1.csv', parse_dates=['Date'])
portreturn.info()
portreturn.head(10)
plt.hist(portreturn['diff'], bins = 20)
plt.show()
from scipy import stats

sample_mean = np.mean(portreturn['diff'])
sample_std = np.std(portreturn['diff'], ddof=1)
sample_size = portreturn['diff'].size
sample_standard_error = sample_std/np.sqrt(sample_size)

z = (sample_mean-0)/(sample_std/np.sqrt(sample_size))

pval = stats.norm.sf(z)  # one-sided pvalue 

print("Point estimate : " + str(sample_mean))
print("Standard error :" + str(sample_standard_error))
print("Z-statistic : " + str(z))

# interpret p-value 
alpha = 0.05
print("P-value : " + str(pval))
if pval <= alpha: 
    print('The portfolio return is improved by diversification (reject H0)') 
else: 
    print('The portfolio return is not improved by diversification (fail to reject H0)')
confidence_level = 0.95

confidence_interval = stats.norm.interval(confidence_level,sample_mean, sample_standard_error)

print("Point estimate : " + str(sample_mean))
print("Confidence interval (0.025, 0.975) : " + str(confidence_interval))