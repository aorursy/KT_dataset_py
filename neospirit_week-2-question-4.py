import numpy as np
from scipy.stats import ttest_1samp

#Step 1 set H0 as they has no different of sale amount (mu0 = 0) and H1 is sale amount increase (mu0 > 0)
#Sample that i give you is difference of sale amount already.

samp = [3,0,5,3,9,1,6,5,3,5]
mu0 = 0

#Steo 2 set significant level (alpha)
alpha = 0.01

samp_mean = np.mean(samp)
print(f"Sample mean is {samp_mean}")

#Step 3 selecting t-test because we don't know population variance
tval, pval = ttest_1samp(samp, mu0)

pval /= 2

print(f"t-value is {round(tval, 5)}")
print(f"p-value is {round(pval, 5)}")

# Step 4 conclude result based on p-value
if pval<alpha and tval>0:
    print("Reject null hypothesis")
    
else:
    print("Fail to reject null hypothesis")