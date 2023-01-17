
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for scatterplots
import statsmodels.formula.api as sm #for regression

dat=pd.read_csv("../input/londondeprivation.csv", names=['borough','dep','np','ne','ps','pb','pe'], delim_whitespace=True)
print(dat.head())
print(len(dat))
print(dat.corr()) #gives correlation matrix - drops column that is not numerical
print("Correlation between dep and ne is {0:.3f}".format(dat['dep'].corr(dat['ne']))) #to get specific correlation.
print("Correlation netween np and ne is {0:.3f}".format(dat['np'].corr(dat['ne'])))
fig, (ax1,ax2)=plt.subplots(nrows=1,ncols=2)
ax1.scatter(dat['ne'],dat['dep']) #had problem accessing using attributes eg dat.ne -may be cuz of use of indexing in q1?
ax1.set_xlabel('ne')
ax1.set_ylabel('dep')
ax2.scatter(dat['ne'],dat['np'])
ax2.set_xlabel('ne')
ax2.set_ylabel('np') #needs to be shifted right
plt.show()
#ne on np
mA=sm.ols('ne~np',data=dat).fit()
mA.summary()
# ne on dep
mB = sm.ols('ne~dep',data=dat).fit()
mB.summary()

#ne on dep and np
mC=sm.ols('ne~np+dep',data=dat).fit()
mC.summary()
print("Prediction for Greenwich {0}".format(mC.predict(pd.DataFrame({'dep':[37.87],'np':[92.7]}))))
print("Computing it manually {0}".format(mC.params[0]+mC.params[1]*92.7+mC.params[2]*37.87))