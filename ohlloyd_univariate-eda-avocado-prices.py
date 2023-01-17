import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as sp
#Read in data

avocadata = pd.read_csv("../input/avocado.csv")

#remove duplicate information
bools = np.any([(avocadata['region'] != "TotalUS")], axis=0)
avocadata = avocadata.loc[bools]

#select the price vector
arrayvocado = avocadata["AveragePrice"]
#Graphical EDA

plt.subplot(2,1,1)
sns.distplot(arrayvocado, axlabel = False, rug = True,
             kde_kws={"label": "Kernel Density", "color" : 'k'},
             hist_kws={"label": "Histogram"},
             rug_kws={"label": "Rug plot"})
plt.ylabel("Density")
plt.yticks([0,0.25,0.5,0.75,1])
plt.xticks([])

plt.subplot(2,1,2)
plt.boxplot(arrayvocado, vert = False)

plt.xlabel("Average avocado price ($)")
plt.xlim(0,3.5)
plt.yticks([])
plt.show()
#Data could be multi-modal, check with histogram of log-price.

sns.distplot((np.log(arrayvocado)))
plt.xlabel("Log of average avocado price (log($))")
plt.ylabel("Density")
plt.show()
#Non-graphical EDA

print("Mean = " + str(np.mean(arrayvocado)))
print("Median = " + str(np.median(arrayvocado)))
print("Mode = " + str(sp.mode(arrayvocado)[0][0]))

#second central moment, variance
print("\nBiased variance = " + str(np.var(arrayvocado)))
print("Unbiased variance = " + str(np.var(arrayvocado, ddof=1)))

#third standardized moment, skew
sd = np.sqrt(np.var(arrayvocado))
skew = sp.stats.moment(arrayvocado,3)/(sd**3)
print("\nSkew = " + str(skew))

#fourth standardized moment, kurtosis
kurtosis = sp.stats.moment(arrayvocado,4)/(sd**4)
print("Excess kurtosis = " + str(kurtosis - 3))