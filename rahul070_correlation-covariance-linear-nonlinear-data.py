# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#We will generate 1,000 samples of two two variables with a strong positive correlation.
from numpy.random import randn
import matplotlib.pyplot as plt
x1=20 * randn(1000)+100
#print(x1) #will print all the random 1000 numbers
x2=x1+(10 * randn(1000)+50)
#print(x2) #will print all the random 1000 numbers
#plot
plt.scatter(x1,x2)
plt.show() # you will expect as a linear relationship between the x1 and x2
sns.distplot(x1)
sns.distplot(x2)
plt.show
from scipy.stats import pearsonr
coefp,pvalue=pearsonr(x1,x2)
print("Pearson Value",coefp,pvalue)
import numpy as np
from numpy.random import randn
import seaborn as sns
x3=randn(1000)
x4=10* randn(1000)
import matplotlib.pyplot as plt
plt.scatter(x3,x4)
plt.show()
sns.distplot(x3)
sns.distplot(x4)
plt.show()
from scipy.stats import pearsonr
coef1,p=pearsonr(x3,x4)
print("Pearsonr Value",coef1,p)
from scipy.stats import spearmanr
coef2,p2=spearmanr(x3,x4)
print("spearmanr value",coef2,p2)
from scipy.stats import kendalltau
coef3,p3=kendalltau(x3,x4)
print("KendallTau value",coef3,p3)
# Applying Spearman's Coeffecient in Uniform Distribution
import numpy as np
import seaborn as sns
s = np.random.uniform(-1,0,1000)
d = np.random.uniform(-1,0,1000)
import matplotlib.pyplot as plt
plt.scatter(s,d)
plt.show()
sns.distplot(s)
sns.distplot(d)
plt.show()
# calculate spearman's correlation
from scipy.stats import spearmanr
coef,p = spearmanr(s,d)
print(coef,p)
#Comparison of Pearson and Spearman's Coeffecient in Non linear or random Distribution
from numpy.random import randn
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
x6=10 * randn(1000) + 20
x7= randn(1000) + 10
coef,p=spearmanr(x6,x7)
pears,p=pearsonr(x6,x7)
print("Spearmans Coeffecient",coef,p)
print("Pearson's Coeffecient",pears,p)
plt.scatter(x6,x7)
plt.show()
#--------#
import seaborn as sns
sns.distplot(x6)
sns.distplot(x7)
plt.show
#Comparison of Pearson and Spearman's Coeffecient in linear or Normal/gaussian Distribution
from numpy.random import randn
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
x6=10 * randn(1000) + 20
x7= x6+randn(1000) + 10
coef,p=spearmanr(x6,x7)
pears,p=pearsonr(x6,x7)
print("Spearmans Coeffecient",coef,p)
print("Pearson's Coeffecient",pears,p)
plt.scatter(x6,x7)
plt.show()
#--------#
import seaborn as sns
sns.distplot(x6)
sns.distplot(x7)
plt.show
# Kendalls Tau Coeffecient of correlation
from scipy.stats import kendalltau
from numpy.random import randn
import matplotlib.pyplot as plt
import seaborn as sns
var1=10 + randn(1000)
var2= randn(1000)
coef,p=kendalltau(var1,var2)
alpha=0.05
print(coef,p)
plt.scatter(var1,var2)
sns.distplot(var1)
sns.distplot(var2)
plt.show()
if p>alpha:
    print("samples are not correlated")
else:
    print("samples are highly correlated")
from scipy.stats import pearsonr
coef1,p1=pearsonr(var1,var2)
print("Pearsonr value",coef1,p1)
from scipy.stats import spearmanr
coef2,p2=spearmanr(var1,var2)
print("spearmanr value",coef2,p2)
from scipy.stats import kendalltau
coef3,p3=kendalltau(var1,var2)
print("KendallTau value",coef3,p3)
#We will generate 1,000 samples of two two variables with a strong positive correlation.
from numpy.random import randn
import matplotlib.pyplot as plt
x1=20 * randn(1000)+100
#print(x1) #will print all the random 1000 numbers
x2=x1+(10 * randn(1000)+50)
#print(x2) #will print all the random 1000 numbers
#plot
#plt.scatter(x1,x2)
#plt.show() # you will expect as a linear relationship between the x1 and x2
from numpy import cov
cov(x1,x2)
from scipy.stats import pearsonr
corr=pearsonr(x1,x2)
print("Pearson's Correlation Coeffecient:%.3f",corr)
