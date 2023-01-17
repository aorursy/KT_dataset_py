# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from scipy import stats

import numpy as np



s = [26, 15, 8, 44, 26, 13, 38, 24, 17, 29]



print(np.mean(s))

print(np.median(s))

print(stats.mode(s))

print(np.percentile(s, [25,75], interpolation='lower'))

print(stats.iqr(s, rng=(25, 75), interpolation='lower'))

print(stats.skew(s))

print(stats.kurtosis(s))
from scipy import stats

import numpy as np



x = stats.norm(loc=32, scale=4.5)

np.random.seed(1)

y = x.pdf(np.random.rand(100))

print(y)

print(np.mean(y) - 32)
from scipy import stats

import numpy as np

np.random.seed(1)  

x = stats.binom.rvs(n=1, p=0.5, size=10000)      

print(np.bincount(x)[0])
from scipy import stats

import numpy as np



s1 = [45, 38, 52, 48, 25, 39, 51, 46, 55, 46]

s2 = [34, 22, 15, 27, 37, 41, 24, 19, 26, 36]



t, p = stats.ttest_ind(s1,s2)



print(t)

print(p)
from scipy import stats

import numpy as np



s1 = [12, 7, 3, 11, 8, 5, 14, 7, 9, 10]

s2 = [8, 7, 4, 14, 6, 7, 12, 5, 5, 8]



t,p = stats.ttest_rel(s1,s2)



print(t)

print(p)
import numpy as np



y = np.array([1, 2, 3, 4, 5])

x1 = np.array([6, 7, 8, 9, 10])

x2 = np.array([11, 12, 13, 14, 15])

X = np.vstack([np.ones(5), x1, x2, x1*x2]).T



print(y)

print(X)
import patsy

import numpy as np



y = np.array([1, 2, 3, 4, 5])

x1 = np.array([6, 7, 8, 9, 10])

x2 = np.array([11, 12, 13, 14, 15])

data = {'y':y, 'x1':x1, 'x2':x2}



y, X = patsy.dmatrices('y ~ 1 + x1 + x2 + x1*x2', data)



print(y)

print(X)
import statsmodels.api as sm

import statsmodels.formula.api as smf

import numpy as np

import patsy



mtcar_data = sm.datasets.get_rdataset('mtcars')

df = mtcar_data.data

wt = np.array(df.wt)

mpg =  np.array(df.mpg)

data = {'wt':wt, 'mpg':mpg}

linear_model1 = smf.ols('wt~ mpg', data)

linear_result1 = linear_model1.fit()

print(linear_result1.rsquared)
import statsmodels.api as sm

import statsmodels.formula.api as smf

import numpy as np

import patsy



mtcar_data = sm.datasets.get_rdataset('mtcars')

df = mtcar_data.data

wt = np.array(df.wt)

mpg =  np.array(df.mpg)

data = {'wt':wt, 'mpg':mpg}

linear_model1 = smf.ols('np.log(wt)~ np.log(mpg)', data)

linear_result1 = linear_model1.fit()

print(linear_result1.rsquared)
import statsmodels.api as sm

import statsmodels.formula.api as smf

import numpy as np

import pandas as pd



iris = sm.datasets.get_rdataset("iris").data 

iris.info()

iris.Species.unique() 



iris_subset = iris[(iris.Species == "versicolor") | (iris.Species == "virginica")].copy()

print(iris_subset.Species.unique())



iris_subset.Species = iris_subset.Species.map({"versicolor": 1, "virginica": 0}) 

data = {'Sepal_Length':iris_subset['Sepal.Length'] ,'Sepal_Width':iris_subset['Sepal.Width'],'Species':iris_subset['Species'],'Petal_Length':iris_subset['Petal.Length'],'Petal_Width':iris_subset['Petal.Width']}

model = smf.logit("Species ~ Petal_Length + Petal_Width", data=data)

result = model.fit() 



print(result.summary())

import statsmodels.api as sm

import statsmodels.formula.api as smf

import numpy as np

import pandas as pd



bio_df = sm.datasets.get_rdataset('biopsy', package='MASS',).data 

bio_df = bio_df.rename(columns={"class": "Class"})

bio_df.Class = bio_df.Class.map({"malignant": 1, "benign": 0}) 

data = {'V1':bio_df['V1'],'Class':bio_df['Class']}

model = smf.logit("Class ~ V1", data=data)

result = model.fit() 

print(result.prsquared)
import pandas as pd

import statsmodels.formula.api as smf



awards_df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/poisson_sim.csv")

poisson_model = smf.poisson('num_awards ~ math + C(prog)', awards_df)

poisson_model_result = poisson_model.fit()

import statsmodels.api as sm

import statsmodels.formula.api as smf

import numpy as np

import pandas as pd



ins_df = sm.datasets.get_rdataset('Insurance', package='MASS',).data 

data = {'holders':ins_df['Holders'],'claims':ins_df['Claims']}

poisson_model = smf.poisson('claims ~ np.log(holders)', data)

poisson_model_result = poisson_model.fit()

print(poisson_model_result.resid.sum())

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.stats import anova

from statsmodels import stats







icecream = sm.datasets.get_rdataset("Icecream", "Ecdat")

icecream_data = icecream.data

model1 = smf.ols('cons ~ temp', icecream_data).fit()

print(anova.anova_lm(model1))

model2 = smf.ols('cons ~ income + temp', icecream_data).fit()

print(anova.anova_lm(model2))

# print(stats.sf(31.81, 2, 27))

print(anova.anova_lm(model1, model2))
import statsmodels.api as sm

import statsmodels.formula.api as smf

import numpy as np

from statsmodels.stats import anova

import patsy



mtcar_data = sm.datasets.get_rdataset('mtcars')

df = mtcar_data.data

wt = np.array(df.wt)

mpg =  np.array(df.mpg)

data = {'wt':wt, 'mpg':mpg}

linear_model1 = smf.ols('wt~ mpg', data)

linear_result1 = linear_model1.fit()

anv = anova.anova_lm(linear_result1)

print(anv['F'][0])
import statsmodels.api as sm

import statsmodels.formula.api as smf

import numpy as np

from statsmodels.stats import anova

import patsy



mtcar_data = sm.datasets.get_rdataset('mtcars')

df = mtcar_data.data

wt = np.array(df.wt)

mpg =  np.array(df.mpg)

data = {'wt':wt, 'mpg':mpg}

linear_model1 = smf.ols('np.log(wt)~ np.log(mpg)', data)

linear_result1 = linear_model1.fit()

anv = anova.anova_lm(linear_result1)

print(anv['F'][0])
from scipy import stats

print(stats.mode([8, 9, 8, 7, 9, 6, 7, 6]))