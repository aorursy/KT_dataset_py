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
data = pd.read_csv('/kaggle/input/iris/Iris.csv')

data.head()
data['Species'].unique()
from scipy.stats import ttest_1samp



PetalLengthCm_mean = data['PetalLengthCm'].mean()

print('PetalLength mean value:',PetalLengthCm_mean)

tset, pval = ttest_1samp(data['PetalLengthCm'], 1.3)

print('p-values',pval)

if pval < 0.05:    # alpha value is 0.05 or 5%

   print(" we are rejecting null hypothesis")

else:

  print("we are accepting null hypothesis")
from scipy.stats import ttest_ind



PetalLengthCm_mean = data['PetalLengthCm'].mean()

SepalLengthCm_mean = data['SepalLengthCm'].mean()

print('PetalLength mean value:',PetalLengthCm_mean)

print('SepalLength mean value:',SepalLengthCm_mean)

PetalLengthCm_std = data['PetalLengthCm'].std()

SepalLengthCm_std = data['SepalLengthCm'].std()

print('PetalLength std value:',PetalLengthCm_std)

print('SepalLength std value:',SepalLengthCm_std)

ttest,pval = ttest_ind(data['PetalLengthCm'],data['SepalLengthCm'])

print('p-value',pval)

if pval <0.05:

  print("we reject null hypothesis")

else:

  print("we accept null hypothesis")
from scipy import stats



ttest,pval = stats.ttest_rel(data['PetalLengthCm'].loc[data['Species']=='Iris-setosa'], data['PetalLengthCm'].loc[data['Species']=='Iris-versicolor'])

print(pval)

if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")
from scipy import stats

from statsmodels.stats import weightstats as stests



ztest ,pval = stests.ztest(data['PetalLengthCm'], x2=None, value=1.3)

print(float(pval))

if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")
ztest ,pval1 = stests.ztest(data['PetalLengthCm'], x2=data['SepalLengthCm'], value=0, alternative='two-sided')

print(float(pval1))

if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")
df_anova = data[['SepalLengthCm','Species']]

grps = pd.unique(df_anova.Species.values)

d_data = {grp:df_anova['SepalLengthCm'][df_anova.Species == grp] for grp in grps}

 

F, p = stats.f_oneway(d_data['Iris-setosa'], d_data['Iris-versicolor'], d_data['Iris-virginica'])

print("p-value for significance is: ", p)

if p<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")
import statsmodels.api as sm

from statsmodels.formula.api import ols

import warnings



warnings.filterwarnings('ignore')

model = ols('PetalLengthCm ~ C(SepalLengthCm)*C(SepalWidthCm)', data).fit()

print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

res = sm.stats.anova_lm(model, typ= 2)

res
df_chi = pd.read_csv('../input/womens-international-football-results/results.csv')

contingency_table=pd.crosstab(df_chi["tournament"],df_chi["neutral"])

print('contingency_table :-\n',contingency_table)

#Observed Values

Observed_Values = contingency_table.values 

print("Observed Values :-\n",Observed_Values)

b=stats.chi2_contingency(contingency_table)

Expected_Values = b[3]

print("Expected Values :-\n",Expected_Values)

no_of_rows=len(contingency_table.iloc[0:2,0])

no_of_columns=len(contingency_table.iloc[0,0:2])

ddof=(no_of_rows-1)*(no_of_columns-1)

print("Degree of Freedom:-",ddof)

alpha = 0.05

from scipy.stats import chi2

chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])

chi_square_statistic=chi_square[0]+chi_square[1]

print("chi-square statistic:-",chi_square_statistic)

critical_value=chi2.ppf(q=1-alpha,df=ddof)

print('critical_value:',critical_value)

#p-value

p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)

print('p-value:',p_value)

print('Significance level: ',alpha)

print('Degree of Freedom: ',ddof)

print('chi-square statistic:',chi_square_statistic)

print('critical_value:',critical_value)

print('p-value:',p_value)

if chi_square_statistic>=critical_value:

    print("Reject H0,There is a relationship between 2 categorical variables")

else:

    print("Retain H0,There is no relationship between 2 categorical variables")

    

if p_value<=alpha:

    print("Reject H0,There is a relationship between 2 categorical variables")

else:

    print("Retain H0,There is no relationship between 2 categorical variables")