import pandas as pd
#Reading the input file
data=pd.read_csv('../input/responses.csv')
data.head()
#Checking the data type of variables
data.info(verbose='True')
#Converting float to categorical variable for Phobia
cols=['Flying','Storm','Darkness','Heights','Spiders','Snakes','Rats','Ageing','Fear of public speaking','Dangerous dogs']
for col in cols:
    data[col] = data[col].astype('category',copy=False)
#Calculating the count of missing values for each variable
null_columns=data.columns[data.isnull().any()]
data[null_columns].isnull().sum()
#Removing the records with missing values for Gender
modata=data[pd.notnull(data['Gender'])]
import matplotlib.pyplot as plt
import seaborn as sns

for i in range(len(cols)):
    plt.figure(i)
    sns.countplot(x=cols[i],hue='Gender', data=modata)
# From the above plots we can infer that the proportion of women having fear is always greater than men.
#For certain phobias like Spiders, Rats etc. we can observe that number of women having more fear(Rating- 4 and 5) 
#is way more than number of men.
# To support the above observation we will conduct hypothesis testing on the data
#Hypothesis Testing:
#Null Hypothesis: We will hypothesize here that gender and phobia are independent (for certain phobias)
#Alternate Hypothesis: Women have more fear than men for certain phobias or Gender and phobia are dependent on each other.
# Using the concept of Statistics, we will here use Chi-Square test as we have two categorical variables in picture.
ctb1=pd.crosstab(modata.Storm,modata.Gender)
ctb=ctb1.iloc[3:5]
import numpy as np
np.set_printoptions(suppress=True,formatter={'float_kind':'{:.70f}'.format})
#pd.options.display.float_format = '{:.4f}'.format
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
res=stats.chisquare(ctb.values, axis=None)
print(ctb)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
ctb_heights=pd.crosstab(modata.Heights,modata.Gender)
ctb_ht=ctb_heights.iloc[3:5]
print(ctb_ht)
res=stats.chisquare(ctb_ht.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_ht)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
ctb_spider=pd.crosstab(modata.Spiders,modata.Gender)
ctb_spi=ctb_spider.iloc[3:5]
print(ctb_spi)
res=stats.chisquare(ctb_spi.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_spi)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
ctb_darkness=pd.crosstab(modata.Darkness,modata.Gender)
ctb_dk=ctb_darkness.iloc[3:5]
print(ctb_dk)
res=stats.chisquare(ctb_dk.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_dk)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
ctb_Snakes=pd.crosstab(modata.Snakes,modata.Gender)
ctb_sk= ctb_Snakes.iloc[3:5]
print(ctb_sk)
res=stats.chisquare(ctb_sk.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_sk)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
ctb_rats=pd.crosstab(modata.Rats,modata.Gender)
ctb_rt=ctb_rats.iloc[3:5]
print(ctb_rt)
res=stats.chisquare(ctb_rt.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_rt)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
ctb_Ageing=pd.crosstab(modata.Ageing,modata.Gender)
ctb_ag=ctb_Ageing.iloc[3:5]
print(ctb_ag)
res=stats.chisquare(ctb_ag.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_ag)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')


ctb_Flying=pd.crosstab(modata.Flying,modata.Gender)
ctb_fl=ctb_Flying.iloc[3:5]
print(ctb_fl)
res=stats.chisquare(ctb_fl.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_fl)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
ctb_ddogs=pd.crosstab(modata["Dangerous dogs"],modata.Gender)
ctb_dd=ctb_ddogs[3:5]
print(ctb_dd)
res=stats.chisquare(ctb_dd.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_dd)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
ctb_fosp=pd.crosstab(modata["Fear of public speaking"],modata.Gender)
ctb_fs=ctb_fosp[3:5]
print(ctb_fs)
res=stats.chisquare(ctb_fs.values, axis=None)
print(res.pvalue)

stat, p, dof, expected = chi2_contingency(ctb_fs)
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
# From the above hypothesis testing for different phobias at 5% level of significance we can conclude that gender is 
#dependent on certain phobias. 
# Gender Dependent Phobia- Spider, Fear of Public Speaking,Rats, Dangerous Dogs and Snakes
#For all of them Female have more phobia than men.