# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/core_dataset.csv')
data.head()
sub1=data[['Employee Number','MaritalDesc','Employment Status','Sex']].dropna()
sub1.head()
#Frequency table for category MarialDesc
List=sub1['MaritalDesc']
List.value_counts()
#Frequency table for category Employment Status
List=sub1['Employment Status']
List.value_counts()
#Removing not future start employee from the set and see if employee martial status has any relation on the status
sub1=sub1[sub1['Employment Status']!='Future Start']
sub1['Employment Status'].value_counts()
#changing Employment status to binary
sub1['EStatus_bin']=np.where(sub1['Employment Status']=='Active',1,0)
sub1['EStatus_bin'].value_counts()
#changing marital status to binary
sub1['MStatus_bin']=np.where(sub1['MaritalDesc']=='Married',1,0)
sub1['MStatus_bin'].value_counts()
#import stats model

import statsmodels.api as sm
import statsmodels.formula.api as smf
reg1 = smf.ols('EStatus_bin ~ MStatus_bin', data=sub1).fit()
print (reg1.summary())
#Q-Q plot for normality
fig1=sm.qqplot(reg1.resid, line='r')

# simple plot of residuals
stdres=pd.DataFrame(reg1.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')


# additional regression diagnostic plots
fig2 = plt.figure(figsize=(12,8))
fig2 = sm.graphics.plot_regress_exog(reg1,  "MStatus_bin", fig=fig2)

# leverage plot
fig3=sm.graphics.influence_plot(reg1, size=8)
print(fig3)


#Considering additional variable sex 
#Frequency table for category Sex
List=sub1['Sex']
List.value_counts()

#changing sex  to binary
sub1['Female_bin']=np.where(sub1['Sex']=='Female',1,0)
sub1['Female_bin'].value_counts()
# logistic regression with Marital status on employment status
lreg1 = smf.logit(formula = 'EStatus_bin ~ MStatus_bin', data = sub1).fit()
print (lreg1.summary())
# odds ratios
print ("Odds Ratios")
print (np.exp(lreg1.params))

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))
# logistic regression with Marital status & sex on employment status
lreg2 = smf.logit(formula = 'EStatus_bin ~ MStatus_bin + Female_bin', data = sub1).fit()
print (lreg2.summary())
# odds ratios
print ("Odds Ratios")
print (np.exp(lreg2.params))

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (np.exp(conf))