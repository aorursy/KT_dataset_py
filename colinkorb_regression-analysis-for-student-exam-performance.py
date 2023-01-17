# Code to analyze and interpret various factors predicting student test score performance

# Load in necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


df = pd.read_csv("../input/StudentsPerformance.csv")

# Creating a scaled composite score
df['total score'] = df['math score'] + df['writing score'] + df['reading score']
df['total_scoreScale'] = np.divide(df['total score'],3)

# Renaming columns, as its easiest within the patsy formula to use single-word column names
df['math'] = df['math score']
df['read'] = df['reading score']
df['write'] = df['writing score']
df['ple'] = df['parental level of education']
df['tpc'] = df['test preparation course']
df['group'] = df['race/ethnicity']

# defining levels of parent education for categorical coding. In this case, everything will be relative to "some high school"
l = ["some high school",'high school',"associate's degree","some college","bachelor's degree","master's degree"]

md = smf.ols("math ~ (group+gender+C(ple,levels=l)+tpc)**1",
 df).fit()

print(md.summary())
mdr = smf.ols("read ~ (group+gender+C(ple,levels=l)+tpc)**1",df).fit()

print(mdr.summary())
mdw = smf.ols("write ~ (group+gender+C(ple,levels=l)+tpc)**1",df).fit()
print(mdw.summary())
mdcomp = smf.ols("total_scoreScale ~ group+gender+C(ple,levels=l)+tpc",df).fit()
print(mdcomp.summary())
fig_resids,ax = plt.subplots(2,2,sharey=True)
fit_y = md.fittedvalues
sns.residplot(fit_y,'math',data=df,lowess=True,ax=ax[0,0],line_kws={'color': 'black', 'lw': 1, 'alpha': 0.8})
ax[0,0].set_xlabel('Fitted Math Score')
ax[0,0].set_ylabel('Residuals')
sns.residplot(mdr.fittedvalues,'read',data=df,lowess=True,ax=ax[0,1],line_kws={'color': 'black', 'lw': 1, 'alpha': 0.8})
ax[0,1].set_xlabel('Fitted Reading Score')
ax[0,1].set_ylabel('')
sns.residplot(mdw.fittedvalues,'write',data=df,lowess=True,ax=ax[1,0],line_kws={'color': 'black', 'lw': 1, 'alpha': 0.8})
ax[1,0].set_xlabel('Fitted Writing Score')
ax[1,0].set_ylabel('Residuals')
sns.residplot(mdcomp.fittedvalues,'total_scoreScale',data=df,lowess=True,ax=ax[1,1],line_kws={'color': 'black', 'lw': 1, 'alpha': 0.8})
ax[1,1].set_xlabel('Fitted Composite Score')
ax[1,1].set_ylabel('')
plt.tight_layout()
#output plot:
rho=np.zeros(3)
spear_p = np.zeros(3)
p_r = np.zeros(3)
pear_p = np.zeros(3)

#Math and Writing
rho[0],spear_p[0] = spearmanr(df['math'],df['write'])
p_r[0],pear_p[0] = pearsonr(df['math'],df['write'])

#Math and Reading
rho[1],spear_p[1] = spearmanr(df['math'],df['read'])
p_r[1],pear_p[1] = pearsonr(df['math'],df['read'])

#Writing and Reading
rho[2],spear_p[2] = spearmanr(df['write'],df['read'])
p_r[2],pear_p[2] = pearsonr(df['write'],df['read'])
#Plotting our Correlations
fig,ax = plt.subplots(1,3,sharey=True)
ax[0].plot(df['math'],df['write'],'ko')
ax[0].set_xlabel('Math Scores')
ax[0].set_ylabel('Writing Scores')
ax[0].set_title("Spearman's Rho: {:.2f} \n Pearson's R: {:.2f}".format(rho[0],p_r[0]),fontsize=9)

ax[1].plot(df['math'],df['read'],'ko')
ax[1].set_xlabel('Math Scores')
ax[1].set_ylabel('Reading Scores')
ax[1].set_title("Spearman's Rho: {:.2f} \n Pearson's R: {:.2f}".format(rho[1],p_r[1]),fontsize=9)

ax[2].plot(df['write'],df['read'],'ko')
ax[2].set_xlabel('Writing Scores')
ax[2].set_ylabel('Reading Scores')
ax[2].set_title("Spearman's Rho: {:.2f} \n Pearson's R: {:.2f}".format(rho[2],p_r[2]),fontsize=9)
fig.subplots_adjust(wspace = 0.6)