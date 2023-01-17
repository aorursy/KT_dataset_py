#pivoting and drop Null values for clean and easy analysis
pr_piv= pr[['fips', 'candidate','fraction_votes']].pivot(index='fips', columns='candidate', values='fraction_votes')
pr_piv.drop(' No Preference', axis=1, inplace=True)
pr_piv.drop(' Uncommitted', axis=1, inplace=True)
pr_facts=pd.merge(pr_piv, facts, right_index=True, left_index=True)
pr_facts=pr_facts.dropna()
#multiindex  to make data more readable
c=pr[['party','candidate']].drop_duplicates().sort_values(by=['party','candidate'])
c = c.loc[c['candidate'] != ' No Preference']
c = c.loc[c['candidate'] != ' Uncommitted']
t=c[['party', 'candidate']].apply(tuple, axis=1).tolist()
index = pd.MultiIndex.from_tuples(t, names=['Democrat', 'Republican'])
#heatmap visualization
def heatmap(data):
  fig, ax = plt.subplots(figsize=(10, 10))
  heatmap = sns.heatmap(data, cmap=plt.cm.Blues,annot=True, annot_kws={"size": 8})
  ax.xaxis.tick_top()
  # rotate
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  plt.tight_layout()
#skipy linregress
#Pearson Correlation
rvalue = DataFrame(np.nan,index=cf_dict.index,columns=index)
rvalue.columns.names=['Party','Candidate']
rvalue.columns.lexsort_depth
rvalue.index.names=['Fact']
#PValue
pvalue = DataFrame(np.nan,index=cf_dict.index,columns=index)
pvalue.columns.names=['Party','Candidate']
pvalue.columns.lexsort_depth
pvalue.index.names=['Fact']
#StdErr
stderr = DataFrame(np.nan,index=cf_dict.index,columns=index)
stderr.columns.names=['Party','Candidate']
stderr.columns.lexsort_depth
stderr.index.names=['Fact']

#
for c_X in pr_piv.columns:
  for c_Y in cf_dict.index:
    R=linregress(pr_facts[[c_X,c_Y]])
    p_X=index.get_loc_level(c_X,1)[1][0]
    rvalue.set_value(c_Y,(p_X,c_X), R.rvalue)
    pvalue.set_value(c_Y,(p_X,c_X), R.pvalue)
    stderr.set_value(c_Y,(p_X,c_X), R.stderr)
#Let's find out the most correlated facts to Democrat candidates choice
#democrats only

DemRvalue=rvalue['Democrat']
DemPvalue=pvalue['Democrat']
DemStdErr=stderr['Democrat']

DemRvalue_idxmax=DemRvalue.idxmax(axis=0)

DemRvalue_max = DataFrame(np.nan,index=DemRvalue_idxmax.tolist(),columns=DemRvalue_idxmax.index)
DemRvalue_max['description']=''

DemPvalue_max = DataFrame(np.nan,index=DemRvalue_idxmax.tolist(),columns=DemRvalue_idxmax.index)
DemPvalue_max['description']=''

DemStdErr_max = DataFrame(np.nan,index=DemRvalue_idxmax.tolist(),columns=DemRvalue_idxmax.index)
DemStdErr_max['description']=''


for c_X in DemRvalue_idxmax.index:
    for c_Y in DemRvalue_idxmax.tolist():
        DemRvalue_max.set_value(c_Y,c_X, DemRvalue[c_X][c_Y])
        DemRvalue_max.set_value(c_Y,'description', cf_dict['description'][c_Y])

        DemPvalue_max.set_value(c_Y,c_X, DemPvalue[c_X][c_Y])
        DemPvalue_max.set_value(c_Y,'description', cf_dict['description'][c_Y])

        DemStdErr_max.set_value(c_Y,c_X, DemStdErr[c_X][c_Y])
        DemStdErr_max.set_value(c_Y,'description', cf_dict['description'][c_Y])
#There is a strong correlation between percent of Asian and Bernie Sanders votes fraction. 
#In the opposite, Hillary Clinton has anti-correlation with Asian percent and 
#stong positive correlation with White percent.

#The PValue is small enough to trust the results

DemRvalue_max=DemRvalue_max.set_index('description')
heatmap(DemRvalue_max)

DemPvalue_max=DemPvalue_max.set_index('description')
heatmap(DemPvalue_max)

DemStdErr_max=DemStdErr_max.set_index('description')
heatmap(DemStdErr_max)
#More details for most correlated facts Democrat candidates
#Asian alone, percent, 2014
sns_plot = sns.jointplot('Bernie Sanders','RHI425214',pr_facts,kind='scatter')

#White alone, percent, 2014
sns_plot = sns.jointplot('Hillary Clinton','RHI125214',pr_facts,kind='scatter')
#republicans only
#most correlated facts to Republican candidates choice

RepRvalue=rvalue['Republican']
RepPvalue=pvalue['Republican']
RepStdErr=stderr['Republican']

RepRvalue_idxmax=RepRvalue.idxmax(axis=0)

RepRvalue_max = DataFrame(np.nan,index=list(set(RepRvalue_idxmax.tolist())),columns=RepRvalue_idxmax.index)
RepRvalue_max['description']=''

RepPvalue_max = DataFrame(np.nan,index=list(set(RepRvalue_idxmax.tolist())),columns=RepRvalue_idxmax.index)
#RepPvalue_max['description']=''

RepStdErr_max = DataFrame(np.nan,index=list(set(RepRvalue_idxmax.tolist())),columns=RepRvalue_idxmax.index)
#RepStdErr_max['description']=''


for c_X in RepRvalue_idxmax.index:
    for c_Y in RepRvalue_idxmax.tolist():
        RepRvalue_max.set_value(c_Y,c_X, RepRvalue[c_X][c_Y])
        RepRvalue_max.set_value(c_Y,'description', cf_dict['description'][c_Y])

        RepPvalue_max.set_value(c_Y,c_X, RepPvalue[c_X][c_Y])
        #RepPvalue_max.set_value(c_Y,'description', cf_dict['description'][c_Y])

        RepStdErr_max.set_value(c_Y,c_X, RepStdErr[c_X][c_Y])
        #RepStdErr_max.set_value(c_Y,'description', cf_dict['description'][c_Y])
#Here is the similar analysis for republicans. 
#The results are more sparse but what we can see the strong positive relationship 
#between percent of Housing units in multi-unit structures and votes fractions of 
#John Kasich, Marco Rubio and Rand Paul.
#There is also the strong correlation between percent of Bachelor's degree or higher and 
#the same republican candidates

#The PValue is very low and we can trust the results.

#Interesting, Donald Trump has the strong anti-correlated results with the percent of 
#Bachelor's degree or higher Fact with a low PValue

#He has a moderate positive correlation with the percent of Persons 65 years and over. 
#The PValue is high in this case Marco Rubio fraction votes is strongly anti-correlated 
#with the percent of Persons 65 years and over fact and PValue is very low.
#FYI Donald Trump and Marco Rubio is one of the most anti-correlated Republican candidates 
#as I discovered in my other analysis

RepRvalue_max=RepRvalue_max.set_index('description')
heatmap(RepRvalue_max)

#RepPvalue_max=RepPvalue_max.set_index('description')
heatmap(RepPvalue_max)

#RepStdErr_max=RepStdErr_max.set_index('description')
heatmap(RepStdErr_max)
#More details for most correlated facs and Republican candidates

#Bachelor's degree or higher, percent of persons age 25+, 2009-2013
sns_plot = sns.jointplot('Marco Rubio','EDU685213',pr_facts,kind='scatter')


sns_plot = sns.jointplot('Donald Trump','EDU685213',pr_facts,kind='scatter')


#Housing units in multi-unit structures
sns_plot = sns.jointplot('Marco Rubio','HSG096213',pr_facts,kind='scatter')



sns_plot = sns.jointplot('Donald Trump','HSG096213',pr_facts,kind='scatter')


#Persons 65 years and over, percent, 2014
sns_plot = sns.jointplot('Marco Rubio','AGE775214',pr_facts,kind='scatter')



sns_plot = sns.jointplot('Donald Trump','AGE775214',pr_facts,kind='scatter')


#General view is a huge image and it's hard to review
heatmap(rvalue)
heatmap(pvalue)
heatmap(stderr)
# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

system("ls ../input")

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.