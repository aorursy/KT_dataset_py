import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

schema = pd.read_csv('../input/survey_results_schema.csv')
schema.head()
results = pd.read_csv('../input/survey_results_public.csv')
results = results.set_index('Respondent')
results.head()
import seaborn as sns
import matplotlib.pyplot as plt
results = results.assign(_MinCompanySize=results.CompanySize.str.replace(',','').str.replace('Fewer','0').str.split().str[0].astype('float'))
g = sns.countplot(results['_MinCompanySize'])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g = sns.countplot(results['StackOverflowVisit'])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
#YearsCodingProf, CareerSatisfaction
plt.figure(figsize=(20,6))
results = results.assign(_MinYearsCodingProf=results.YearsCodingProf.str.replace('-',' ').str.split().str[0].astype('float'))
g = sns.violinplot(y="_MinYearsCodingProf", x="CareerSatisfaction", data=results,hue='OpenSource')
g.set_xticklabels(g.get_xticklabels(),rotation=90)
F = sns.FacetGrid(results[results.Country.isin(results.Country.value_counts().head(5).index)],col='Country',col_wrap=5)
F.map(sns.countplot,"AIFuture")
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in F.axes.flat]
sns.pairplot(results[['AssessJob'+str(i) for i in range(1,5)]].dropna())
#sns.jointplot(x='Country',y='OpenSource',data=results) 
#results.plot.scatter(x='Country',y='OpenSource')
#sns.heatmap(results[['AdsPriorities5','Country','OpenSource']].corr(),annot=True)

results = results.assign(_StackOverflowRecommend=results.StackOverflowRecommend.str.split().str[0].astype('float'))
res_ = results[results.LastNewJob.isin(results.LastNewJob.value_counts().head(5).index) & results.YearsCoding.isin(results.YearsCoding.value_counts().head(3).index)]
F = sns.FacetGrid(res_,row='LastNewJob',col='YearsCoding', gridspec_kws={"wspace":1},size=5)
g = F.map(sns.countplot,"_StackOverflowRecommend").fig.subplots_adjust(wspace=2.5, hspace=1)
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in F.axes.flat]

