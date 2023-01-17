# Importing packages

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import scipy.stats # Needed to compute statistics for categorical data
kidney_data = pd.read_csv('../input/ckdisease/kidney_disease.csv')

kidney_data.head()
kidney_data.isnull().sum()
kidney_data = kidney_data.dropna(axis=0)

kidney_data.isnull().sum()
kidney_data['rbc'] = kidney_data.rbc.replace(['normal','abnormal'], ['1', '0'])

kidney_data['pc'] = kidney_data.pc.replace(['normal','abnormal'], ['1', '0'])

kidney_data['pcc'] = kidney_data.pcc.replace(['present','notpresent'], ['1', '0'])

kidney_data['ba'] = kidney_data.ba.replace(['present','notpresent'], ['1', '0'])

kidney_data['htn'] = kidney_data.htn.replace(['yes','no'], ['1', '0'])

kidney_data['dm'] = kidney_data.dm.replace(['yes','no'], ['1', '0'])

kidney_data['cad'] = kidney_data.cad.replace(['yes','no'], ['1', '0'])

kidney_data['appet'] = kidney_data.appet.replace(['good','poor'], ['1', '0'])

kidney_data['pe'] = kidney_data.pe.replace(['yes','no'], ['1', '0'])

kidney_data['ane'] = kidney_data.ane.replace(['yes','no'], ['1', '0'])

kidney_data['classification'] = kidney_data.classification.replace(['ckd','ckd\t','notckd'], ['positive', 'positive','negative'])

kidney_data.head()
g = sns.pairplot(kidney_data, vars =['age', 'bp','bgr', 'bu', 'sc','sod', 'pot', 'hemo'],hue = 'classification')

g.map_diag(sns.distplot)

g.add_legend()

g.fig.suptitle('FacetGrid plot', fontsize = 20)

g.fig.subplots_adjust(top= 0.9);
kidney_data[kidney_data['classification'] == 'positive'].describe()
kidney_data[kidney_data['classification'] == 'negative'].describe()
kidney_data1 = pd.read_csv('../input/ckdisease/kidney_disease.csv')

kidney_data1[kidney_data1['rbc'].isnull()].groupby(['classification']).size().reset_index(name = 'count')
kidney_data = pd.read_csv('../input/ckdisease/kidney_disease.csv')
kidney_data = kidney_data.interpolate(method='pad')

kidney_data.rbc = kidney_data.rbc.interpolate(method='pad')

kidney_data.pc = kidney_data.pc.interpolate(method='pad')

kidney_data['rbc'] = kidney_data.rbc.replace(['normal','abnormal'], [1,0])

kidney_data['pc'] = kidney_data.pc.replace(['normal','abnormal'], [1,0])

kidney_data['pcc'] = kidney_data.pcc.replace(['present','notpresent'], [1,0])

kidney_data['ba'] = kidney_data.ba.replace(['present','notpresent'], [1,0])

kidney_data['htn'] = kidney_data.htn.replace(['yes','no'], [1,0])

kidney_data['dm'] = kidney_data.dm.replace(['yes','no'], [1,0])

kidney_data['cad'] = kidney_data.cad.replace(['yes','no'], [1,0])

kidney_data['appet'] = kidney_data.appet.replace(['good','poor'], [1,0])

kidney_data['pe'] = kidney_data.pe.replace(['yes','no'], [1,0])

kidney_data['ane'] = kidney_data.ane.replace(['yes','no'], [1,0])

kidney_data['classification'] = kidney_data.classification.replace(['ckd','ckd\t','notckd'], [1,1,0])

kidney_data['wc'] = kidney_data.wc.replace(['\t6200','\t8400'], [6200,8400])

kidney_data = kidney_data.dropna(axis=0)

kidney_data.isnull().sum()
gg = sns.pairplot(kidney_data, vars =['age', 'bp','bgr', 'bu', 'sc','sod', 'pot', 'hemo'],hue = 'classification')

gg.map_diag(sns.distplot)

gg.add_legend()

gg.fig.suptitle('FacetGrid plot', fontsize = 20)

gg.fig.subplots_adjust(top= 0.9);
corr = kidney_data.corr()

corr.style.background_gradient(cmap='RdBu_r')
plt.figure(figsize=(70,25))

plt.legend(loc='upper left')

g = sns.countplot(data = kidney_data, x = 'age', hue = 'classification')

g.legend(title = 'Chronic kidney disease patient?', loc='center left', bbox_to_anchor=(0.1, 0.5), ncol=1)

g.tick_params(labelsize=20)

plt.setp(g.get_legend().get_texts(), fontsize='32')

plt.setp(g.get_legend().get_title(), fontsize='42')

g.axes.set_title('Graph of age vs number of patients with chronic kidney disease',fontsize=50)

g.set_xlabel('Count',fontsize=40)

g.set_ylabel("Age",fontsize=40)
age_corr = ['age', 'classification']

age_corr1 = kidney_data[age_corr]

age_corr_y = age_corr1[age_corr1['classification'] == 1].groupby(['age']).size().reset_index(name = 'count')

age_corr_y.corr()
sns.regplot(data = age_corr_y, x = 'age', y = 'count').set_title("Correlation graph for Age vs chronic kidney disease patient")
age_corr_n = age_corr1[age_corr1['classification'] == 0].groupby(['age']).size().reset_index(name = 'count')

age_corr_n.corr()
sns.regplot(data = age_corr_n, x = 'age', y = 'count').set_title("Correlation graph for Age vs healthy patient")
# Chi-sq test

cont = pd.crosstab(kidney_data["rbc"],kidney_data["classification"])

scipy.stats.chi2_contingency(cont)
# Chi-sq test

cont = pd.crosstab(kidney_data["pc"],kidney_data["classification"])

scipy.stats.chi2_contingency(cont)
# Measuring blood glucose and chronic kidney disease patient 

bgr_corr = ['bgr', 'classification']

bgr_corr1 = kidney_data[bgr_corr]

bgr_corr1.bgr = bgr_corr1.bgr.round(-1)

bgr_corr_y = bgr_corr1[bgr_corr1['classification'] == 1].groupby(['bgr']).size().reset_index(name = 'count')

bgr_corr_y.corr()
sns.regplot(data = bgr_corr_y, x = 'bgr', y = 'count').set_title("Correlation graph for blood glucose vs chronic kidney disease patient")
bgr_corr_n = bgr_corr1[bgr_corr1['classification'] == 0].groupby(['bgr']).size().reset_index(name = 'count')

bgr_corr_n.corr()
sns.regplot(data = bgr_corr_n, x = 'bgr', y = 'count').set_title("Correlation graph for blood glucose vs healthy patient")
# Measuring blood urea and chronic kidney disease patient 

bu_corr = ['bu', 'classification']

bu_corr1 = kidney_data[bu_corr]

bu_corr1.bu = kidney_data.bu.round(-1)

bu_corr_y = bu_corr1[bu_corr1['classification'] == 1].groupby(['bu']).size().reset_index(name = 'count')

bu_corr_y.corr()
sns.regplot(data = bu_corr_y, x = 'bu', y = 'count').set_title('Correlation graph for blood urea vs CKD patient')
bu_corr_n = bu_corr1[bu_corr1['classification'] == 0].groupby(['bu']).size().reset_index(name = 'count')

bu_corr_n.corr()
sns.regplot(data = bu_corr_n, x = 'bu', y = 'count').set_title('Correlation graph for blood urea vs healthy patient')
# Measuring blood sodium and chronic kidney disease patient 

sod_corr = ['sod', 'classification']

sod_corr1 = kidney_data[sod_corr]

sod_corr_y = sod_corr1[sod_corr1['classification'] == 1].groupby(['sod']).size().reset_index(name = 'count')

sod_corr_y.corr()
sns.regplot(data = sod_corr_y, x = 'sod', y = 'count').set_title('Correlation graph for blood sodium vs CKD patient')
sod_corr_n = sod_corr1[sod_corr1['classification'] == 0].groupby(['sod']).size().reset_index(name = 'count')

sod_corr_n.corr()
sns.regplot(data = sod_corr_n, x = 'sod', y = 'count').set_title('Correlation graph for blood sodium vs healthy patient')
# Chi-sq test

cont = pd.crosstab(kidney_data["pe"],kidney_data["classification"])

scipy.stats.chi2_contingency(cont)
# Chi-sq test

cont = pd.crosstab(kidney_data["ane"],kidney_data["classification"])

scipy.stats.chi2_contingency(cont)
# Measuring serum creatinine and chronic kidney disease patient 

sc_corr = ['sc', 'classification']

sc_corr1 = kidney_data[sc_corr]

sc_corr1.sc = sc_corr1.sc.round(1)

sc_corr_y = sc_corr1[sc_corr1['classification'] == 1].groupby(['sc']).size().reset_index(name = 'count')

sc_corr_y.corr()
sns.regplot(data = sc_corr_y, x = 'sc', y = 'count').set_title('Correlation graph for serum creatinine vs CKD patient')
sc_corr_n = sc_corr1[sc_corr1['classification'] == 0].groupby(['sc']).size().reset_index(name = 'count')

sc_corr_n.corr()
sns.regplot(data = sc_corr_n, x = 'sc', y = 'count').set_title('Correlation graph for serum creatinine vs CKD patient')
# Chi-sq test

cont = pd.crosstab(kidney_data["dm"],kidney_data["classification"])

scipy.stats.chi2_contingency(cont)
# Chi-sq test

cont = pd.crosstab(kidney_data["cad"],kidney_data["classification"])

scipy.stats.chi2_contingency(cont)