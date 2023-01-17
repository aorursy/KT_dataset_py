import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import math
from matplotlib import pyplot as plt
import seaborn as sns

# We don't need much libraries for now, we'll keep adding them as we move on.
main_path = '/kaggle/input/siim-isic-melanoma-classification/'
sub = pd.read_csv(main_path + '/sample_submission.csv')
te = pd.read_csv(main_path + '/test.csv')
tr = pd.read_csv(main_path + '/train.csv')
tr.head(5) #taking a look at data
tr.info()
f = 1.4
sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['sex'],tr['age_approx']).apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Age', fontsize=20)
plt.ylabel('Sex', fontsize=20)
sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['age_approx'],tr['benign_malignant']).apply(lambda r: r/r.sum()*100, axis=1)).T.round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Age', fontsize=20)
plt.ylabel('Harmful or not?', fontsize=20)
sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['age_approx'],tr['benign_malignant']).T.apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Age', fontsize=20)
plt.ylabel('Harmful or not?', fontsize=20)
sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['sex'],tr['anatom_site_general_challenge']).apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Body Parts', fontsize=20)
plt.ylabel('Sex', fontsize=20)
sns.set(rc={'figure.figsize':(22,15)})
m = (pd.crosstab(tr['age_approx'],tr['anatom_site_general_challenge']).apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Body Parts', fontsize=20)
plt.ylabel('Age', fontsize=20)
sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['sex'],tr['diagnosis']).apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Sex', fontsize=20)
sns.set(rc={'figure.figsize':(22,3)})
m = (pd.crosstab(tr['diagnosis'],tr['sex'])[:-1].T.apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Sex', fontsize=20)
sns.set(rc={'figure.figsize':(22,15)})
m = (pd.crosstab(tr['diagnosis'],tr['age_approx'])[:-1].T.apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Age', fontsize=20)
sns.set(rc={'figure.figsize':(16,10)})
m = (pd.crosstab(tr['diagnosis'],tr['anatom_site_general_challenge'])[:-1].T.apply(lambda r: r/r.sum()*100, axis=1)).round(1)  
     
sns.set(font_scale = f)
sns.heatmap(m,annot = True,fmt = 'g',linewidths = '0.2',cmap="viridis")
plt.xlabel('Diagnosis', fontsize=20)
plt.ylabel('Body Parts', fontsize=20)