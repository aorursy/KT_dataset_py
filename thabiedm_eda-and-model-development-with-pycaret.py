!pip install pycaret
!pip install seaborn==0.11.0
from pycaret.utils import enable_colab 
enable_colab()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pycaret.classification import *
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.info()
df.shape
plt.figure(figsize=(18,10))
sns.heatmap(df.corr(),
            vmin=-1,
            cmap='coolwarm',
            annot=True)
plt.show()
deaths = len(df[df['DEATH_EVENT'] == 1])/len(df)
survivors = len(df[df['DEATH_EVENT'] == 0])/len(df)

print('% death class: ',round(deaths,2))
print('% survived class: ',round(survivors,2))
g = sns.displot(df, x="serum_creatinine", hue="DEATH_EVENT", element="step")
g.fig.set_size_inches(12,7)
g = sns.displot(df, x="ejection_fraction", hue="DEATH_EVENT", element="step")
g.fig.set_size_inches(12,7)
sns.set_theme(style="darkgrid")

g = sns.displot(
    df, x="serum_creatinine", col="DEATH_EVENT", row="sex",
    binwidth=0.3, height=3, facet_kws=dict(margin_titles=True))

g.fig.set_size_inches(14,8)
sns.set_theme(style="darkgrid")

g = sns.displot(
    df, x="ejection_fraction", col="DEATH_EVENT", row="sex",
    binwidth=5, height=3, facet_kws=dict(margin_titles=True))

g.fig.set_size_inches(14,8)
g = sns.catplot(x="DEATH_EVENT", y="serum_creatinine", data=df,ax=(12,8))
g.fig.set_size_inches(12,7)
g = sns.catplot(x="DEATH_EVENT", y="ejection_fraction", data=df,ax=(12,8))
g.fig.set_size_inches(12,7)
df.plot.scatter(x='serum_sodium', y='serum_creatinine', c='DEATH_EVENT', colormap='coolwarm',s=40 ,figsize=(12,7), title='Creatinine to Sodium')
# Creating figure 
fig = plt.figure(figsize = (16, 9)) 
ax = plt.axes(projection ="3d")

# Add x, y gridlines  
ax.grid(b = True, color ='grey',  
        linestyle ='-.', linewidth = 0.3,  
        alpha = 0.2)  
  
# Creating plot 
sctt = ax.scatter3D(df['serum_sodium'], df['serum_creatinine'], df['ejection_fraction'], 
                    alpha = 0.8, 
                    c = df['DEATH_EVENT'],
                    s = 40) 
  
plt.title('Creatinine to Sodium to Ejection Fraction') 
ax.set_xlabel('serum_sodium', fontweight ='bold')  
ax.set_ylabel('serum_creatinine', fontweight ='bold')  
ax.set_zlabel('ejection_fraction', fontweight ='bold') 
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5) 
  
# show plot 
plt.show() 
dfnew = df[['age','ejection_fraction','serum_creatinine','DEATH_EVENT']]
experiment1 = setup(data = dfnew, target = 'DEATH_EVENT', session_id=123,
                  normalize = True, 
                  transformation = True, 
                  numeric_features=['ejection_fraction'],
                  bin_numeric_features = ['age'],
                  log_experiment = True, experiment_name = 'heart_failure1')
top3 = compare_models(n_select = 3)
model = create_model('catboost', fold = 5)
tuned_model = tune_model(model, optimize = 'Accuracy')
