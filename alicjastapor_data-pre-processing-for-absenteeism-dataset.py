import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
rawdata = pd.read_csv('/kaggle/input/absenteeism-at-works/absenteeism.csv')
pd.options.display.max_columns = None
rawdata
df = rawdata.copy()
df.info()
df = df.drop(['ID'], axis = 1) 
len(df['Reason for absence'].unique())
sorted(df['Reason for absence'].unique())
rcol = pd.get_dummies(df['Reason for absence'])
rcol['check'] = rcol.sum(axis=1)
rcol['check'].unique() #checking if for sure every person have only one reason 
rcol = rcol.drop(['check'], axis=1)
rcol = pd.get_dummies(df['Reason for absence'], drop_first = True) #drop reason '0'
rcol
df = df.drop(['Reason for absence'], axis=1) #drop 'Reason for absence', replace with dummies. 
df
# merging dummies into 4 categories based on reason for abscence
reasontype1 = rcol.loc[:, 1:14].max(axis=1)
reasontype2 = rcol.loc[:, 15:17].max(axis=1)
reasontype3 = rcol.loc[:, 18:21].max(axis=1)
reasontype4 = rcol.loc[:, 22:28].max(axis=1)
print(reasontype1.sum(), reasontype2.sum(), reasontype3.sum(), reasontype4.sum())
df = pd.concat([df, reasontype1, reasontype2, reasontype3, reasontype4], axis = 1)
df
column_names = ['Month of absence', 'Day of the week', 'Seasons',
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours', 'Reason1', 'Reason2', 'Reason3', 'Reason4']
df.columns = column_names
df
reordered = ['Reason1', 'Reason2', 'Reason3', 'Reason4','Month of absence', 'Day of the week', 'Seasons',
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours']
df = df[reordered]
df
df_mod1 = df.copy()
df_mod1
#correlation matrix
cormatrix = df_mod1.corr()
plt.subplots(figsize=(8, 8))
sns.heatmap(cormatrix, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
cols = cormatrix.nlargest(10, 'Absenteeism time in hours')['Absenteeism time in hours'].index
corrcoef = np.corrcoef(df_mod1[cols].values.T)
plt.subplots(figsize=(8, 8))
sns.heatmap(corrcoef, annot=True,  yticklabels=cols.values, xticklabels=cols.values, vmin=-1, vmax=1, center= 0,  cmap= 'coolwarm')
df_mod2 = df_mod1.copy() 
df_mod2 = df_mod2.drop(['Month of absence','Distance from Residence to Work','Body mass index'], axis = 1)
df_mod2['Education'].unique()
df_mod2['Education'].value_counts()
df_mod2['Education'] = df_mod2['Education'].map({1:0, 2:1, 3:1, 4:1})
df_mod2['Education'].value_counts()
d_pre = df_mod2.copy()
d_pre
d_pre.to_csv('Absenteeism_preprocessed.csv', index=False)