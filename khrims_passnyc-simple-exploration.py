import pandas as pd
import seaborn as sns
sns.set()
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows',999)
df1 = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
df2 = pd.read_csv('../input/2016 School Explorer.csv')
df2.head(3)
dummy = df2.loc[:, 'School Income Estimate':'Percent White']
dummy.head()
dummy.dtypes
dummy['School Income Estimate'] = dummy['School Income Estimate'].str.replace('$','')
dummy['Percent ELL'] = dummy['Percent ELL'].str.replace('%','')
dummy['Percent Asian'] = dummy['Percent Asian'].str.replace('%','')
dummy['Percent Black'] = dummy['Percent Black'].str.replace('%','')
dummy['Percent Hispanic'] = dummy['Percent Hispanic'].str.replace('%','')
dummy['Percent Black / Hispanic'] = dummy['Percent Black / Hispanic'].str.replace('%','')
dummy['Percent White'] = dummy['Percent White'].str.replace('%','')
dummy['Percent ELL'] = dummy['Percent ELL'].astype(int).div(100)
dummy['Percent Asian'] = dummy['Percent Asian'].astype(int).div(100)
dummy['Percent Black'] = dummy['Percent Black'].astype(int).div(100)
dummy['Percent Hispanic'] = dummy['Percent Hispanic'].astype(int).div(100)
dummy['Percent Black / Hispanic'] = dummy['Percent Black / Hispanic'].astype(int).div(100)
dummy['Percent White'] = dummy['Percent White'].astype(int).div(100)
dummy['School Income Estimate'] = dummy['School Income Estimate'].str.replace(',','')
dummy['School Income Estimate'] = dummy['School Income Estimate'].astype(float)
dummy.head()
df2['School Income Estimate'] = dummy['School Income Estimate']
df2['Percent ELL'] = dummy['Percent ELL']
df2['Percent Asian'] = dummy['Percent Asian']
df2['Percent Black'] = dummy['Percent Black']
df2['Percent Hispanic'] = dummy['Percent Hispanic']
df2['Percent Black / Hispanic'] = dummy['Percent Black / Hispanic']
df2['Percent White'] = dummy['Percent White']
df2.loc[:, 'Economic Need Index':'Percent White'].dtypes
df2.isnull().sum()
df2.dtypes
#creating a variable with no null School Income Index
df2_no_null = df2.dropna(subset=['School Income Estimate'],how='all')
dummy = df2_no_null.loc[:, 'Student Attendance Rate':'Student Achievement Rating']
dummy.head()
dummy.describe()
dummy['Student Attendance Rate'] = dummy['Student Attendance Rate'].str.replace('%','')
dummy['Percent of Students Chronically Absent'] = dummy['Percent of Students Chronically Absent'].str.replace('%','')
dummy['Rigorous Instruction %'] = dummy['Rigorous Instruction %'].str.replace('%','')
dummy['Collaborative Teachers %'] = dummy['Collaborative Teachers %'].str.replace('%','')
dummy['Supportive Environment %'] = dummy['Supportive Environment %'].str.replace('%','')
dummy['Effective School Leadership %'] = dummy['Effective School Leadership %'].str.replace('%','')
dummy['Strong Family-Community Ties %'] = dummy['Strong Family-Community Ties %'].str.replace('%','')
dummy['Trust %'] = dummy['Trust %'].str.replace('%','')
dummy['Student Attendance Rate'] = dummy['Student Attendance Rate'].astype(float)
dummy['Percent of Students Chronically Absent'] = dummy['Percent of Students Chronically Absent'].astype(float)
dummy['Rigorous Instruction %'] = dummy['Rigorous Instruction %'].astype(float)
dummy['Collaborative Teachers %'] = dummy['Collaborative Teachers %'].astype(float)
dummy['Supportive Environment %'] = dummy['Supportive Environment %'].astype(float)
dummy['Effective School Leadership %'] = dummy['Effective School Leadership %'].astype(float)
dummy['Strong Family-Community Ties %'] = dummy['Strong Family-Community Ties %'].astype(float)
dummy['Trust %'] = dummy['Trust %'].astype(float)
dummy['Student Attendance Rate'].fillna(value=dummy['Student Attendance Rate'].mean(),inplace=True)
dummy['Percent of Students Chronically Absent'].fillna(value=dummy['Percent of Students Chronically Absent'].mean(),inplace=True)
dummy['Rigorous Instruction %'].fillna(value=dummy['Rigorous Instruction %'].mean(),inplace=True)
dummy['Collaborative Teachers %'].fillna(value=dummy['Collaborative Teachers %'].mean(),inplace=True)
dummy['Supportive Environment %'].fillna(value=dummy['Supportive Environment %'].mean(),inplace=True)
dummy['Effective School Leadership %'].fillna(value=dummy['Effective School Leadership %'].mean(),inplace=True)
dummy['Strong Family-Community Ties %'].fillna(value=dummy['Strong Family-Community Ties %'].mean(),inplace=True)
dummy['Trust %'].fillna(value=dummy['Trust %'].mean(),inplace=True)
df2_no_null['Student Attendance Rate'] = dummy['Student Attendance Rate']
df2_no_null['Percent of Students Chronically Absent'] = dummy['Percent of Students Chronically Absent']
df2_no_null['Rigorous Instruction %'] = dummy['Rigorous Instruction %']
df2_no_null['Collaborative Teachers %'] = dummy['Collaborative Teachers %']
df2_no_null['Supportive Environment %'] = dummy['Supportive Environment %']
df2_no_null['Effective School Leadership %'] = dummy['Effective School Leadership %']
df2_no_null['Strong Family-Community Ties %'] = dummy['Strong Family-Community Ties %']
df2_no_null['Trust %'] = dummy['Trust %']
df2_no_null['Economic Need Index'].fillna(value=df2_no_null['Economic Need Index'].mean(),inplace=True)
df2_no_null.isnull().sum()
df2_no_null.shape
plt.figure(figsize=(12,10))
sns.set_style('white')
plt.scatter(x='Longitude', y='Latitude', data=df2_no_null, c='Economic Need Index', alpha=.5,\
            linewidth=2, s=df2_no_null['School Income Estimate']/250, cmap='RdBu_r')
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.title('Geo Mapping Economic Need Index and School Income Estimate', fontsize=20)

colorbar = plt.colorbar(orientation='vertical', shrink=.5)
colorbar.set_label('Economic Need Index')

l2 = plt.scatter([],[], s=150, edgecolors='none', c='r')
l3 = plt.scatter([],[], s=350, edgecolors='none', c='r')
l4 = plt.scatter([],[], s=450, edgecolors='none', c='r')

legend = ['50','350','450']

leg = plt.legend([l2, l3, l4], legend, frameon=False, fontsize=10, labelspacing=1.3,
handlelength=1,loc=2, handletextpad=2, title='School Income Estimate')
fig, axarr = plt.subplots(nrows =2, ncols=2, figsize=(14,12.5))
#ax1 = fig.add_subplot(111)

sns.set_style('white')
axarr[0,0].scatter(x='Longitude', y='Latitude', data=df2_no_null, c='Percent Asian', alpha=1,\
            linewidth=2, s=df2_no_null['School Income Estimate']/250, cmap='Reds')
axarr[0,0].set_title('Percent Asian')

axarr[0,1].scatter(x='Longitude', y='Latitude', data=df2_no_null, c='Percent Black', alpha=1,\
            linewidth=2, s=df2_no_null['School Income Estimate']/250, cmap='binary')
axarr[0,1].set_title('Percent Black')

axarr[1,0].scatter(x='Longitude', y='Latitude', data=df2_no_null, c='Percent Hispanic', alpha=1,\
            linewidth=2, s=df2_no_null['School Income Estimate']/250, cmap='YlOrBr')
axarr[1,0].set_title('Percent Hispanic')

axarr[1,1].scatter(x='Longitude', y='Latitude', data=df2_no_null, c='Percent White', alpha=1,\
            linewidth=2, s=df2_no_null['School Income Estimate']/250, cmap='Blues')
axarr[1,1].set_title('Percent White')

for ax in axarr.flat:
    ax.set(xlabel='Longitude', ylabel='Latitude')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axarr.flat:
    ax.label_outer()


df2_no_null['Zip_Prefix'] = df2_no_null['Zip']
df2_no_null['Zip_Prefix'] = df2_no_null['Zip_Prefix'].astype(str).str[:3]
df2_no_null['Borough'] = np.where(df2_no_null['Zip_Prefix'] == '100', 'Manhattan',\
        np.where(df2_no_null['Zip_Prefix'] == '101', 'Manhattan',\
                np.where(df2_no_null['Zip_Prefix'] == '102', 'Manhattan',\
                        np.where(df2_no_null['Zip_Prefix'] == '103', 'Staten Island',\
                                np.where(df2_no_null['Zip_Prefix'] == '104', 'Bronx',\
                                        np.where(df2_no_null['Zip_Prefix'] == '110', 'Queens',\
                                                np.where(df2_no_null['Zip_Prefix'] == '111', 'Queens',\
                                                        np.where(df2_no_null['Zip_Prefix'] == '112', 'Brooklyn',\
                                                                np.where(df2_no_null['Zip_Prefix'] == '113', 'Queens',\
                                                                        np.where(df2_no_null['Zip_Prefix'] == '114', 'Queens',\
                                                                                np.where(df2_no_null['Zip_Prefix'] == '116', 'Queens','')))))))))))
#test if Borough field is correct
dummy = df2_no_null.loc[:, ['School Income Estimate', 'Zip', 'Zip_Prefix','Borough']]
dummy[dummy['Zip_Prefix'] == '112'].head()
data1 = df2_no_null.loc[:, ['School Income Estimate', 'Community School?', 'Borough']].groupby(['Borough', 'Community School?']).sum().reset_index()
data2 = df2_no_null.sort_values(['School Income Estimate','Community School?'],ascending=[False,False])
order = df2_no_null['Borough'].value_counts().index

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

sns.set_style('whitegrid')

sns.barplot(x='Borough', y='School Income Estimate', hue='Community School?', data=data1.sort_values(['School Income Estimate','Community School?'],ascending=[False,False]), palette='pastel', ax=ax[0])\
.set_title('Community vs Non Community School Income')

sns.countplot(x='Borough', data=data2, palette='pastel', order=order, hue='Community School?',ax=ax[1]).set_title('Number of Schools')

plt.show()
plt.figure(figsize=(10,8))
sns.boxplot(x='Borough', y='Economic Need Index', hue='Community School?', data=df2_no_null.sort_values(['Community School?']), palette='pastel')
plt.show()
dataA = df2_no_null.loc[:,['Percent Asian', 'Borough', 'Percent of Students Chronically Absent']]

plt.figure(figsize=(10,8))
sns.lmplot(x='Percent Asian', y='Percent of Students Chronically Absent', data=dataA, col='Borough')

plt.show()
dataB = df2_no_null.loc[:,['Percent Black', 'Borough', 'Percent of Students Chronically Absent']]

plt.figure(figsize=(10,8))
sns.lmplot(x='Percent Black', y='Percent of Students Chronically Absent', data=dataB, col='Borough')

plt.show()
dataH = df2_no_null.loc[:,['Percent Hispanic', 'Borough', 'Percent of Students Chronically Absent']]

plt.figure(figsize=(10,8))
sns.lmplot(x='Percent Hispanic', y='Percent of Students Chronically Absent', data=dataH, col='Borough')

plt.show()
dataW = df2_no_null.loc[:,['Percent White', 'Borough', 'Percent of Students Chronically Absent']]

plt.figure(figsize=(10,8))
sns.lmplot(x='Percent White', y='Percent of Students Chronically Absent', data=dataW, col='Borough')

plt.show()
df2_ratings = df2_no_null.loc[:,['Borough','Student Attendance Rate','Rigorous Instruction %','Collaborative Teachers %','Supportive Environment %',\
                  'Effective School Leadership %','Strong Family-Community Ties %','Trust %','Student Achievement Rating','Rigorous Instruction Rating',\
                                'Collaborative Teachers Rating','Supportive Environment Rating','Effective School Leadership Rating',\
                                'Strong Family-Community Ties Rating','Trust Rating']]
df2_ratings['Student Attendance Rate'].replace(0,df2_ratings['Student Attendance Rate'].mean(), inplace=True)
df2_ratings['Rigorous Instruction %'].replace(0,df2_ratings['Rigorous Instruction %'].mean(), inplace=True)
df2_ratings['Collaborative Teachers %'].replace(0,df2_ratings['Collaborative Teachers %'].mean(), inplace=True)
df2_ratings['Supportive Environment %'].replace(0,df2_ratings['Supportive Environment %'].mean(), inplace=True)
df2_ratings['Effective School Leadership %'].replace(0,df2_ratings['Effective School Leadership %'].mean(), inplace=True)
df2_ratings['Strong Family-Community Ties %'].replace(0,df2_ratings['Strong Family-Community Ties %'].mean(), inplace=True)
df2_ratings['Trust %'].replace(0,df2_ratings['Trust %'].mean(), inplace=True)
plt.figure(figsize=(12,4))
sns.violinplot(x='Borough',y='Student Attendance Rate',data=df2_ratings,inner='box',palette='pastel')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.show()
plt.figure(figsize=(12,4))
ax = sns.violinplot(x='Borough',y='Rigorous Instruction %',data=df2_ratings,inner='box',\
                    palette='pastel',hue='Rigorous Instruction Rating')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.show()
plt.figure(figsize=(12,4))
sns.violinplot(x='Borough',y='Collaborative Teachers %',data=df2_ratings,inner='box',palette='pastel',\
               hue='Collaborative Teachers Rating')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.show()
plt.figure(figsize=(12,4))
sns.violinplot(x='Borough',y='Supportive Environment %',data=df2_ratings,inner='box',palette='pastel',\
               hue='Supportive Environment Rating')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.show()
plt.figure(figsize=(12,4))
sns.violinplot(x='Borough',y='Effective School Leadership %',data=df2_ratings,inner='box',palette='pastel',\
               hue='Effective School Leadership Rating')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.show()
plt.figure(figsize=(12,4))
sns.violinplot(x='Borough',y='Strong Family-Community Ties %',data=df2_ratings,inner='box',palette='pastel',\
               hue='Strong Family-Community Ties Rating')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.show()
plt.figure(figsize=(12,4))
sns.violinplot(x='Borough',y='Trust %',data=df2_ratings,inner='box',palette='pastel',hue='Trust Rating')
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
plt.show()