import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



plt.style.use('seaborn-darkgrid')

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
drop_out_ratio=pd.read_csv('../input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')

drop_out_ratio.head(10)
drop_out_ratio[drop_out_ratio['Upper Primary_Boys']=='Uppe_r_Primary'].head(10)
drop_out_ratio['Upper Primary_Boys'].loc[0]='NR'
dor=drop_out_ratio.copy()
dor.info()

#here although it seems all have non null values, null values are represented as 'NR'
dor.set_index(['State_UT','year'],inplace=True)
def Replace(vals):

      if(vals=='NR'):

        return np.nan

      else:

        return float(vals)



# all the values are in str type. we must convert them to float64 type

dor=dor.loc[:,'Primary_Boys':].applymap(func=Replace)

dor.reset_index(inplace=True)



dor.head()
dor.info()
#some states have different number of spaces at different years. This is will tamper with grouping the data

def remove_spaces(vals):

  if (' ' in vals):

    return vals.replace(' ','')

  elif('  ' in vals):

    return vals.replace('  ','')

  else:

    return vals



dor['State_UT']=dor['State_UT'].apply(func=remove_spaces)

dor[dor['State_UT']=='Kerala']

# as you can see, a lot of data here is NaN across all three years. We must drop such rows since the mean() will be a NaN as well.
df=dor.groupby('State_UT').transform(lambda x: x.fillna(np.mean(x)))

df.insert(0,'State_UT',dor['State_UT'])

df.insert(1,'year',dor['year'])

df.dropna(inplace=True)

#df is the final, cleaned data.
dp=df.copy()

dp=dp.groupby('year').mean()

dp.reset_index(inplace=True)





boys=['Primary_Boys','Upper Primary_Boys','Secondary _Boys','HrSecondary_Boys']

girls=['Primary_Girls','Upper Primary_Girls','Secondary _Girls','HrSecondary_Girls']

dp_boys=pd.melt(dp,id_vars=['year'], value_vars=boys )

dp_girls= pd.melt(dp,id_vars=['year'],value_vars=girls)
plt.style.use('fivethirtyeight')



f,ax= plt.subplots(1,2,figsize=(18,12))

#sns barplot for boys

ax1= sns.barplot(x='year',y='value',hue='variable', data=dp_boys, palette='muted',edgecolor='black',ax=ax[0])

ax1.legend(fancybox=True,prop={'size':10})

ax1.set(ylim=(0,20))

ax[0].set_title('All India Drop Out Ratio For Boys')

ax[0].set_ylabel('Drop Out Ratio')



#sns barplot for girls

ax2=sns.barplot(x='year',y='value',hue='variable', data=dp_girls, palette='muted',edgecolor='black',ax=ax[1])

ax2.legend(fancybox=True,prop={'size':10})

ax2.set(ylim=(0,20))

ax[1].set_title('All India Drop Out Ratio For Girls')

ax[1].set_ylabel('Drop Out Ratio')

plt.show()
def plot_drop_out_wrt_state(df,state_name):

  %matplotlib inline

  plt.style.use('fivethirtyeight')



  df=df[df['State_UT']==state_name]

  years=df['year'].values

  #set barwidth

  bar_width=0.1

  #set figure

  fig=plt.figure()

  ax=fig.add_axes([5,5,2,2])

  #bar positions for X axis

  x_indexes= np.arange(len(df['Primary_Boys']))



  ax.bar(x_indexes,df['Primary_Girls'],width=bar_width,color='#FFC300',edgecolor='white', label='Primary Girls')

  ax.bar(x_indexes+bar_width,df['Primary_Boys'],width=bar_width,color='#DAF7A6',edgecolor='white', label='Primary Boys')

  ax.bar(x_indexes+2*bar_width,df['Upper Primary_Boys'],width=bar_width,color='#0071FB',edgecolor='white', label='Upper Primary Boys')

  ax.bar(x_indexes+3*bar_width,df['Upper Primary_Girls'],width=bar_width,color='#0BE1AC',edgecolor='white', label='Upper Primary Girls')

  ax.bar(x_indexes+4*bar_width,df['Secondary _Boys'],width=bar_width,color='#8B0BE1',edgecolor='white', label='Secondary Boys')

  ax.bar(x_indexes+5*bar_width,df['Secondary _Girls'],width=bar_width,color='#900C3F',edgecolor='white', label='Secondary Girls')

  ax.bar(x_indexes+6*bar_width,df['HrSecondary_Boys'],width=bar_width,color='#E10B7D',edgecolor='white', label='HrSecondary Boys')

  ax.bar(x_indexes+7*bar_width,df['HrSecondary_Girls'],width=bar_width,color='#FB0000',edgecolor='white', label='HrSecondary Girls')



  #adding ticks and label names

  plt.xticks([r+bar_width for r in x_indexes],years)# this creates the ticks at the x axis at the appropriate distance

  #plt.ylabel('Drop out Ratio')

  plt.xlabel('YEAR')



  #showing plot

  plt.title(state_name,fontweight='bold')

  plt.legend()

  plt.show()
dp=df.copy() # copy for plotting a bar plot

plot_drop_out_wrt_state(dp,'TamilNadu')
dp_hm=df.copy() # copy for heat map plotting

dp_hm.drop(['year','Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total'],axis=1,inplace=True)

dp_hm.drop(index=dp_hm[dp_hm['State_UT']=='AllIndia'].index,inplace=True)

dp_hm=dp_hm.groupby('State_UT').mean()

plt.style.use('fivethirtyeight')



ax=plt.subplots(figsize=(20,10))

ax=sns.heatmap(dp_hm,linewidths=0.02,annot=True)

plt.title('Heat Map')

plt.show()
gtoilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-girls-toilet-2013-2016.csv')

btoilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-boys-toilet-2013-2016.csv')
gtoilet.State_UT = gtoilet.State_UT.str.capitalize()

gtoil = gtoilet[gtoilet.State_UT == 'All india']

gtoil = gtoil.iloc[:,[0,1,2,5,9,11,12]]

gtoil = pd.melt(gtoil, id_vars=['State_UT', 'year'], value_vars= gtoil.iloc[:,2:6])



btoilet.State_UT = btoilet.State_UT.str.capitalize()

btoil = btoilet[btoilet.State_UT == 'All india']

btoil = btoil.iloc[:,[0,1,2,5,9,11,12]]

btoil = pd.melt(btoil, id_vars=['State_UT', 'year'], value_vars= btoil.iloc[:,2:6])
plt.style.use('fivethirtyeight')

f, axes = plt.subplots(1, 2, figsize=(20, 10))



ax1 = sns.barplot(x = 'year' , y = "value" ,hue = "variable", data = gtoil, palette = sns.cubehelix_palette(8), edgecolor = 'black',ax=axes[0])

ax1.set(ylim=(50, 120))

axes[0].set_title('Percentage of Schools with Girls Toilet',size = 20 , pad = 20)

axes[0].set_ylabel('Percentage')

ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

for p in ax1.patches:

             ax1.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=11.5, color='black', xytext=(0, 8),

                 textcoords='offset points')

ax2 = sns.barplot(x = 'year' , y = "value" ,hue = "variable", data = btoil, palette = 'Blues', edgecolor = 'black',ax=axes[1])

ax2.set(ylim=(50, 120))

axes[1].set_title('Percentage of Schools with Boys Toilet',size = 20 , pad = 20)

axes[1].set_ylabel('Percentage')

ax2.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)

for p in ax2.patches:

             ax2.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=11.5, color='black', xytext=(0, 8),

                 textcoords='offset points')
enroll=pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')

enroll.head(10)
def remove_spaces(vals):

  if (' ' in vals):

    return vals.replace(' ','')

  elif('  ' in vals):

    return vals.replace('  ','')

  else:

    return vals



enroll['State_UT']=dor['State_UT'].apply(func=remove_spaces)
#enroll.State_UT = enroll.State_UT.str.capitalize()

enroll = enroll.replace('NR', np.nan, regex=True)

enroll = enroll.replace('@', np.nan, regex=True)
for i in (enroll.columns[11:14]):

  enroll[i]=enroll[i].astype(float)

enroll.info()
dp=enroll.copy()

dp=dp.groupby('Year').mean()

dp.reset_index(inplace=True)



boys=['Primary_Boys','Upper_Primary_Boys','Secondary_Boys','Higher_Secondary_Boys']

girls=['Primary_Girls','Upper_Primary_Girls','Secondary_Girls','Higher_Secondary_Girls']

dp_boys=pd.melt(dp,id_vars=['Year'], value_vars=boys )

dp_girls= pd.melt(dp,id_vars=['Year'],value_vars=girls)



plt.style.use('fivethirtyeight')



f,ax= plt.subplots(1,2,figsize=(18,12))

#sns barplot for boys

ax1= sns.barplot(x='Year',y='value',hue='variable', data=dp_boys, palette='muted',edgecolor='black',ax=ax[0])

ax1.legend(fancybox=True,prop={'size':10})

ax1.set(ylim=(0,120))

ax[0].set_title('All India Gross Enrollment Ratio For Boys')

ax[0].set_ylabel('Gross Enrollment Ratio')



#sns barplot for girls

ax2=sns.barplot(x='Year',y='value',hue='variable', data=dp_girls, palette='muted',edgecolor='black',ax=ax[1])

ax2.legend(fancybox=True,prop={'size':10})

ax2.set(ylim=(0,120))

ax[1].set_title('All India Gross Enrollment Ratio For Girls')

ax[1].set_ylabel('Gross Enrollment Ratio')

plt.show()
total=['Primary_Total','Upper Primary_Total','Secondary _Total','HrSecondary_Total']

dp_total=pd.melt(enroll,id_vars=['Year'], value_vars=boys )



plt.style.use('fivethirtyeight')



f,ax= plt.subplots(figsize=(18,12))

#sns barplot for boys

ax1= sns.barplot(x='Year',y='value',hue='variable', data=dp_total, palette='Pastel2',edgecolor='black')

ax1.legend(fancybox=True,prop={'size':10})

ax1.set(ylim=(0,120))

ax.set_title('All India Gross Enrollment Ratio Total')

ax.set_ylabel('Gross Enrollment Ratio')



for p in (ax1.patches):

  ax1.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                 ha='center', va='center', fontsize=13.5, color='black', xytext=(0, 8),

                 textcoords='offset points')

plt.show()
