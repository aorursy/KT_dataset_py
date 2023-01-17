# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# load libs

import sys

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



# General

pd.set_option('display.max_columns', 500)

np.set_printoptions(sys.maxsize)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

pd.options.display.float_format = '{:,}'.format



# plot style

matplotlib.style.use('seaborn-pastel')



# common functions

def add_comma(value):

    return '{:,d}'.format(int(value))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        # print(os.path.join(dirname, filename))

        pass



# Any results you write to the current directory are saved as output.



# load full data set

loan =  pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv')



# drop all columns with all null values 

loan_no_nan = loan.dropna(axis=1, how = 'all')
cols = ['loan_amnt','funded_amnt','funded_amnt_inv']

compare_df = loan_no_nan[cols]



labels = [ "[0, 4450.0]", "[4450.0, 8400.0]", "[8400.0, 12350.0]", "[12350.0, 16300.0]", "[16300.0, 20250.0]",   

"[20250.0, 24200.0]", "[24200.0, 28150.0]", "[28150.0, 32100.0]","[32100.0, 36050.0]", "[36050.0, 40000.0]" ]



loan_amnt_cat = pd.cut( compare_df['loan_amnt'], 10, labels=labels )

loan_amnt_cat_asc = loan_amnt_cat.value_counts().sort_index()



funded_amnt_cat = pd.cut( compare_df['funded_amnt'], 10, labels=labels )

funded_amnt_cat_asc = funded_amnt_cat.value_counts().sort_index()



funded_amnt_inv_cat = pd.cut( compare_df['funded_amnt_inv'], 10, labels=labels )

funded_amnt_inv_cat_asc = funded_amnt_inv_cat.value_counts().sort_index()





# Styling plot

f, ax = plt.subplots(1,1,figsize=(18,12), sharex=True)

ax.set_title("Fig 1: Distribution of loan amounts vs loan amount invested",y=-.14,fontsize=18)



barWidth = 0.27



ax.grid(linestyle='-', linewidth=.5, alpha=0.5)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)





# Set position of bar on X axis

r1 = np.arange(len(labels))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

r4 = [x + barWidth for x in r3]



# Create blue bars

bar1 = plt.bar(r1, loan_amnt_cat_asc, width = barWidth, color = '#D4BEBE', edgecolor = 'black', capsize=7, label='loan_amnt')

bar2 = plt.bar(r2, funded_amnt_cat_asc, width = barWidth, color = '#ADA0A6', edgecolor = 'black', capsize=7, label='funded_amnt')

bar3 = plt.bar(r3, funded_amnt_inv_cat_asc, width = barWidth, color = '#CF8BA9', edgecolor = 'black', capsize=7, label='funded_amnt_inv')



h = []

def autolabel(rects):

    for rect in rects:

        ax.text(rect.get_x()+.10, rect.get_height() + 23000, str(add_comma(rect.get_height())), fontsize=10, rotation=50)



        

autolabel(bar1)

autolabel(bar2)

autolabel(bar3)



# general layout

plt.ylabel('Loan Amount')

plt.xticks([r + barWidth for r in range(len(labels))], labels, rotation=12)

plt.legend()

 

# Show graphic

plt.show()
features = ['loan_amnt','funded_amnt','funded_amnt_inv']

sub_df_3a = loan_no_nan.filter(items=features)



f_VS_finv = sub_df_3a.funded_amnt - sub_df_3a.funded_amnt_inv

bins = [1,100,200,300,400,500,1000,5000,10000,20000,max(f_VS_finv)]

# index = f_VS_finv > 10

# f_VS_finv = f_VS_finv.loc[index]

f_VS_finv_cat = pd.cut(f_VS_finv,bins=bins,include_lowest=True,precision=1)

f_VS_finv_cat_asc = f_VS_finv_cat.value_counts().sort_index()



# Styling plot

f, ax = plt.subplots(1,1,figsize=(20,10), sharex=True)

ax.set_title("Fig 2: Distribution of difference in loan amounts requested & funded",y=-.24,fontsize=18)



ax.grid(linestyle='-', linewidth=.5, alpha=0.5)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



# plot

f_VS_finv_asc = f_VS_finv.value_counts(bins=bins).sort_index(ascending=True)

f_VS_finv_asc.plot(kind='bar', rot=25, fontsize=14, ax=ax)



# annotation

for p in ax.patches:

    ax.annotate(str(add_comma(p.get_height())), (p.get_x() * 1.010, p.get_height() + 1000))



plt.show()
# Fig 3: Distribution of loan amounts asked by borrowers



# loan_amnt

loan_amnt = loan_no_nan.loan_amnt



# sort

loan_amnt_asc = loan_amnt.sort_values(ascending=True)



# Styling plot

f, ax = plt.subplots(figsize=(15,9))

ax.set_title("Fig 3: Distribution of loan amounts asked by borrowers",y=-.20,fontsize=18)



ax.grid(linestyle='-', linewidth=.5, alpha=0.5)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



# plot

loan_amnt_asc = loan_amnt.value_counts(bins=10).sort_index(ascending=True)

print(loan_amnt_asc)

loan_amnt_asc.plot(kind='bar', rot=12, fontsize=14)



# annotation

for p in ax.patches:

    ax.annotate(str(add_comma(p.get_height())), (p.get_x() * 1.010, p.get_height() + 4000))
# Fig 4: Loan amount & loan term



features  = ['term','funded_amnt_inv']

sub_df_4a = loan_no_nan.filter(items=features)



sub_df_4a.term = sub_df_4a.term.str.extract('(\d+)')



# Styling plot

f, ax = plt.subplots(1,1,figsize=(12,7))

ax.set_title("Fig 4: Loan amount & loan term",y=-.16,fontsize=18)





ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



# plot

sns.boxplot(x="term", y="funded_amnt_inv", data=sub_df_4a,width=0.3, ax=ax)
 # Fig 5: Loan Grades 

    

grade = loan_no_nan.grade.value_counts(normalize=False).rename_axis('type').to_frame('counts').reset_index()

print(grade)



# Styling plot

f, ax = plt.subplots(1,1,figsize=(10,10), sharex=False)

ax.set_title("Fig 6: Loan Grades",y=-.11,fontsize=18)



# Data to plot

labels = grade.type.unique()

sizes = grade.counts

explode = (0.0,0,0,0,0,0,0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=45)



plt.axis('equal')

plt.show()

## Fig x: Purpose of Loan 



title = ['Debt Consolidation','Credit Card Refinancing','Home Improvement','Other Reasons']

counts = ['1194386','482211','141151','350398']

percentage = [55.02,22.21,6.50,16.14]



purpose_of_loan = pd.DataFrame({'title':title,'counts':counts})

print(purpose_of_loan)         

# Styling plot

f, ax = plt.subplots(1,1,figsize=(10,10), sharex=False)

ax.set_title("Fig 7: Purpose of loan",y=-.11,fontsize=18)



# Data to plot

labels = purpose_of_loan.title.unique()

sizes = purpose_of_loan.counts

explode = (0.0,0,0,0)



# Plot

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=270)



plt.axis('equal')

plt.show()
# Fig 7: Home ownership/rent/mortgage Vs employee length



features  = ['emp_length','home_ownership']

sub_df_5a = loan_no_nan.filter(items=features)



#label = {idx:labels for labels,idx in enumerate(sub_df_5a.emp_length.unique())}

labels = {'10+ years': 10.0, '6 years': 6.0, '4 years': 4.0, '< 1 year': 0.9, '2 years': 2.0,'9 years': 9.0,

         'nan': 0, '5 years': 5.0, '3 years': 3.0, '7 years': 7.0, '1 year': 1.0,'8 years': 8.0}



sub_df_5a.emp_length = sub_df_5a.emp_length.map(labels)



bins = [0.9,5,9,10]

re_grouped = sub_df_5a.groupby([pd.cut(sub_df_5a.emp_length, bins),'home_ownership']).size()

re_grouped = re_grouped.reset_index()

re_grouped.columns = ['emp_length', 'home_ownership', 'counts']

re_grouped.drop(re_grouped.loc[re_grouped['home_ownership'].isin(['ANY','NONE','OTHER'])].index, inplace=True)

re_grouped = re_grouped.pivot(index='emp_length', columns='home_ownership', values='counts')



# Styling plot

f, ax = plt.subplots(1,1,figsize=(16,10), sharex=False)

ax.set_title("Fig 7: Home ownership/rent/mortgage Vs employee length",y=-.14,fontsize=18)



ax.grid(linestyle='-', linewidth=.2, alpha=0.1)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)



#re_grouped = re_grouped.sort_values(by='emp_length', ascending=True)

re_grouped.plot(kind='bar',ax=ax, rot=0)



# annotation

for p in ax.patches:

    ax.annotate(str(add_comma(p.get_height())), (p.get_x() * 1.012, p.get_height() + 2500))
# Fig 8: Heatmap of Borrowers



import plotly.offline as plot



features = ['addr_state','funded_amnt_inv']

sub_df_6a = loan_no_nan.filter(items=features)



# Data pre processing

states_funded = sub_df_6a.groupby(['addr_state']).sum().reset_index()

states_count = sub_df_6a.addr_state.value_counts().rename_axis('code').to_frame('counts').reset_index()

states = pd.merge(left=states_funded, right=states_count, left_on='addr_state', right_on='code')



# states.round(2)





# Plotting

states['text']= 'Amount : '+states['funded_amnt_inv'].astype(str)



# print('gia tri cua states',states)

data=[dict(type='choropleth',autocolorscale=False,locations=states['code'], z = states['counts']

           ,locationmode='USA-states',text= states['text'], colorscale='burgyl'

           ,colorbar=dict(title='Total Loans'))]



# print('gia tri data',data)



layout=dict(title='Fig 8: Heatmap of Borrowers', geo=dict(scope='usa'

        ,projection=dict(type='albers usa'),showlakes=True, lakecolor='rgb(66,165,245)',),)

# print('gia tri layout',layout)



fig = dict(data=data, layout = layout)

plot.iplot(fig, filename='d3-choropleth-map')
# Fig 0: Heatmap of Population Density
# Fig 10: Distribution of Debt to Income Ratios





features = ['dti']

sub_df_8 = loan_no_nan.filter(items=features)

print("median : {}".format(sub_df_8.dti.median()))



# remove outliers

from scipy import stats

z = (sub_df_8.dti - sub_df_8.dti.mean())/sub_df_8.dti.std(ddof=0)

sub_df_8.dti = sub_df_8.dti.loc[(z < 2)]

sub_df_8.describe().round(2)



# Styling plot

f, ax = plt.subplots(1,1,figsize=(12,7), sharex=True)

ax.set_title('Fig 10: Heatmap of Population Density',y=-0.16,fontsize=18)



ax.grid(linestyle='-', linewidth=.2, alpha=0.3)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

plt.legend(prop={'size': 16})



# plot

sns.distplot(sub_df_8.dti,norm_hist=True)
# Fig 11 : Distribution of loan amounts for borrowers with annual income > $100k



features = ['annual_inc','loan_amnt']

sub_df_7a = loan_no_nan.filter(items=features)



pd.set_option('precision', 2)

lsr_than_1million = sub_df_7a[sub_df_7a['annual_inc'] > 100000]

lsr_than_1million.describe().round(2)



bins = [min(lsr_than_1million.loan_amnt),10000,20000,30000,max(lsr_than_1million.loan_amnt)]

loan_amnt_cat = pd.cut(lsr_than_1million.loan_amnt,bins=bins,include_lowest=True,precision=1,duplicates='drop')

loan_amnt_cat_asc = loan_amnt_cat.value_counts().sort_index().rename_axis('loan_amnt').to_frame('count')



print(loan_amnt_cat_asc)





# Styling plot

f, ax = plt.subplots(1,1,figsize=(12,7), sharex=False)

ax.set_title('Fig 11 : Distribution of loan amounts for borrowers with annual income > $100k',y=-0.14,fontsize=18)



ax.grid(linestyle='-', linewidth=.2, alpha=0.1)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

plt.legend(prop={'size': 16})

plt.title('')

plt.xlabel('Loan amount')

plt.ylabel('No of Loan availed')



# Plot

ax = loan_amnt_cat_asc.plot(ax=ax,linestyle='-',color='DarkOrange',legend=False, marker='x',alpha=0.7)

loan_amnt_cat_asc.plot(kind='bar', rot=0, fontsize=14, ax=ax, alpha=0.5)





# annotation

for p in ax.patches:

    ax.annotate(str(add_comma(p.get_height())), (p.get_x() * 1.010, p.get_height() + 1500))
