import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches



import seaborn as sns

import warnings

import math

import random as random

import os

print(os.listdir("../input"))


#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

#url1= "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"

#names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

#df= pd.read_csv(url, names = names, sep = '\s+')

#df.to_csv('auto.csv')

!ls

df = pd.read_csv('../input/auto.csv')

df.head()

df.isnull().sum()
df.info()
print('Chking if the Horsepower feature has ?,0,-,* strings')



[print(a,a in df.horsepower.unique()) for a in '?0-*'];
df[df['horsepower'].str.contains(r"\?")]
mask = np.column_stack([df[col].astype(str).str.contains(r"\?") for col in df])

print(df[mask])
df.loc[df['horsepower'].str.contains(r"\?"),'horsepower'] = np.nan

print('='*40)

missing = df.isna().sum().sort_values(ascending = False)

missing_percent = round(df.isna().sum().sort_values(ascending = False)*100/df.shape[0],2)

missing_train = pd.concat([missing, missing_percent], axis = 1, keys = ['Total', 'Percent'])

missing_train = missing_train[missing_train.Total != 0]

print("Missing values in the data set")

print(missing_train)



df.to_csv('auto_v1.csv')
df['horsepower'] = df['horsepower'].astype(float)

df.info()
print('Unique values in CYLINDERS feature are',np.sort(df.cylinders.unique()))

print('Unique values in ORIGIN feature are',np.sort(df.origin.unique()))

print('Unique values in MODEL_YEAR feature are',np.sort(df.model_year.unique()))
cylinders_cat_dtype = pd.api.types.CategoricalDtype(categories=np.sort(df.cylinders.unique()), ordered=True)

origin_cat_dtype = pd.api.types.CategoricalDtype(categories=np.sort(df.origin.unique()), ordered=False)

model_year_cat_dtype = pd.api.types.CategoricalDtype(categories=np.sort(df.model_year.unique()), ordered=True)

df['cat_cyl'] = df.cylinders.astype(cylinders_cat_dtype)

df['cat_org'] = df.origin.astype(origin_cat_dtype)

df['cat_year'] = df.model_year.astype(model_year_cat_dtype)
#print('chevrolet cars',df[df['car_name'].str.contains(r"chevrolet")].shape[0])

#print('buick cars',df[df['car_name'].str.contains(r"buick")].shape[0])

#print('plymouth cars',df[df['car_name'].str.contains(r"plymouth")].shape[0])

#print('amc cars',df[df['car_name'].str.contains(r"amc")].shape[0])

#print('ford cars',df[df['car_name'].str.contains(r"ford")].shape[0])
brand = {}

for car_name in df['car_name']:

    if car_name.split(" ")[0] not in brand:

        brand[car_name.split(" ")[0]] = 1

    else:

        brand[car_name.split(" ")[0]] = brand[car_name.split(" ")[0]] + 1

print('Total number of companies',len(brand))

#print(brand)

Car_bands = pd.DataFrame.from_dict(brand, orient='index')

Car_bands.columns = ['No_cars']

#Car_bands

print("Top ten occurring companies")

#print(Car_bands['No_cars'].sort_values(ascending = False).head(10))

temp = Car_bands['No_cars'].sort_values(ascending = False)

temp.index.name = 'Company'

temp.column = ['No_cars']

temp.head(10)
fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (18,12))

plt.plot(temp, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

ax.set_title('# of Cars for a given company',fontsize=20)

ax.set_xlabel('Company',fontsize=20)

ax.set_ylabel('# of Cars',fontsize=20)

plt.xticks(rotation = '60', fontsize=15)

plt.show()

df['company'] = df['car_name'].apply(lambda x: x.split(' ')[0])

df.head()
company_cat_dtype = pd.api.types.CategoricalDtype(categories=np.sort(df.company.unique()), ordered=False)

df['company'] = df.company.astype(company_cat_dtype)

df.info()
temp2 = df.groupby('company')['mpg'].mean().sort_values(ascending = False)

company_avg_mpg = pd.merge(temp.to_frame(), temp2.to_frame(), left_index=True, right_index=True)

company_avg_mpg = company_avg_mpg.sort_values('mpg', ascending = False)

#company_avg_mpg.head()
fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (18,12))

plt.plot('No_cars', data = company_avg_mpg,  color='red', marker='.', linestyle='none', linewidth=2, markersize=12)

plt.plot('mpg', data = company_avg_mpg, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

sns.barplot(

    x='company',

    y='mpg',

    order = company_avg_mpg.index,

    data=df,

#palette=pal

)

ax.set_title('Average mile per gallon for each company',fontsize=20)

ax.set_xlabel('Company',fontsize=20)

ax.set_ylabel('mpg',fontsize=20)

plt.xticks(rotation = '90', fontsize=15)

for i, v in enumerate(company_avg_mpg["No_cars"].iteritems()):        

#    ax.text(i ,v[1], "{:,}".format(v[1]), color='m', va ='bottom', rotation=45)

    ax.text(i ,5, "{:,}".format(v[1]), color='b', va ='bottom', rotation=0)

plt.legend()

plt.show()
temp3 = df.groupby('company')['mpg'].agg(['mean', 'count', 'std'])

ci95_hi = []

ci95_lo = []

ci95 = []

for i in temp3.index:

    m, c, s = temp3.loc[i,['mean', 'count', 'std']]

    ci95_hi.append(m + 1.96*s/math.sqrt(c))

    ci95_lo.append(m - 1.96*s/math.sqrt(c))

    ci95.append(2*1.96*s/math.sqrt(c))



temp3['ci95_hi'] = ci95_hi

temp3['ci95_lo'] = ci95_lo

temp3['ci95'] = ci95

company_avg_mpg = pd.merge(temp.to_frame(), temp3, left_index=True, right_index=True)

company_avg_mpg = company_avg_mpg.sort_values('ci95', ascending = True)



company_avg_mpg = company_avg_mpg[company_avg_mpg['No_cars']>0]

fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (18,12))

#plt.plot('No_cars', data = company_avg_mpg,  color='red', marker='.', linestyle='none', linewidth=2, markersize=12)

plt.plot('mpg', data = company_avg_mpg, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

sns.barplot(

    x='company',

    y='mpg',

    order = company_avg_mpg.index,

    data=df,

    #palette=sns.color_palette()#sns.color_palette("BuGn_r")

)

ax.set_title('Average mile per gallon for each company (in the increasing order of confidence intervals)',fontsize=20)

ax.set_xlabel('Company',fontsize=20)

ax.set_ylabel('avg mpg',fontsize=20)

plt.xticks(rotation = '90', fontsize=15)

for i,v in enumerate(company_avg_mpg[["ci95","mean","No_cars"]].values):

    ax.text(i ,v[1], "{:,}".format(round(v[0]/2,1)), color='m', va ='bottom', rotation=45, fontsize = 12)

    ax.text(i ,5, "{:,}".format(int(v[2])), color='b', va ='bottom', rotation=0, fontsize = 12)

#plt.legend()

red_patch = mpatches.Patch(color='m', label='95% CI')

blue_patch = mpatches.Patch(color='b', label='# of Cars')



plt.legend(handles=[red_patch, blue_patch])

plt.show()
temp1 = df.groupby('cat_cyl')['mpg'].mean().sort_values(ascending = False)

temp = pd.merge(temp1.reset_index(), 

         temp1.reset_index()['cat_cyl'].apply(lambda x: temp1[x]/x).to_frame(),

         left_index=True, 

         right_index=True)

temp.columns= ['cyl', 'mpg', 'mpg_per_cyl']

temp = temp.sort_values('cyl')

print(temp)
df['mpg_per_cyl'] = df['mpg']/df['cylinders']
fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,8))

sns.barplot(

    x='cat_cyl',

    y='mpg',

    saturation=.25,

    #order = df.cat_cyl,

    data=df,

#palette=pal

)

sns.barplot(

    x='cat_cyl',

    y='mpg_per_cyl',

    saturation=1,

    #order = df.cat_cyl,

    data=df,

    #palette=sns.color_palette('white')

)

ax.set_title('mpg vs # of cylinders',fontsize=20)

ax.set_xlabel('number of cylinders',fontsize=20)

ax.set_ylabel('Average mileage',fontsize=20)

plt.xticks(rotation = '0', fontsize=15)

for i,v in enumerate(temp.values):

    ax.text(i,v[1], "{:,}".format(round(v[1],1)), color='m', va ='bottom', rotation=0, fontsize = 12)

    ax.text(i ,v[2], "{:,}".format(round(v[2],1)), color='b', va ='bottom', rotation=0, fontsize = 12)

red_patch = mpatches.Patch(color='m', label='avg mpg')

blue_patch = mpatches.Patch(color='b', label='avg mpg per cylinder')



plt.legend(handles=[red_patch, blue_patch])

plt.show()
temp = round(pd.pivot_table(df, 

                            values='mpg', 

                            index=['cat_cyl'], 

                            columns='origin', 

                            aggfunc=[lambda x: len(x),'mean'] ),2)

temp.rename(columns={'<lambda>': '# of cars', 'mean':'average mileage'}, inplace=True)

temp.fillna('-', inplace = True)

print(temp)

print('_'*80+'\n')

fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,8))

sns.barplot(

    x='cat_cyl',

    y='mpg',

    #saturation=.25,

    hue = 'origin',

    data=df,

#palette=pal

)

ax.set_title('mpg vs # of cylinders',fontsize=20)

ax.set_xlabel('# of cylinders',fontsize=20)

ax.set_ylabel('Average mileage',fontsize=20)

plt.xticks(rotation = '0', fontsize=15)

plt.legend()

plt.show()
fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,8))

sns.boxplot(

    x='cat_cyl',

    y='mpg',

    #saturation=.25,

    hue = 'origin',

    data=df,

#palette=pal

)

ax.set_title('mpg vs # of cylinders',fontsize=20)

ax.set_xlabel('# of cylinders',fontsize=20)

ax.set_ylabel('mileage',fontsize=20)

plt.xticks(rotation = '0', fontsize=15)



plt.show()
df.info()
sns.pairplot(df[['mpg', 'displacement', 'horsepower', 'weight',

       'acceleration']])

plt.show()

sns.pairplot(df[['mpg', 'displacement', 'horsepower', 'weight',

       'acceleration','origin']], hue = 'origin')

plt.show()
sns.pairplot(df[['mpg', 'displacement', 'horsepower', 'weight',

       'acceleration','model_year']], hue = 'model_year')

plt.show()
sns.pairplot(df[['mpg', 'displacement', 'horsepower', 'weight',

       'acceleration','cylinders']], hue = 'cylinders')

plt.show()
sns.pairplot(df[['mpg', 'displacement', 'horsepower', 'weight',

       'acceleration','company']], hue = 'company')

plt.show()
sns.jointplot(x='mpg',y='displacement',data=df, kind='reg')

# "scatter" | "reg" | "resid" | "kde" | "hex"

sns.set(style="ticks")

# Display the plot

plt.show()
#grid = sns.FacetGrid(df, row='cylinders', col = 'origin', size=2.5, aspect=1.6)

#grid.map(plt.scatter, "mpg", "displacement", alpha=.7)

#grid.add_legend()

#plt.show()
def Scatter_dist_vs_mpg(df, x = 'displacement', hue = 'cat_cyl'):

    number_of_colors = len(df[hue].unique())

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

             for i in range(number_of_colors)]

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize = (16,12))

    patch = []

    for i, item in enumerate(df[hue].unique()):

        data = df[df[hue] == item]

        ax[0].scatter(data[x], data['mpg'],  c = color[i], marker ='o', s = 100)

        ax[1].scatter(1/data[x], data['mpg'], c = color[i], marker ='o', s = 100)

        patch.append(mpatches.Patch(color=color[i], label=hue +' = {}'.format(item)))

    ax[0].set_title('mpg vs {}'.format(x), fontsize=20)

    ax[0].set_xlabel('{}'.format(x), fontsize=20)

    ax[0].set_ylabel('mpg', fontsize=20)

    ax[1].set_title('mpg vs 1/{}'.format(x), fontsize=20)

    ax[1].set_xlabel('{}'.format(x), fontsize=20)

    ax[1].set_ylabel('mpg', fontsize=20)

    plt.legend(handles=patch)

    plt.show()

    

def sns_lmplot(df, y ='mpg', x = 'displacement', col = None, hue =None):

    data= df.copy()

    data['i_x'] = 1/data[x]

    sns.lmplot(x = x, y = 'mpg', data = data, col = col, hue = hue,sharex=False,sharey=False)

    sns.lmplot(x = 'i_x', y = 'mpg', data = data, col = col, hue = hue,sharex=False,sharey=False)

    plt.show()
Scatter_dist_vs_mpg(df, x = 'displacement', hue = 'cylinders')

sns_lmplot(df, x = 'displacement')



sns_lmplot(df, x = 'displacement', col = 'cylinders')

sns_lmplot(df, x = 'displacement', hue = 'cylinders')
Scatter_dist_vs_mpg(df, hue = 'origin')
Scatter_dist_vs_mpg(df, x = 'horsepower', hue = 'origin')
Scatter_dist_vs_mpg(df, x = 'horsepower', hue = 'cylinders')
df.columns
Scatter_dist_vs_mpg(df, x = 'weight', hue = 'origin')
Scatter_dist_vs_mpg(df, x = 'weight', hue = 'cylinders')
Scatter_dist_vs_mpg(df, x = 'acceleration', hue = 'origin')
Scatter_dist_vs_mpg(df, x = 'acceleration', hue = 'cylinders')
df.columns
print('Work on progress')
print('Work on progress')