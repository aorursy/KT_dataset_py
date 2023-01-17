import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re



# The series

f = open('../input/wm.series.csv', 'r')

d = []

for l in f.readlines():

    l = re.sub(', [a-zA-Z]', '_',str(l))

    d.append(l.split(','))

f.close()

series = pd.DataFrame(d[1:], columns = d[0])



# Area codes

f = open('../input/wm.area.csv', 'r')

d = []

for l in f.readlines():

    l = re.sub(', ', '_',str(l))

    d.append(l.split(','))

f.close()

area = pd.DataFrame(d[1:], columns = d[0])

area = area.iloc[:,[0,1]]

area = area.set_index('area_code')

area = area.to_dict()

area = area['area_text']



# Occupations

f = open('../input/wm.occupation.csv', 'r')

d = []

for l in f.readlines():

    l = re.sub(', [a-zA-Z]', '_',str(l))

    d.append(l.split(','))

f.close()

occ = pd.DataFrame(d[1:], columns = d[0])

occ = occ.iloc[:,[0,1]]

occ = occ.set_index('occupation_code')

occ = occ.to_dict()

occ = occ['occupation_text']



#Subcell codes 

subc = pd.read_csv('../input/wm.subcell.csv')

subc = subc.iloc[0:7,[0,1]]

#subc['subcell_code'] = subc.subcell_code.astype('int')

subc = subc.set_index('subcell_code')

subc = subc.to_dict()

subc = subc['subcell_text']



#Replace the codes with the text for them

series['occupation_code'] = series['occupation_code'].replace(occ)

series['subcell_code'] = series['subcell_code'].replace(subc)

series['area_code'] = series['area_code'].replace(area)



# Merge the hourly wage values

all_df = pd.read_csv('../input/wm.data.1.AllData.csv')

all_df= all_df.iloc[:,[0,1,3]]



series = pd.merge(series,all_df,left_on = ['series_id'], right_on = ['series_id'], how = 'left')
sns.boxplot(x = 'subcell_code', y = 'value', data = series, palette="Set3", hue = 'seasonal')

plt.xticks(rotation=90)
t = series.groupby('occupation_code').agg({'value':['mean','count','max','min',np.std]})

t = t.sort_values(by = [('value','mean')], ascending = False)

print('Top 10 occupations : \n')

sns.heatmap(t.head(10), annot = True, cbar = False)
print('Bottom 10 occupations : \n')

sns.heatmap(t.tail(10), annot = True, cbar = False)
series['state'] = series['area_code'].map(lambda x : x.split('_')[-1])

t = series.groupby('state').agg({'value':['mean','count','max','min',np.std]})

t = t.sort_values(by = [('value','mean')], ascending = False)

print('Top 10 States (highest hourly wage) : \n')

sns.heatmap(t.head(10), annot = True, cbar = False)
print('Bottom 10 States (lowest hourly wage) : \n')

sns.heatmap(t.tail(10), annot = True, cbar = False)
occ = series.groupby('occupation_code').agg({'value':['mean','count','max','min',np.std]})

occ = occ.sort_values(by = [('value','mean')], ascending = False)

top_occ =occ.index[0:10]

# Plotting the distribution of both the top and bottom

t = series.groupby(['occupation_code','state']).agg({'value':['mean']})

t = t.reset_index()

t.columns = ['occupation','location', 'mean_wage']

sub_t_a = t[t.occupation.isin(top_occ)]

low_occ = occ.index[-10:]

sub_t_b = t[t.occupation.isin(low_occ)]

sub_t = pd.concat([sub_t_a, sub_t_b])

plt.figure(figsize=(10,10))

sns.boxplot(y = 'occupation' , x = 'mean_wage', data = sub_t, palette="Set3")

plt.xticks(rotation=90)
lawyers = series[series.occupation_code == 'Lawyers']

lawyers = lawyers.groupby('state')['value'].mean().plot(rot = 45)