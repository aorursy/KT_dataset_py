import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Reading the file to enironment for analysis
df = pd.read_csv('../input/cities_r2.csv')
df.shape
df.columns # Display variout columns 
df.head()
df1 = df[['name_of_city', 'population_total',  'total_graduates' ]].sort_values(by = 'total_graduates', ascending = False)
df1['lit_ratio'] = df['total_graduates']/df['population_total']*100
df1['lit_ratio'] = df1['lit_ratio'].astype(int)
df1 = df1.sort_values(by = 'lit_ratio', ascending = False)
df1 = df1.reset_index(drop = True)
df1.head(10)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
fig.set_size_inches(15, 5)
sns.barplot(x = 'name_of_city' , y = 'lit_ratio', data = df1.head(10))
df2 = df[['state_name', 'total_graduates']]
df2 = df2.sort_values(by = 'total_graduates', ascending = False)
table1 = df2.pivot_table(index = 'state_name', values = 'total_graduates', aggfunc = sum)
df3 = pd.DataFrame()
df3['state'] = np. array(list(table1.index))
df3['Graduates'] = np.array(table1['total_graduates'])
df3 = df3.sort_values(by = 'Graduates', ascending = False)
df3.reset_index(drop = True)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
fig.set_size_inches(15, 5)
sns.barplot(x = 'state', y = 'Graduates',  data = df3.head(10))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
fig.set_size_inches(15, 5)
sns.barplot(x = 'state', y = 'Graduates',  data = df3.tail(10))
men_grad = df[['name_of_city', 'population_male', 'male_graduates']]
men_grad = men_grad.sort_values(by = 'male_graduates', ascending = False)
men_grad = men_grad.reset_index(drop = True)
men_grad['ratio'] = (men_grad['male_graduates']/ men_grad['population_male']) *100
men_grad['ratio'] = men_grad['ratio'].astype(int)
men_grad = men_grad.sort_values(by = 'male_graduates', ascending = False)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
fig.set_size_inches(15, 5)
sns.barplot(x = 'name_of_city', y = 'male_graduates', data = men_grad.head(10))
men_grad = men_grad.sort_values(by = 'ratio', ascending = False)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
fig.set_size_inches(15, 5)
sns.barplot(x = 'name_of_city', y = 'ratio', data = men_grad.head(10))
women_grad = df[['name_of_city', 'population_female', 'female_graduates']]
women_grad = women_grad.sort_values(by = 'female_graduates', ascending = False)
women_grad = women_grad.reset_index(drop = True)
women_grad['ratio'] = (women_grad['female_graduates']/ women_grad['population_female']) *100
women_grad['ratio'] = women_grad['ratio'].astype(int)
women_grad = women_grad.sort_values(by = 'female_graduates', ascending = False)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
fig.set_size_inches(15, 5)
sns.barplot(x = 'name_of_city', y = 'female_graduates', data = women_grad.head(10))
women_grad = women_grad.sort_values(by = 'ratio', ascending = False)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
fig.set_size_inches(15, 5)
sns.barplot(x = 'name_of_city', y = 'ratio', data = women_grad.head(10))

#polulation Vs Literates 
pop_lit = df[['name_of_city', 'state_name', 'effective_literacy_rate_total' ]]
pop_lit = pop_lit.sort_values(by= 'effective_literacy_rate_total', ascending = False)
pop_lit= pop_lit.reset_index(drop = True)

pop_lit.pivot_table(index ='state_name' , values ='effective_literacy_rate_total' ,  aggfunc = np.mean).sort_values(by = 'effective_literacy_rate_total', ascending = False)
fig, axes = plt.subplots(nrows=8, ncols=4)
axes_list = [item for sublist in axes for item in sublist] 
for st  in list(pop_lit['state_name'].unique()):
    ax = axes_list.pop(0)
    a = pop_lit[pop_lit['state_name'] == st]
    city_max = a[a['effective_literacy_rate_total'] == a['effective_literacy_rate_total'].max()]
    city_min = a[a['effective_literacy_rate_total'] == a['effective_literacy_rate_total'].min()]
    min_max = pd.concat([city_max, city_min])
    min_max.plot(x = min_max['name_of_city'], kind = 'bar', ax = ax, legend = False, title = st, figsize=(20,50),
                 fontsize = 15,sharey = True)
    ax.set_xlabel('')
    fig.tight_layout()
for ax in axes_list:
    ax.remove()     
plt.show()


