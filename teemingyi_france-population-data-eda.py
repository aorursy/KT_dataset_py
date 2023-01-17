import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pop_data = pd.read_csv('../input/population.csv')
pop_data.head()
pop_data.isna().describe()
print('There are {} unique towns.'.format(len(pop_data.CODGEO.unique())))
pop_data.drop('NIVGEO',axis=1, inplace=True)
print('The total population size is {:,}.'.format(pop_data.NB.sum()))
pop_data = pop_data.loc[pop_data.CODGEO.isin(pop_data.groupby('CODGEO').NB.sum().index[pop_data.groupby('CODGEO').NB.sum()!=0])]
geo_data = pd.read_csv('../input/base_etablissement_par_tranche_effectif.csv')
town_name={}
for i,x in enumerate(geo_data.CODGEO):
    town_name[x] = geo_data.LIBGEO.iloc[i]
pop_data['CODGEO'] = pop_data['CODGEO'].astype(str) 
pop_data['CODGEO'] = pop_data['CODGEO'].map(lambda x: '0'+x if len(x)<5 else x)
pop_df = pd.DataFrame()

pop_by_town = pop_data.groupby('CODGEO').NB.sum()
pop_by_town.describe()
print('The median population size is {} per town.'.format(int(pop_by_town.describe().iloc[5])))
pop_df['pop_size'] = pop_by_town
pop_df = pd.concat([pop_df, pop_data.groupby(['CODGEO','SEXE']).NB.sum().reset_index().pivot(index='CODGEO',columns='SEXE')],axis=1)
pop_df.columns = ['pop_size','male_pop','female_pop']
pop_df['male_propn'] = pop_df['male_pop']/pop_df['pop_size']
pop_df['female_propn'] = pop_df['female_pop']/pop_df['pop_size']
print('''{:.2%} of the French population is male ({:,}),
{:.2%} of the French population is female ({:,}).'''.format(pop_df.male_pop.sum()/pop_df.pop_size.sum(),pop_df.male_pop.sum(),
                                             pop_df.female_pop.sum()/pop_df.pop_size.sum(),pop_df.female_pop.sum()))
pop_df = pd.concat([pop_df, pop_data.groupby(['CODGEO','AGEQ80_17']).NB.sum().reset_index().pivot(index='CODGEO',columns='AGEQ80_17')],axis=1)
pop_df.columns = pop_df.columns[['NB' not in x for x in pop_df.columns]].tolist()+['age_'+str(x) for x in range(0,81,5)]
pop_df['children'] = pop_df['age_0']+pop_df['age_5']+pop_df['age_10']
pop_df['aged'] = pop_df['age_65']+pop_df['age_70']+pop_df['age_75']+pop_df['age_80']
pop_df['working'] = pop_df['pop_size']-pop_df['children']-pop_df['aged']
pop_df['children_propn'] = pop_df['children']/pop_df['pop_size']
pop_df['aged_propn'] = pop_df['aged']/pop_df['pop_size']
pop_df['working_propn'] = pop_df['working']/pop_df['pop_size']
pop_df['child_dependency_ratio'] = pop_df['children_propn']/pop_df['working_propn']
pop_df['aged_dependency_ratio'] = pop_df['aged_propn']/pop_df['working_propn']
pop_df['dependency_ratio'] = pop_df['child_dependency_ratio']+pop_df['aged_dependency_ratio']
print('''{:.2%} of the population is children,
{:.2%} of the population is working-age,
{:.2%} of the population is aged.
The child dependency ratio is {:.2%},
aged dependency ratio is {:.2%},
dependency ratio is {:.2%}.'''.format(pop_df.children.sum()/pop_df.pop_size.sum(),
                                        pop_df.working.sum()/pop_df.pop_size.sum(),
                                        pop_df.aged.sum()/pop_df.pop_size.sum(),
                                      pop_df.children.sum()/pop_df.working.sum(),
                                      pop_df.aged.sum()/pop_df.working.sum(),
                                      (pop_df.children.sum()+pop_df.aged.sum())/pop_df.working.sum(),
                                     ))
pop_df = pd.concat([pop_df, pop_data.groupby(['CODGEO','MOCO']).NB.sum().reset_index().pivot(index='CODGEO',columns='MOCO')],axis=1)
pop_df.columns = pop_df.columns[['NB' not in x for x in pop_df.columns]].tolist()+['children_w_both_parents','children_w_one_parent','couple_no_child','couple_w_child','single_adult_w_child','staying_in_non_family_home','staying_alone']
for x in ['children_w_both_parents','children_w_one_parent','couple_no_child','couple_w_child','single_adult_w_child','staying_in_non_family_home','staying_alone']:
    pop_df[x+'_propn'] = pop_df[x]/pop_df['pop_size']
print('{:.2%} of the population is staying alone.'.format(pop_df.staying_alone.sum()/pop_df.pop_size.sum()))
pop_df.columns
pop_df['pop_size_log'] = np.log(pop_df['pop_size'])
pop_df['gender_propn_diff'] = pop_df['male_propn'] - pop_df['female_propn']
pop_df['gender_propn_diff_abs'] = abs(pop_df['gender_propn_diff'])
columns = ['pop_size_log', 'male_propn','gender_propn_diff_abs',
       'child_dependency_ratio', 'aged_dependency_ratio', 'dependency_ratio',
       'children_w_both_parents_propn','children_w_one_parent_propn', 'couple_no_child_propn',
       'couple_w_child_propn', 'single_adult_w_child_propn','staying_alone_propn']
pop_fil = pop_df.loc[pop_df.pop_size>10000]
sns.pairplot(data=pop_fil[columns])
sns.scatterplot(x=pop_fil['gender_propn_diff_abs'],y=pop_fil['staying_alone_propn'])
plt.title('Difference in gender proportion against \n proportion of people staying alone')
plt.show()
sns.scatterplot(x=pop_fil['aged_dependency_ratio'],y=pop_fil['staying_alone_propn'])
plt.title('Aged dependency ratio against \n proportion of people staying alone')
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))
sns.scatterplot(x=pop_fil['gender_propn_diff_abs'],y=pop_fil['dependency_ratio'], ax=ax1)
ax1.set_title('Difference in gender proportion against \n dependency ratio')
sns.scatterplot(x=pop_fil['gender_propn_diff_abs'],y=pop_fil['child_dependency_ratio'], ax=ax2)
ax2.set_title('Difference in gender proportion against \n child dependency ratio')
sns.scatterplot(x=pop_fil['gender_propn_diff_abs'],y=pop_fil['aged_dependency_ratio'], ax=ax3)
ax3.set_title('Difference in gender proportion against \n aged dependency ratio')
plt.show()