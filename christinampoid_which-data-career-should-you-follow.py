# load the necessary modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
# configure the seaborn aesthetics
sns.set_context('notebook')
sns.set(style="whitegrid")
# import our dataframe skipping the row holding the questions
df_multiplechoice = pd.read_csv('../input/multipleChoiceResponses.csv', 
                                low_memory=False,
                                skiprows=[1])
df_multiplechoice.head(5)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
g = sns.countplot(y='Q6', 
                  data=df_multiplechoice,
                  order = df_multiplechoice['Q6'].value_counts().index,
                  palette= sns.cubehelix_palette(22, start=2, rot=0, dark=.3, light=.8, reverse=True))
g.set_xlabel('')
g.set_ylabel('')
g.tick_params(labelsize=14)
sns.despine()
df_title_gender = df_multiplechoice.groupby('Q6')['Q1'].value_counts().to_frame().unstack()
df_title_gender.columns = df_title_gender.columns.droplevel()
df_title_gender = df_title_gender.reset_index()
df_title_gender = df_title_gender.fillna(0)
df_title_gender
g = sns.PairGrid(df_title_gender.sort_values('Prefer not to say', ascending=False),
                 
                 x_vars=['Male', 'Female', 'Prefer not to say', 'Prefer to self-describe'], 
                 y_vars=['Q6'],
                 
                 )
g.fig.set_size_inches(16,10)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
       linewidth=1, palette=sns.cubehelix_palette(22, start=2, rot=0, dark=.3, light=.8, reverse=True), edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(-100, 4300), xlabel="", ylabel="")

# Use semantically meaningful titles for the columns
titles = ["Male", "Female", "Prefer not to say", "Prefer to self-describe"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
sns.despine(left=True, bottom=True)
sns.set(style="whitegrid")
df_title_gender['rest'] = df_title_gender['Female'] + df_title_gender['Prefer not to say'] + df_title_gender['Prefer to self-describe']
g = sns.PairGrid(df_title_gender.sort_values('rest', ascending=False),
                 
                 x_vars=['Male', 'rest'], 
                 y_vars=['Q6'],
                 
                 )
g.fig.set_size_inches(16,8)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
       linewidth=1, palette=sns.cubehelix_palette(22, start=2, rot=0, dark=.3, light=.8, reverse=True), edgecolor="w")

# Use the same x axis limits on all columns and remove labels
g.set(xlim=(-100, 4300), xlabel="", ylabel="")

# Use semantically meaningful titles for the columns
titles = ["Male", "Female/Prefer not to say/Prefer to self-describe"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
sns.despine(left=True, bottom=True)
df_multiplechoice['Q3'] = df_multiplechoice['Q3'].replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
df_multiplechoice['Q3'] = df_multiplechoice['Q3'].replace('I do not wish to disclose my location', 'Not disclosed')
df_multiplechoice['Q3'] = df_multiplechoice['Q3'].replace('United States of America', 'USA')
df_multiplechoice['Q3'] = df_multiplechoice['Q3'].replace('Iran, Islamic Republic of...', 'Iran')

df_title_country = df_multiplechoice.groupby('Q6')['Q3'].value_counts().to_frame()
df_title_country.columns = ['count']
df_title_country = df_title_country.reset_index()
df_title_country.sort_values(['Q3', 'Q6']).head(20)
colors =sns.color_palette("Paired", 11)
order = df_title_country['Q3'].unique()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
ax = sns.stripplot(ax=ax1, x='count', y="Q3", hue='Q6', data=df_title_country[0:498],
                   palette=colors, jitter=True, size=7, order=order)
ax = sns.stripplot(ax=ax2, x='count', y="Q3", hue='Q6', data=df_title_country[498:],
                   palette=colors, jitter=True, size=7, order=order)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)
ax2.xaxis.grid(True)
ax2.yaxis.grid(True)
sns.set_style('whitegrid')
plt.legend()
df_title_edu = pd.crosstab(df_multiplechoice['Q4'], df_multiplechoice['Q6']).apply(lambda r: r/r.sum()*100, axis=0)
df_title_edu = df_title_edu.unstack().reset_index()
df_title_edu = df_title_edu.rename(columns = {0: 'count'})
df_title_edu.head(10)
colors =sns.color_palette("Paired", 7)
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
#order = df_title_edu['Q6'].unique()
df_title_edu = df_title_edu.sort_values(['Q4', 'count'])
ax = sns.stripplot(x='count', y="Q6", hue='Q4', data=df_title_edu,
                   palette=colors, jitter=True, size=8)
ax.set_xlabel('Percentage of people on each profession')
ax.set_ylabel('')
ax.xaxis.grid(True)
ax.yaxis.grid(True)

plt.legend()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), gridspec_kw = {'width_ratios':[10, 1]})
df_heatmap = df_multiplechoice.loc[df_multiplechoice['Q6']!='Student']
df_heatmap = df_heatmap[['Q6', 'Q9']].pivot_table(index=['Q6'], columns='Q9', aggfunc=np.count_nonzero)
df_heatmap['sum'] = df_heatmap.sum(axis=1)
df_heatmap.sort_values('0-10,000', inplace=True, ascending=False)

#df_heatmap.drop(columns=['sum'], inplace=True)
df_heatmap1 = 100*df_heatmap[['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000',
                         '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
                         '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000',
                         '300-400,000', '400-500,000', '500,000+']].div(df_heatmap['sum'], axis=0)


df_heatmap1 = df_heatmap1.rename(columns={"0-10,000": "0-10", "10-20,000": "10-20", '20-30,000': '20-30',
                                        '30-40,000': '30-40', '40-50,000': '40-50', '50-60,000': '50-60',
                                        '60-70,000': '60-70', '70-80,000': '70-80', '80-90,000': '80-90',
                                        '90-100,000': '90-100', '100-125,000': '100-125', '125-150,000':'125-150',
                                        '150-200,000': '150-200', '200-250,000': '200-250', '250-300,000': '250-300',
                                        '300-400,000': '300-400', '400-500,000': '400-500', '500,000+': '500+'
                                       })

g = sns.heatmap(df_heatmap1, ax=ax1, annot=True,cmap=sns.cubehelix_palette(200, start=2, rot=0, dark=.05, light=.95, reverse=False))
g.set_ylabel('')
g.set_xlabel('Salary in thousands')
g.tick_params(labelsize=14)
g.set_title('Salary earned by percentage of people per job title', fontsize=16)

df_heatmap2 = 100*df_heatmap['I do not wish to disclose my approximate yearly compensation'].to_frame().rename(columns={'I do not wish to disclose my approximate yearly compensation': 'Not disclosed'}).div(df_heatmap['sum'], axis=0)

g = sns.heatmap(df_heatmap2, ax=ax2, annot=True, cmap=sns.cubehelix_palette(200, start=2, rot=0, dark=.05, light=.95, reverse=False))
g.set_ylabel('')
g.tick_params(labelsize=14)
g.set_title('Percentage of people per job title \n not disclosing their salary', fontsize=16)


columns_q11 = [column for column in df_multiplechoice.columns if column.startswith('Q11_Part')]
columns_q11.append('Q6')
df_role = df_multiplechoice[columns_q11]
columns_q11.remove('Q6')
for column in columns_q11:
    df_temp = df_role[column]
    value = df_temp.loc[df_temp.first_valid_index()]
    df_role = df_role.rename(columns = {column: value})
df_role.head(5)
groups_role = df_role.groupby('Q6')
groups_role_pct = pd.DataFrame()
for name, group in groups_role:
    groups_role_pct[name] = pd.Series(group.notnull().mean()*100)
groups_role_pct = groups_role_pct.unstack().to_frame().reset_index()
groups_role_pct = groups_role_pct.loc[groups_role_pct['level_1'] !='Q6']
groups_role_pct = groups_role_pct.rename(columns = {'level_0': 'Q3', 'level_1': 'role', 0: 'count'})
groups_role_pct.head(10)
colors =sns.color_palette("Paired", 14)
groups_role_pct = groups_role_pct.sort_values(['count'], ascending=False)
order = groups_role_pct['Q3'].unique()
fig, ax = plt.subplots(figsize=(16, 10))
ax = sns.stripplot(ax=ax, x='count', y="Q3", hue='role', data=groups_role_pct,
                   palette=colors, jitter=False, size=8, alpha=.75, 
                   edgecolor="black", order=order)
ax.set_xlabel('Percentage of people')
ax.set_ylabel('')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.tick_params(labelsize=14)

plt.legend()
columns_q13 = [column for column in df_multiplechoice.columns if column.startswith('Q13_Part')]
columns_q13.append('Q6')
df_ide = df_multiplechoice[columns_q13]
columns_q13.remove('Q6')
for column in columns_q13:
    df_temp = df_ide[column]
    value = df_temp.loc[df_temp.first_valid_index()]
    df_ide = df_ide.rename(columns = {column: value})
df_ide.head(5)
groups_ide = df_ide.groupby('Q6')
df_ide_pct = pd.DataFrame()
for name, group in groups_ide:
    df_ide_pct[name] = pd.Series(group.notnull().mean()*100)
df_ide_pct = df_ide_pct.unstack().to_frame().reset_index()
df_ide_pct = df_ide_pct.loc[df_ide_pct['level_1'] !='Q6']
df_ide_pct = df_ide_pct.rename(columns = {'level_0': 'role', 'level_1': 'ide', 0: 'count'})
df_ide_pct.head(10)

colors = sns.color_palette("Set2", 7)
colors2 = sns.color_palette("Paired", 8)
colors.extend(colors2)
df_ide_pct = df_ide_pct.sort_values(['ide', 'count'], ascending=False)
order = df_ide_pct['role'].unique()
fig, ax = plt.subplots(figsize=(16, 10))
ax = sns.stripplot(ax=ax, x='count', y="role", hue='ide', data=df_ide_pct,
                   palette=colors, jitter=False, size=8, alpha=.75, 
                   edgecolor="black")
ax.set_xlabel('Percentage of people')
ax.set_ylabel('')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.tick_params(labelsize=14)

plt.legend()
columns_q21 = [column for column in df_multiplechoice.columns if column.startswith('Q21_Part')]
columns_q21.append('Q6')
df_datavis = df_multiplechoice[columns_q21]
columns_q21.remove('Q6')
for column in columns_q21:
    df_temp = df_datavis[column]
    value = df_temp.loc[df_temp.first_valid_index()]
    df_datavis = df_datavis.rename(columns = {column: value})
df_datavis.head(5)
groups_datavis = df_datavis.groupby('Q6')
df_datavis_pct = pd.DataFrame()
for name, group in groups_datavis:
    df_datavis_pct[name] = pd.Series(group.notnull().mean()*100)
df_datavis_pct = df_datavis_pct.unstack().to_frame().reset_index()
df_datavis_pct = df_datavis_pct.loc[df_datavis_pct['level_1'] !='Q6']
df_datavis_pct = df_datavis_pct.rename(columns = {'level_0': 'Q3', 'level_1': 'datavis', 0: 'count'})
df_datavis_pct.head(10)

colors =sns.color_palette("Paired", 10)
colors2 = sns.color_palette("Set2", 10)
colors.extend(colors2)
df_datavis_pct = df_datavis_pct.sort_values(['datavis', 'count'])
order = df_datavis_pct['Q3'].unique()
fig, ax = plt.subplots(figsize=(20, 10))
ax = sns.stripplot(ax=ax, x='count', y="Q3", hue='datavis', data=df_datavis_pct,
                   palette=colors, jitter=False, size=10, alpha=.75, 
                   edgecolor="black", order=order)
ax.set_xlabel('Percentage of people')
ax.set_ylabel('')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.tick_params(labelsize=14)

plt.legend()
columns_q16 = [column for column in df_multiplechoice.columns if column.startswith('Q16_Part')]
columns_q16.append('Q6')
df_langs = df_multiplechoice[columns_q16]
columns_q16.remove('Q6')
for column in columns_q16:
    df_temp = df_langs[column]
    value = df_temp.loc[df_temp.first_valid_index()]
    df_langs = df_langs.rename(columns = {column: value})
df_langs.head(5)
groups_langs = df_langs.groupby('Q6')
df_langs_pct = pd.DataFrame()
for name, group in groups_langs:
    df_langs_pct[name] = pd.Series(group.notnull().mean()*100)
df_langs_pct = df_langs_pct.unstack().to_frame().reset_index()
df_langs_pct = df_langs_pct.loc[df_langs_pct['level_1'] !='Q6']
df_langs_pct = df_langs_pct.rename(columns = {'level_0': 'Q3', 'level_1': 'langs', 0: 'count'})
df_langs_pct.head(10)
colors =sns.color_palette("Paired", 10)
colors2 = sns.color_palette("Set2", 10)
colors.extend(colors2)
df_langs_pct = df_langs_pct.sort_values(['langs', 'count'])
order = df_langs_pct['Q3'].unique()
fig, ax = plt.subplots(figsize=(20, 10))
ax = sns.stripplot(ax=ax, x='count', y="Q3", hue='langs', data=df_langs_pct,
                   palette=colors, jitter=False, size=10, alpha=.75, 
                   edgecolor="black", order=order)

ax.set_xlabel('Percentage of people')
ax.set_ylabel('')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.tick_params(labelsize=14)

plt.legend()
columns_q19 = [column for column in df_multiplechoice.columns if column.startswith('Q19_Part')]
columns_q19.append('Q6')
df_ml_frams = df_multiplechoice[columns_q19]
columns_q19.remove('Q6')
for column in columns_q19:
    df_temp = df_ml_frams[column]
    value = df_temp.loc[df_temp.first_valid_index()]
    df_ml_frams = df_ml_frams.rename(columns = {column: value})
df_ml_frams.head(5)
groups_ml_frams = df_ml_frams.groupby('Q6')
df_ml_pct = pd.DataFrame()
for name, group in groups_ml_frams:
    df_ml_pct[name] = pd.Series(group.notnull().mean()*100)
df_ml_pct = df_ml_pct.unstack().to_frame().reset_index()
df_ml_pct = df_ml_pct.loc[df_ml_pct['level_1'] !='Q6']
df_ml_pct = df_ml_pct.rename(columns = {'level_0': 'Q3', 'level_1': 'frameworks', 0: 'count'})
df_ml_pct.head(10)
colors =sns.color_palette("Paired", 10)
colors2 = sns.color_palette("Set2", 10)
colors.extend(colors2)
df_ml_pct = df_ml_pct.sort_values(['Q3', 'count'])
order = df_ml_pct['Q3'].unique()
fig, ax = plt.subplots(figsize=(20, 10))
ax = sns.stripplot(ax=ax, x='count', y="Q3", hue='frameworks', data=df_ml_pct,
                   palette=colors, jitter=False, size=10, alpha=.75, 
                   edgecolor="black", order=order)

ax.set_xlabel('Percentage of people')
ax.set_ylabel('')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.tick_params(labelsize=14)
df_title_coding = pd.crosstab(df_multiplechoice['Q23'], df_multiplechoice['Q6']).apply(lambda r: r/r.sum()*100, axis=0)
df_title_coding = df_title_coding.unstack().reset_index()
df_title_coding = df_title_coding.rename(columns = {0: 'pct'})
df_title_coding.head(10)
colors =sns.color_palette("Paired", 5)
colors2 = sns.color_palette("Set2", 10)
colors3 = sns.color_palette("Set1", 10)
colors.extend(colors2)
colors.extend(colors3)
fig, ax = plt.subplots(figsize=(15, 10))
order = ['0% of my time', '1% to 25% of my time', '25% to 49% of my time', '50% to 74% of my time', '75% to 99% of my time', '100% of my time']
df_title_coding.sort_values(['pct'], inplace=True)
ax = sns.stripplot(x='pct', y="Q23", hue='Q6', data=df_title_coding,
                   palette=colors, jitter=True, size=8, order=order)

ax.set_ylabel('')
ax.set_xlabel('Percentage of people')
ax.xaxis.grid(True)

plt.legend()
columns_q36 = [column for column in df_multiplechoice.columns if column.startswith('Q36_Part')]
columns_q36.append('Q6')
df_platforms = df_multiplechoice[columns_q36]
columns_q36.remove('Q6')
for column in columns_q36:
    df_temp = df_platforms[column]
    value = df_temp.loc[df_temp.first_valid_index()]
    df_platforms = df_platforms.rename(columns = {column: value})
df_platforms.head(5)
groups_platforms = df_platforms.groupby('Q6')
df_platforms_pct = pd.DataFrame()
for name, group in groups_platforms:
    df_platforms_pct[name] = pd.Series(group.notnull().mean()*100)
df_platforms_pct = df_platforms_pct.unstack().to_frame().reset_index()
df_platforms_pct = df_platforms_pct.loc[df_platforms_pct['level_1'] !='Q6']
df_platforms_pct = df_platforms_pct.rename(columns = {'level_0': 'role', 'level_1': 'platform', 0: 'pct'})
df_platforms_pct.head(10)
colors = sns.color_palette("Set2", 7)
colors2 = sns.color_palette("Paired", 8)
colors.extend(colors2)
df_platforms_pct = df_platforms_pct.sort_values(['platform', 'pct'], ascending=False)
order = df_platforms_pct['role'].unique()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.stripplot(ax=ax, x='pct', y="role", hue='platform', data=df_platforms_pct,
                   palette=colors, jitter=False, size=8, alpha=.75, 
                   edgecolor="black")
ax.set_xlabel('Percentage of people')
ax.set_ylabel('')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.tick_params(labelsize=14)

plt.legend()
fig, g = plt.subplots(figsize=(18, 8))
df_heatmap_lang = df_multiplechoice[['Q6', 'Q18']].pivot_table(index=['Q18'], columns='Q6', aggfunc=np.count_nonzero)
df_heatmap_lang['sum'] = df_heatmap_lang.sum(axis=1)
df_heatmap_lang.sort_values('sum', inplace=True, ascending=False)
df_heatmap_lang.drop(columns=['sum'], inplace=True)
g = sns.heatmap(df_heatmap_lang, annot=True, fmt=".0f", cmap=sns.cubehelix_palette(200, start=2, rot=0, dark=.05, light=.95, reverse=False))
g.tick_params(labelsize=14)
g.set_ylabel('')
g.set_xlabel('')
df_title_ach = pd.crosstab(df_multiplechoice['Q40'], df_multiplechoice['Q6']).apply(lambda r: r/r.sum()*100, axis=0)
df_title_ach = df_title_ach.unstack().reset_index()
df_title_ach = df_title_ach.rename(columns = {0: 'count'})
df_title_ach.head(10)
colors =sns.color_palette("Paired", 7)
fig, ax = plt.subplots(figsize=(16, 10))
#order = df_title_edu['Q6'].unique()
df_title_edu = df_title_ach.sort_values(['count', 'Q40'])
ax = sns.stripplot(x='count', y="Q6", hue='Q40', data=df_title_ach,
                   palette=colors, jitter=True, size=8)
ax.set_xlabel('Percentage of people on each profession')
ax.set_ylabel('')
ax.xaxis.grid(True)
ax.yaxis.grid(True)

plt.legend()