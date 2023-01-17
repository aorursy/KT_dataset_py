import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
%matplotlib inline
data = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
regions = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
summer = data[data.Season == "Summer"]
summer.sample(5)
female_dcast = pd.read_csv('../input/female-decastcsv/female_decast.csv')
female_dcast.head()
female_dcast = female_dcast.drop(female_dcast.columns[[0]], axis=1)
female_dcast.head()
female_dcast_tidy_df=female_dcast.melt(id_vars=['year','all','f_prct',],var_name='sex',value_name='frequencies')
female_dcast_tidy_df.head()
female_dcast_tidy_df['female_percent']=female_dcast_tidy_df.f_prct.mul(100).round(1)
female_dcast_tidy_df.head()
female_dcast_tidy_df['male_percentage']=100- female_dcast_tidy_df['female_percent']
female_dcast_tidy_df['percentage']=female_dcast_tidy_df['frequencies']/female_dcast_tidy_df['all']
female_dcast_tidy_df.tail()
source = female_dcast_tidy_df

alt.Chart(source, title = 'Changes in female athletes participation in the Olympics (1896-2016)').mark_line(size=4
).encode(
    x=alt.X('year',title='Year'),
    y= alt.Y('percentage', axis=alt.Axis(format='%', title='Percentage')),
    color='sex',
    tooltip=['year','percentage:Q']
).interactive().properties(
    width=600,
    height=300).configure_title(
    fontSize=16).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    labelFontSize=14
)

alt.Chart(female_dcast_tidy_df, title = 'Changes in female athletes participation in the Olympics (1896-2016)').mark_bar(size=4).encode(
    x=alt.X('sum(frequencies)', stack="normalize", axis=alt.Axis(format='%', title='Percentage',grid=False)),
    y=alt.Y('year',title='Year'),
    color=alt.Color('sex'),
    tooltip=['year','female_percent:Q'],
    order=alt.Order(
    'sex',
    sort='ascending')).interactive().properties(
    width=600,
    height=300).configure_title(
    fontSize=16).configure_axis(
    labelFontSize=14,
    titleFontSize=16
).configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    labelFontSize=14
)
base=alt.Chart(female_dcast_tidy_df).encode(x='year')

domain=["F","M"]
scl=["blue","grey"]
bar=alt.Chart(female_dcast_tidy_df, title='Changes in female athletes participation in the Olympics (1896-2016)').mark_bar(size=10).encode(
    x=alt.X('year',title='Year'),
    y=alt.Y('sum(frequencies):Q', stack="normalize", axis=alt.Axis(format='%', title='Percentage', grid=False)),
    color=alt.Color('sex', scale=alt.Scale(domain=domain, range=scl)),order=alt.Order(
    'sex',
    sort='ascending'),
    tooltip=['year','female_percent:Q'],
)


line= alt.Chart(female_dcast_tidy_df).mark_line(point=True,color='red').encode(
    x=alt.X('year',title='Year'),
    y='f_prct'
)

alt.layer(bar+line).properties(
    width=600,
    height=300).configure_title(
    fontSize=20).configure_axis(
    labelFontSize=14,
    titleFontSize=16
).configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    labelFontSize=14
)
alt.Chart(female_dcast_tidy_df,title = 'Changes in female athletes participation in the Olympics (1896-2016)').mark_area().encode(
    x=alt.X("year:N",title='Year'),
    y=alt.Y("sum(frequencies):Q", title='Number of Athletes'),
    color="sex:N",
    tooltip=['year','female_percent:Q']
).interactive().properties(
    width=600,
    height=300).configure_title(
    fontSize=16).configure_axis(
    labelFontSize=14,
    titleFontSize=16
).configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    labelFontSize=14
)
continent=pd.read_csv('../input/continent/continent.csv')
continent_sp=continent[['Continent_Name','Three_Letter_Country_Code']]
continent_sp.rename(columns={'Three_Letter_Country_Code':'code_3'},inplace=True)
continent_4=pd.read_excel('../input/continent-4/continent_4.xlsx')
continent_4.rename(columns={'A-3':'code_3'},inplace=True)
continent_4_sp=continent_4[['249 countries','code_3','IOC']]
continent_merged=continent_4_sp.merge(continent_sp, on='code_3', how='left')
continent_merged.rename(columns={'IOC':'NOC'},inplace=True)
continent_merged.head()
summer_con=summer.merge(continent_merged,on='NOC',how='left')
summer_con_simple = summer_con[['Sex', 'Team','Year','Continent_Name']]
summer_con_classified2=pd.crosstab(summer_con_simple.Year, [summer_con_simple.Sex,summer_con_simple.Continent_Name])
summer_con_classified2['M', 'total'] = summer_con_classified2.iloc[:,0:12].sum(axis=1)
summer_con_classified2['M', 'female_percent']=summer_con_classified2.iloc[:,0:6].sum(axis=1)/summer_con_classified2['M', 'total'] 
summer_con_classified2['M','Africa_percent']=summer_con_classified2.iloc[:,0]/(summer_con_classified2.iloc[:,0]+
                                                                              summer_con_classified2.iloc[:,6])
summer_con_classified2['M','Asia_percent']=summer_con_classified2.iloc[:,1]/(summer_con_classified2.iloc[:,1]+
                                                                              summer_con_classified2.iloc[:,7])
summer_con_classified2['M','Europe_percent']=summer_con_classified2.iloc[:,2]/(summer_con_classified2.iloc[:,2]+
                                                                              summer_con_classified2.iloc[:,8])
summer_con_classified2['M','North America_percent']=summer_con_classified2.iloc[:,3]/(summer_con_classified2.iloc[:,3]+
                                                                              summer_con_classified2.iloc[:,9])
summer_con_classified2['M','Oceania_percent']=summer_con_classified2.iloc[:,4]/(summer_con_classified2.iloc[:,4]+
                                                                              summer_con_classified2.iloc[:,10])
summer_con_classified2['M','South America_percent']=summer_con_classified2.iloc[:,5]/(summer_con_classified2.iloc[:,5]+
                                                                              summer_con_classified2.iloc[:,11])
df=summer_con_classified2.drop(summer_con_classified2.iloc[:,0:12],axis=1,inplace=False)
df_new = df.xs('M', axis=1, drop_level=True)
df2=df_new.drop(df_new.iloc[:,0:1],axis=1,inplace=False)
df3=df2.reset_index(inplace=False)
df3.columns = ['Year', 'Global', 'Africa', 'Asia', 'Europe','N_America','Oceania','S_America']
df3_tidy=pd.melt(df3,id_vars=['Year'],var_name='Continent',value_name='Percentage')
df3_tidy.sample(5)
alt.Chart(df3_tidy, title='How did the percentage of female athelets change for each continent?').mark_line().encode(
    x='Year',
    y=alt.Y('Percentage',title='Female Percentage'),
    color='Continent'
).properties(
    width=600,
    height=300).configure_title(
    fontSize=16).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    labelFontSize=14
)
plt.figure(figsize=(30,30))

# Create a grid : initialize it
g = sns.FacetGrid(df3_tidy, col='Continent', hue='Continent', col_wrap=3, )
 
# Add the line over the area with the plot function
g = g.map(plt.plot, 'Year', 'Percentage').set_titles("{col_name} Continent")

# Control the title of each facet
g = g.set_titles("{col_name}")

plt.subplots_adjust(top=0.92)
g = g.fig.suptitle('How did the percentage of female athelets change for each continent?', fontsize=20)
alt.Chart(df3_tidy).mark_line().encode(
    x='Year:N',
    y='Percentage:Q',
    color='Continent:N'
).properties(
    width=200,
    height=200
).facet(
    facet='Continent:N',
    columns=3
)
# Create a grid : initialize it
g = sns.FacetGrid(df3_tidy, col='Continent', hue='Continent', col_wrap=3, )
 
# Add the line over the area with the plot function
g = g.map(plt.plot, 'Year', 'Percentage')

# Fill the area with fill_between
g = g.map(plt.fill_between, 'Year', 'Percentage', alpha=0.2).set_titles("{col_name} Continent")
 
# Control the title of each facet
g = g.set_titles("{col_name}")
 
# Add a title for the whole plo
plt.subplots_adjust(top=0.92)
g = g.fig.suptitle('How did the percentage of female athelets change for each continent?', fontsize=20)
source = df3_tidy

alt.Chart(source, title='How did the percentage of female athelets change for each continent?').transform_filter(
    alt.datum.symbol != 'GOOG'
).mark_area().encode(
    x='Year:N',
    y='Percentage:Q',
    color='Continent:N',
    row=alt.Row('Continent:N')
).properties(height=100, width=350).configure_title(
    fontSize=16).configure_axis(
    labelFontSize=14,
    titleFontSize=16
).configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    labelFontSize=14
)
df4=df3
df4.columns = ['Year', 'All','Africa','Asia','Europe','N_America','Oceania','S_America']
df4.head()
plt.figure(figsize=(15,7))

# Initialize the figure
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('tab10')
 
# multiple line plot
num=0
for column in df4.drop(['Year', 'All'], axis=1):
    num+=1
 
    # Find the right spot on the plot
    plt.subplot(2,3, num)
 
    # plot every groups, but discreet
    for v in df4.drop(['Year', 'All'], axis=1):
        plt.plot(df4['Year'], df4[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
 
    # Plot the lineplot
    plt.plot(df4['Year'], df3[column], marker='', color=palette(num), linewidth=2.4, alpha=0.9, label=column)
 

    # Add title
    plt.title(column, loc='left', fontsize=20, fontweight=0, color=palette(num) )
 
# general title
plt.suptitle("How did the percentage of female athelets change for each continent?", fontsize=30, fontweight=3, color='black', style='normal')
plt.figure(figsize=(20,10))

# Initialize the figure
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('tab10')
 
# multiple line plot
num=0
for column in df4.drop(['Year', 'All'], axis=1):
    num+=1
 
    # Find the right spot on the plot
    plt.subplot(2,3, num)
 
    # plot every groups, but discreet
    for v in df4.drop(['Year', 'All'], axis=1):
        plt.plot(df4['Year'], df4[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
 
    # Plot the lineplot
    plt.plot(df4['Year'], df3[column], marker='', color=palette(num), linewidth=2.4, alpha=1, label=column)
    plt.plot(df4['Year'], df4['All'], marker='', color='k', linewidth=2.4,alpha=0.8)

    # Add title
    plt.title(column, loc='left', fontsize=20, fontweight=0, color=palette(num) )
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Female Percentage', fontsize=16,rotation=90)
 
# general title
plt.suptitle("How did the percentage of female athelets change for each continent? (Black line showing global total)", fontsize=30, fontweight=3, color='black', style='normal')
