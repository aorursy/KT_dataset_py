import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import sqlite3
sns.set_style('whitegrid')
palette = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.70),
 (1.0, 0.4980392156862745, 0.054901960784313725,0.70),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 0.70),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392,0.70),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353,0.70),
 (0.8901960784313725, 0.4666666666666667, 0.7607843137254902,0.70),                                  
 (0.1, 0.1, 0.1,0.70)]
sns.set_palette(palette)
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15
sns.palplot(palette)
#Connecting to DB
conn = sqlite3.connect('../input/world-development-indicators/database.sqlite')
#Investigating the format of our features and observations
pd.read_sql(
    """
    PRAGMA table_info(indicators)
    """,
    con=conn)
#6 features, how many rows?
pd.read_sql(
    """
    SELECT MAX(rowid) FROM indicators
    """,
    con=conn)
#In the indicators table, there are 5.66m rows containing data from 1300+ indicators.
#I'm going to export the unique indicators to a CSV for manual review to select which I am going to investigate.
pd.read_sql(
    """
    SELECT DISTINCT IndicatorCode, IndicatorName
    FROM indicators
    """,
    con=conn).to_csv('Indicator List.csv')
atmosphere = pd.read_sql(
    """
    SELECT indicators.* FROM indicators
    WHERE 
        IndicatorCode IN (
            'EN.ATM.CO2E.PC',
            'EN.CO2.BLDG.ZS',
            'EN.CO2.ETOT.ZS',
            'EN.CO2.MANF.ZS',
            'EN.CO2.TRAN.ZS',
            'EN.ATM.METH.EG.ZS') AND 
        CountryCode IN ('NAC','EAS','EUU','LCN','MEA','SSF','WLD') AND
        Year > 1970
    ORDER BY Year,IndicatorCode,CountryCode
    """,
    con=conn
)
#Seperating our atmosphere dataframe in to individual dataframes for each indicator, as objects within a dictionary so we can iterate through them easily
atmosphere_dict = {}
for i in atmosphere.IndicatorCode.unique():
    atmosphere_dict[i] = atmosphere[atmosphere.IndicatorCode == i]
#Creating the subplots objects to graph on to
fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (27,12),gridspec_kw={'wspace':0.13,'hspace':0.4})

#Index variable to allow iteration through the dictionary
indy = -1
indx = -1
for i in atmosphere.IndicatorCode.unique():
    indy +=1
    indx +=1
    if indx >1:
        indx = 0
    #Plotting the data
    sns.lineplot(x='Year',y='Value',hue='CountryName',data=atmosphere_dict[i],ax=axes[int((1/2)*indy)][indx],palette=palette)
    
    #Pulling title from IndicatorName field
    axes[int((1/2)*indy)][indx].set_title(str(atmosphere_dict[i]['IndicatorName'].unique())[2:-2])
    
    #Removing legend to allow for a single common legend
    axes[int((1/2)*indy)][indx].get_legend().remove()
    
    #Setting y labels for graphs
    axes[int((1/2)*indy)][indx].set_ylabel('% of total')
    
    #Making the line for World value bolder to allow easy identification of global direction on indicator
    plt.setp(axes[int((1/2)*indy)][indx].lines[6],linewidth=2.5)
        
#Creating patches for labels
handles, labels = axes[0][1].get_legend_handles_labels()
#Skipping first value as we do not want the legend title
handles= handles[1:]
#Setting label values
labels = ['East Asia & Pacific', 'European Union', 'Latin America & Caribbean', 'North America','Middle East & North Africa', 'Sub-Saharan Africa', 'World']
#Setting legend position on graph
axes[0][1].legend(handles,labels,bbox_to_anchor=(1.01, 1.017))
#Setting unique y label values
axes[0][0].set_ylabel('Metric tons per capita')
#Setting x ticks to avoid half-year values
axes[2][1].set_xticks(range(1990,2015,5))
#Setting y ticks to prevent them from running in to the title
axes[0][1].set_yticks(range(6,22,2))
plt.show()
#Creating our dataframe
popspread = pd.read_sql(
    """
    SELECT indicators.* FROM indicators
    WHERE 
        IndicatorCode IN (
        'SP.POP.0014.TO.ZS',
        'SP.POP.1564.TO.ZS',
        'SP.POP.65UP.TO.ZS',
        'SP.RUR.TOTL.ZS',
        'SP.URB.TOTL.IN.ZS') AND 
        CountryCode IN ('WLD') 
        
    
    ORDER BY Year,IndicatorCode,CountryCode
    """,
    con=conn
)
#Creating our 3 x N array for the stackplot using list comprehension and np.vstack operation
yage = np.vstack([[popspread[popspread['IndicatorCode']==i]['Value']] for i in ['SP.POP.0014.TO.ZS','SP.POP.1564.TO.ZS','SP.POP.65UP.TO.ZS']])
#Creating our x values for year
xage = popspread[popspread['IndicatorCode']=='SP.POP.0014.TO.ZS']['Year']

#Creating our 2 x N array for the stackplot using list comprehension and np.vstack operation
yloc = np.vstack([[popspread[popspread['IndicatorCode']==i]['Value']] for i in ['SP.RUR.TOTL.ZS','SP.URB.TOTL.IN.ZS']])
#Creating our x values for year
xloc = popspread[popspread['IndicatorCode']=='SP.URB.TOTL.IN.ZS']['Year']
#Creating the subplots objects to graph on to
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (27,6),gridspec_kw={'wspace':0.13})

#Labels for our data
agelabels = ['Population, ages 0-14 (% of total)','Population, ages 15-64 (% of total)','Population ages 65 and above (% of total)']
#Creating the stackplot
axes[0].stackplot(xage,yage,labels=agelabels,colors=sns.cubehelix_palette(6, start=.5, rot=-.65))
#Setting graph title
axes[0].set_title('Population age spread among 3 age categories for the World, values % of total')
#Setting graph legend
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[::-1],labels[::-1],loc=4)

#Labels for our data
loclabels = ['Rural population (% of total)', 'Urban population (% of total)']
#Creating the stackplot
axes[1].stackplot(xloc,yloc,labels=loclabels,colors=sns.cubehelix_palette(6, start=.5, rot=-.65))
#Setting graph title
axes[1].set_title('Population split between rural and urban for the World, values % of total')
#Setting graph legend
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles[::-1],labels[::-1],loc=4)

#Setting axis titles
for ax in axes:
    ax.set_xlabel('Year')
    ax.set_ylabel('% Total')
wealthdisparity = pd.read_sql(
    """

    SELECT  A.CountryName as 'Country Name', A.[Year], A.Value AS 'Income held by top 20%',
            B.Value AS 'Income held by bottom 20%',
            C.Value AS 'GINI Index'
            
    FROM indicators A
        
    INNER JOIN indicators B ON
    A.[Year] = B.[Year] AND A.CountryName = B.CountryName AND A.IndicatorName != B.IndicatorName
    
    INNER JOIN indicators C ON
    
    A.[Year] = C.[Year] AND A.CountryName = C.CountryName

    WHERE A.IndicatorName = 'Income share held by highest 20%' AND B.IndicatorName = 'Income share held by lowest 20%' AND C.IndicatorName = 'GINI index (World Bank estimate)'


    """,
    con=conn
)
#What country names do we have access to?
wealthdisparity['Country Name'].unique()
pd.read_sql(
    """
    SELECT DISTINCT(CountryName)
    FROM indicators
    """,
    con=conn
).to_csv('CountryList.csv')
countries = pd.read_csv('../input/world-development-indicators-files/Countries.csv')
countries = countries[['Country List Grouped By Continent','Continent']]
countries.columns = ['Country','Continent']
#So now we have a list of each country and the respective continent it belongs to
countries.head(3)
#Pairing the countries with their continents
wealthdisparity['Continent'] = wealthdisparity['Country Name'].apply(lambda x: str(countries[countries['Country'] == x]['Continent'].values)[2:-2])
wealthdisparity.head(3)
#Are there any countries which have not been mapped?
wealthdisparity[wealthdisparity['Continent']=='']['Country Name'].unique()
#Dropping these
wealthdisparity = wealthdisparity[wealthdisparity['Continent']!='']
#Checking none remaining
wealthdisparity[wealthdisparity['Continent']=='']['Country Name'].unique()
plt.figure(figsize=(4,4))
sns.lineplot(x='Income held by top 20%',y='Income held by bottom 20%',hue='Country Name',legend=False,data=wealthdisparity)
#Calculation of these variables and what they are is covered in the method to follow
fig, axes = plt.subplots(figsize=(14,4))
wealthmean = wealthdisparity.groupby(['Year','Continent'],as_index=False).mean()
wealthmax = wealthdisparity.groupby(['Year','Continent'],as_index=False).max().drop('Country Name',axis=1)
wealthmin = wealthdisparity.groupby(['Year','Continent'],as_index=False).min().drop('Country Name',axis=1)
index_dict = {}
for cont in wealthmean['Continent'].unique():    
    index_dict[cont] = {}
    for c in wealthmean.columns[2:-1]:          
        index_dict[cont][c] = np.vstack([[i[i['Continent']==cont][c] for i in [wealthmin,wealthmean,wealthmax]]])
delta_dict = index_dict
for cont in delta_dict:
    for index in delta_dict[cont]:
        delta_dict[cont][index][1] = delta_dict[cont][index][1]-delta_dict[cont][index][0]
        delta_dict[cont][index][2] = delta_dict[cont][index][2] - delta_dict[cont][index][1] - delta_dict[cont][index][0]

axes.stackplot(wealthmean[wealthmean['Continent']=='Africa']['Year'],delta_dict['Africa']['Income held by bottom 20%'],colors=[[1,1,1,0],[0.54,0.54,0.80078431, 0.35],[0.54,0.54,0.80078431, 0.35]])
for table in [wealthmean,wealthmax,wealthmin]:
    sns.lineplot(table[table['Continent']=='Africa']['Year'],table[table['Continent']=='Africa']['Income held by bottom 20%'],ax=axes)

axes.set_title('Africa')
plt.setp(axes.lines[1:],ls='--')
plt.setp(axes.lines[0],c=(sns.light_palette("navy")[5]))
plt.show()
wealth_palette = []
#Selecting the colours we want from the generated palette
for i in range(0,5,2):
    wealth_palette.append(sns.cubehelix_palette(8, start=.5, rot=-.65, reverse=True)[i])
#Setting alpha value
for i in wealth_palette:
    i.append(0.65)
    
sns.palplot(wealth_palette)
#Same colour values but slightly increased alpha to make graph lines more distinct
wealth_palette_lines = wealth_palette
for i in wealth_palette_lines:
    i[-1] = 0.8
    
sns.palplot(wealth_palette_lines)
#Creating mean table
wealthmean = wealthdisparity.groupby(['Year','Continent'],as_index=False).mean()
wealthmean['Type'] = 'Mean'
wealthmean.head(3)
#Creating max table
wealthmax = wealthdisparity.groupby(['Year','Continent'],as_index=False).max().drop('Country Name',axis=1)
wealthmax['Type'] = 'Max'
wealthmax.head(3)
#Creating min table
wealthmin = wealthdisparity.groupby(['Year','Continent'],as_index=False).min().drop('Country Name',axis=1)
wealthmin['Type'] = 'Min'
wealthmin.head(3)
index_dict = {}

#Iterating through continents to create dictionary value for each continent key
for cont in wealthmean['Continent'].unique():    
    index_dict[cont] = {}
    
    #Iterating through indexes for each continent
    for c in wealthmean.columns[2:-1]:          
        #This creates a dictionary containing keys for each index, which have keys for each continent, whose values are a 3 x M array of the min, mean and max values
        index_dict[cont][c] = np.vstack([[i[i['Continent']==cont][c] for i in [wealthmin,wealthmean,wealthmax]]])
        
#This dictionary should allow us to iterate through the values and make graphing much simpler  
#Pulling an example key
index_dict['Africa']['Income held by top 20%']
#Creating a copy of the dictionary made earlier
delta_dict = index_dict

#Iterating through the continents
for cont in delta_dict:
    
    #Iterating through the indicators for each continent
    for index in delta_dict[cont]:
        #Mean = Mean - Min i.e. delta value
        delta_dict[cont][index][1] = delta_dict[cont][index][1]-delta_dict[cont][index][0]
        #Mean = Mean - (Mean-Min) - Min i.e. delta value
        delta_dict[cont][index][2] = delta_dict[cont][index][2] - delta_dict[cont][index][1] - delta_dict[cont][index][0]
#Pulling an example key
delta_dict['Africa']['Income held by top 20%']
#Values seem to be correct!
fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(27,15))
#Variable for iterative indexing
ind = -1

for cont in list(delta_dict.keys()):
    ind +=1
    #Variable to select common colours for each graph
    colour =-1
    for index in delta_dict[cont]:      
        colour += 1
        #Stack plot - Year values for each continent as the x, delta indicator values for the y
        axes[int((1/3)*ind)][ind-3].stackplot(wealthmean[wealthmean['Continent']==cont]['Year'],delta_dict[cont][index],colors=[[1,1,1,0],list(wealth_palette[colour]),list(wealth_palette[colour])])
        #Setting graph title for each graph
        axes[int((1/3)*ind)][ind-3].set_title(cont)
        #Plotting linegraphs for mean, max and min on top of the stack plots to improve visuals
        for table in [wealthmean,wealthmax,wealthmin]:
            sns.lineplot(table[table['Continent']==cont]['Year'],table[table['Continent']==cont][index],label=index,ax=axes[int((1/3)*ind)][ind-3])
        #Setting y label for each graph        
        axes[int((1/3)*ind)][ind-3].set_ylabel('% Value')     
            
    #Matching colour lines in linegraphs to those in stackplot and boldening the lines
    for i in range(0,7,3):
        plt.setp(axes[int((1/3)*ind)][ind-3].lines[i],c=wealth_palette_lines[int(i/3)])
        plt.setp(axes[int((1/3)*ind)][ind-3].lines[i],lw=2.3)
     
    #Setting linestyle for max and min values to dashed lines
    for i in [1,2,4,5,7,8]:
        plt.setp(axes[int((1/3)*ind)][ind-3].lines[i],ls='--')
        
    #Removing legend
    axes[int((1/3)*ind)][ind-3].get_legend().remove()
    
    #Setting common y&x axes to allow easier comparison between each continent
    axes[int((1/3)*ind)][ind-3].set_yticks(range(0,90,10))
    axes[int((1/3)*ind)][ind-3].set_xticks(range(1980,2020,5))
        
#Creating common legend
handles, labels = axes[0][1].get_legend_handles_labels()
#Creating custom legend labels
labels = ['Max income held by top 20%','Mean income held by top 20%','Min income held by top 20%','Max GINI Index','Mean GINI Index','Min GINI Index','Max income held by bottom 20%','Mean income held by bottom 20%','Min income held by bottom 20%']
#Reordering legend handles
handles = [handles[i] for i in [1,0,2,7,6,8,4,3,5]]
axes[0][2].legend(handles,labels,bbox_to_anchor=(1.01, 1.017),fontsize=13)
fig.suptitle('Wealth distribution for all 6 continents, showing income held by the top and bottom 20%, and the GINI index',y=0.92, fontsize=15)

plt.show()
sns.palplot(sns.cubehelix_palette(6, start=.5, rot=-.65)[:3][::-1])
lit_rate = pd.read_sql(
    """
    SELECT indicators.* FROM indicators
    WHERE 
        IndicatorCode IN (
        'SE.ADT.1524.LT.ZS',
        'SE.ADT.LITR.ZS') AND 
        CountryCode IN ('NAC','EAS','EUU','LCN','MEA','SSF','WLD') 
        
    
    ORDER BY Year,IndicatorCode,CountryCode
    """,
    con=conn
)
lit_rate.head(3)
#Do we have any missing values?
lit_rate.groupby(['CountryName','IndicatorName']).count()['Year']
#Europe is missing one set of values, which are these?
lit_rate[lit_rate['CountryCode']=='EUU']['Year'].unique()
#Inserting 0 values for Europe so that the graphing is correct.
lit_rate = lit_rate.append(lit_rate[lit_rate['CountryCode']=='EUU'].iloc[:2,:-2].join(pd.DataFrame(data=[[1990,0],[1990,0]],columns=['Year','Value'],index=[11,17])))
lit_rate.tail(3)
#Re-sorting values
lit_rate = lit_rate.sort_values(['IndicatorName','Year','CountryName'])
#Neatening up country names
lit_rate['CountryName'] = lit_rate['CountryName'].apply(lambda x: x.replace(' (all income levels)',''))
fig, axes = plt.subplots(ncols=2, figsize = (27,12),gridspec_kw={'wspace':0.0},sharey=True)

#Iterating through the indicators
for index in lit_rate['IndicatorName'].unique():
    
    #Variable index value equaling 0 or 1 depending on the index
    ind = int(len('Adult literacy rate, population 15+ years, both sexes (%)')/len(index))
    #Variable to select colour
    colour=-1
    #Creating sub-table to select individual indicators values
    table = lit_rate[lit_rate['IndicatorName']==index]
    #Iterating through the years to create seperate bars for each
    for year in lit_rate['Year'].unique()[::-1]:
        colour+=1
        ytable = table[table['Year']==year]
        sns.barplot(x='Value',y='CountryName',data=ytable,palette = [[i]*6 for i in sns.cubehelix_palette(6, start=.5, rot=-.65)[:3][::-1]][colour],ax=axes[ind], label=year)
    #Setting axis labels
    axes[ind].set_ylabel('')
    axes[ind].set_xlabel(index, fontsize=13)
    #For some strange reason, Seaborn doesn't pull the alpha value for the palette when making the bar patches. Manually amending the alpha value.
    for p in axes[ind].patches:
        plt.setp(p,alpha=.80)

#Setting axes limits
axes[0].set_xlim(100,50)
axes[1].set_xlim(50,100)
#Setting central axis to be thicker
axes[1].spines['left'].set_linewidth(3)
#Setting legend and diagram title
axes[1].legend(loc="lower right",title='Year', fontsize=13, frameon=True)
fig.suptitle('% Literacy rate for 15-24 year olds and all those 15+, from 1990-2010',y=0.9, fontsize=13)

plt.show()
health = pd.read_sql(
    """
    SELECT CountryName,CountryCode,'Average physicians, nurses and hospital beds (per 1,000 people)'  as IndicatorName,'SH.MED.PNHB.ZS' as IndicatorCode,Year,AVG(Value) as Value FROM indicators
    WHERE 
        IndicatorCode IN (
        'SH.MED.BEDS.ZS',
        'SH.MED.NUMW.P3',
        'SH.MED.PHYS.ZS') AND 
        CountryCode IN ('NAC','EAS','EUU','LCN','MEA','SSF','WLD') AND
        Year > 1970
    GROUP BY CountryName,Year
    UNION ALL
    SELECT CountryName,CountryCode,'Smoking prevalence (% of adults)'  as IndicatorName,'SH.PRV.SMOK.AD' as IndicatorCode,Year,AVG(Value) as Value FROM indicators
    WHERE 
        IndicatorCode IN (
        'SH.PRV.SMOK.FE',
        'SH.PRV.SMOK.MA') AND 
        CountryCode IN ('NAC','EAS','EUU','LCN','MEA','SSF','WLD') AND
        Year > 1970
    GROUP BY CountryName,Year    
    UNION ALL
    SELECT indicators.* FROM indicators
    WHERE 
        IndicatorCode IN (
        'SH.STA.MALN.ZS',
        'SH.STA.OWGH.ZS',
        'SH.XPD.TOTL.ZS') AND 
        CountryCode IN ('NAC','EAS','EUU','LCN','MEA','SSF','WLD') AND
        Year > 1970
    ORDER BY Year,IndicatorCode,CountryCode
    """,
    con=conn
)
#Seperating our health dataframe in to individual dataframes for each indicator, as objects within a dictionary so we can iterate through them easily
health_dict = {}
for i in health.IndicatorCode.unique():
    health_dict[i] = health[health.IndicatorCode == i]
#Creating the subplots objects to graph on to
fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (27,12),gridspec_kw={'wspace':0.13})
#Index variable to allow iteration through the dictionary
ind = -1
#As the final graph only contains world values, using custom list of IndicatorCodes to order the graphs correctly
for i in ['SH.XPD.TOTL.ZS','SH.MED.PNHB.ZS','SH.PRV.SMOK.AD','SH.STA.OWGH.ZS','SH.STA.MALN.ZS']:
    ind +=1
    if ind < 2:
        #First row of graphs
        sns.lineplot(x='Year',y='Value',hue='CountryName',data=health_dict[i],ax=axes[0][ind],palette=palette)
        #Pulling title from IndicatorName field
        axes[0][ind].set_title(str(health_dict[i]['IndicatorName'].unique())[2:-2])
        #Removing legend to allow for a single common legend
        axes[0][ind].get_legend().remove()
        #Making the line for World value bolder to allow easy identification of global direction on indicator
        plt.setp(axes[0][ind].lines[6],linewidth=2.5)
        #Setting the line alpha as there's no way to set the alpha level on the colour palette
        for l in axes[0][ind].lines:
            plt.setp(l,alpha=.70)
        
    elif ind == 2:
        #Bottom left graph
        sns.lineplot(x='Year',y='Value',hue='CountryName',data=health_dict[i],ax=axes[1][ind-2])
        #Pulling title from IndicatorName field
        axes[1][ind-2].set_title(str(health_dict[i]['IndicatorName'].unique())[2:-2])
        #Removing legend to allow for a single common legend
        axes[1][ind-2].get_legend().remove()
        #Making the line for World value bolder to allow easy identification of global direction on indicator
        plt.setp(axes[1][ind-2].lines[6],linewidth=2.5)
        #Setting the line alpha as there's no way to set the alpha level on the colour palette
        for l in axes[1][ind-2].lines:
            plt.setp(l,alpha=.70)

    elif ind < 5:
        #Bottom right graph, plotting the underweight and overweight values on the same graph
        sns.lineplot(x='Year',y='Value',data=health_dict[i],label=['Overweight, weight for height','Underweight, weight for height'][ind-3],color=sns.color_palette("Set2", 2)[ind-3],ax=axes[1][1])
        #Setting bottom right graph title
        axes[1][1].set_title('Prevelance of abnormal weight, weight for height (% of children under 5)')
        #Creating the legend for bottom right axis, reversing order to match order of lines
        handles, labels = axes[1][1].get_legend_handles_labels()
        axes[1][1].legend(handles[::-1],labels[::-1])
        
    else:
        break
        
#Creating patches for labels
handles, labels = axes[0][1].get_legend_handles_labels()
#Skipping first value as we do not want the legend title
handles= handles[1:]
#Setting legend label values
labels = ['East Asia & Pacific', 'European Union', 'Latin America & Caribbean', 'North America','Middle East & North Africa', 'Sub-Saharan Africa', 'World']
#Adding the legend to top right graph
axes[0][1].legend(handles,labels, bbox_to_anchor=(0.76, 0.69))
#Setting unique y label values
axes[0][0].set_ylabel('% of GDP')
axes[0][1].set_ylabel('per 1,000 people')
axes[1][0].set_ylabel('% of adults')
axes[1][1].set_ylabel('% of children under 5')
#Setting x ticks for top left graph to replace decimal year values automatically generated
axes[0][0].set_xticks(range(1995,2015,4))

plt.show()
lifespan = pd.read_sql(
    """
    SELECT indicators.* FROM indicators
    WHERE 
        IndicatorCode IN (
        'SP.DYN.AMRT.FE',
        'SP.DYN.AMRT.MA',
        'SP.DYN.TO65.FE.ZS',
        'SP.DYN.TO65.MA.ZS') AND 
        CountryCode IN ('NAC','EAS','EUU','LCN','MEA','SSF','WLD') AND
        Year > 1970
    ORDER BY Year,IndicatorCode,CountryCode
    """,
    con=conn
)
lifespan_dict = {}
for i in lifespan.IndicatorCode.unique():
    lifespan_dict[i] = lifespan[lifespan.IndicatorCode == i]

#Creating the subplots objects to graph on to
fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (27,12),gridspec_kw={'wspace':0.13})
#Index variable to allow iteration through the dictionary
ind = -1
for i in lifespan.IndicatorCode.unique():
    ind +=1
    if ind < 2:
        #First row of graphs
        axes[0][ind].scatter(x='Year',y='CountryName',data=lifespan_dict[i],s='Value')


    else:
        break
        

plt.show()
#Creating our subset of the data, selecting interesting indicators and these regions of the globe: 
#'East Asia & Pacific', 'European Union', 'Latin America & Caribbean', 'North America','Middle East & North Africa', 'Sub-Saharan Africa', 'World'
electricity = pd.read_sql(
    """
    SELECT indicators.* FROM indicators
    WHERE 
        IndicatorCode IN ('EG.ELC.FOSL.ZS',
            'EG.ELC.HYRO.ZS',
            'EG.ELC.NUCL.ZS',
            'EG.ELC.RNWX.ZS',
            'EG.FEC.RNEW.ZS',
            'EG.USE.COMM.FO.ZS',
            'EG.USE.ELEC.KH.PC') AND 
        CountryCode IN ('NAC','EAS','EUU','LCN','MEA','SSF','WLD') AND
        Year > 1970
    ORDER BY Year,IndicatorCode,CountryCode
    """,
    con=conn
)
#Seperating our electricity dataframe in to individual dataframes for each indicator, as objects within a dictionary so we can iterate through them easily
electricity_dict = {}
for i in electricity.IndicatorCode.unique():
    electricity_dict[i] = electricity[electricity.IndicatorCode == i]
#Creating the subplots objects to graph on to
fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (27,12),gridspec_kw={'wspace':0.13})
#Index variable to allow iteration through the dictionary
ind = -1
for i in electricity.IndicatorCode.unique():
    ind +=1
    if ind < 3:
        #First row of graphs
        sns.lineplot(x='Year',y='Value',hue='CountryName',data=electricity_dict[i],ax=axes[0][ind])
        #Pulling title from IndicatorName field
        axes[0][ind].set_title(str(electricity_dict[i]['IndicatorName'].unique())[2:-2])
        #Removing legend to allow for a single common legend
        axes[0][ind].get_legend().remove()
        #Setting y labels for graphs
        axes[0][ind].set_ylabel('% of total')
        #Making the line for World value bolder to allow easy identification of global direction on indicator
        plt.setp(axes[0][ind].lines[6],linewidth=2.5)
        #Setting the line alpha as there's no way to set the alpha level on the colour palette
        for l in axes[0][ind].lines:
            plt.setp(l,alpha=.70)
        
    elif ind < 6:
        #Second row of graphs
        sns.lineplot(x='Year',y='Value',hue='CountryName',data=electricity_dict[i],ax=axes[1][ind-3])
        #Pulling title from IndicatorName field
        axes[1][ind-3].set_title(str(electricity_dict[i]['IndicatorName'].unique())[2:-2])
        #Removing legend to allow for a single common legend
        axes[1][ind-3].get_legend().remove()
        #Setting y labels for graphs    
        axes[1][ind-3].set_ylabel('% of total')
        #Making the line for World value bolder to allow easy identification of global direction on indicator
        plt.setp(axes[1][ind-3].lines[6],linewidth=2.5)
        #Setting the line alpha as there's no way to set the alpha level on the colour palette
        for l in axes[1][ind-3].lines:
            plt.setp(l,alpha=.70)
    else:
        break
        
#Creating patches for labels
handles, labels = axes[0][2].get_legend_handles_labels()
#Skipping first value as we do not want the legend title
handles= handles[1:]
#Setting label values
labels = ['East Asia & Pacific', 'European Union', 'Latin America & Caribbean', 'North America','Middle East & North Africa', 'Sub-Saharan Africa', 'World']
#Setting legend position on graph
axes[0][2].legend(handles,labels,bbox_to_anchor=(1.01, 1.017))
#Setting unique y label value
axes[1][2].set_ylabel('kWh per capita')