

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Import_dataset = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')

Export_dataset = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
#Visualising top 5 rows of our DataFrame

Import_dataset.head(5)
#Info of dataset

Import_dataset.info()
#Checking for stats of this Import Data

Import_dataset.describe()
#Checking For Duplicate rows

Import_dataset[Import_dataset.duplicated()].shape
#Droping Duplicate values

Import_dataset.drop_duplicates(keep='first', inplace=True)

Import_dataset.shape
#Checking for null values

Import_dataset.isnull().sum()
Import_dataset.groupby('country').agg( {'value': 'median'}).head(20)
t = Import_dataset.groupby('country').agg( {'value': 'median'})

t[t['value'] > 4.9]
Country_lst = []

Total_count_lst = []

Null_count_lst = []

Median_lst = []

for country in t[t['value'] > 4.9].index:

    Null_count = Import_dataset[Import_dataset['country'] == country].value.isnull().sum()

    Total_count = Import_dataset[Import_dataset['country']  == country].shape[0]

    Median = Import_dataset[Import_dataset['country']  == country].value.median()

    Country_lst.append(country)

    Total_count_lst.append(Total_count)

    Null_count_lst.append(Null_count)

    Median_lst.append(Median)

    

df = pd.DataFrame({'Country' : Country_lst, 'Total_count':Total_count_lst, 'Null_count': Null_count_lst, 

                   'Median': Median_lst})

df
#Since the null value corressponds to different countries, so will replace null values of a country with median value of import of that country

#Importing all country name in a array country



#country = Import_dataset['country'].unique()

country = df['Country'].unique()



#defing a temporary dataframe similar to original dataframe

imp_d_1 = pd.DataFrame(columns=Import_dataset.columns)

tmp = pd.DataFrame(columns=Import_dataset.columns)



#for loop to fill na values country wise with median of that country

for i in country:

    for cmdty in Import_dataset[Import_dataset['country'] == i]['Commodity'].unique():

        median = Import_dataset[(Import_dataset['country'] == i) & (Import_dataset['Commodity'] == cmdty)].value.median()

        tmp1 = Import_dataset[(Import_dataset['country'] == i) & (Import_dataset['Commodity'] == cmdty)].fillna(median)

        tmp = pd.concat([tmp, tmp1], axis=0)

    

imp_d_1 = tmp

imp_d_1.head()
#Filling null values for rest countries

filled_countries = imp_d_1['country'].unique()

all_countries = Import_dataset['country'].unique()

rest_countries=list(set(filled_countries)^set(all_countries))





#defing a temporary dataframe similar to original dataframe

tmp = pd.DataFrame(columns=Import_dataset.columns)



#for loop to fill na values country wise with median of that country

for i in rest_countries:

    median = Import_dataset[Import_dataset['country'] == i].value.median()

    tmp1 = Import_dataset[Import_dataset['country'] == i].fillna(median)

    tmp = pd.concat([tmp, tmp1], axis=0)

    

imp_d_2 = tmp

imp_d_2.head()
#Combining the two DataFrame for a complete Import DataFrame



Import = pd.concat([imp_d_1, imp_d_2])

Import.sort_values(["country", "Commodity"], ascending=[True, True], inplace=True)
#Checking for null values

Import.isnull().sum()
Import[Import.value.isnull()]
Import.dropna(inplace=True)
#Checking for null values

Import.isnull().sum()
Import.drop(Import[Import['country'] == 'UNSPECIFIED'].index, inplace=True)
Import_data = Import
#Visualising top 5 rows of our DataFrame

Export_dataset.head(5)
#Info of dataset

Export_dataset.info()
#Checking for stats of this Import Data

Export_dataset.describe()
#Checking For Duplicate rows

Export_dataset[Export_dataset.duplicated()].shape
#Checking for null values

Export_dataset.isnull().sum()
Export_dataset.groupby('country').agg( {'value': 'median'}).head(20)
t = Export_dataset.groupby('country').agg( {'value': 'median'})

t[t['value'] > 10]
Country_lst = []

Total_count_lst = []

Null_count_lst = []

Median_lst = []

for country in t[t['value'] > 10].index:

    Null_count = Export_dataset[Export_dataset['country'] == country].value.isnull().sum()

    Total_count = Export_dataset[Export_dataset['country']  == country].shape[0]

    Median = Export_dataset[Export_dataset['country']  == country].value.median()

    Country_lst.append(country)

    Total_count_lst.append(Total_count)

    Null_count_lst.append(Null_count)

    Median_lst.append(Median)

    

df = pd.DataFrame({'Country' : Country_lst, 'Total_count':Total_count_lst, 'Null_count': Null_count_lst, 

                   'Median': Median_lst})

df
df['Country'].unique()
#Since the null value corressponds to different countries, so will replace null values of a country with median value of exmport of that country

#Importing all country name in a array country







#defing a temporary dataframe similar to original dataframe

exp_d = pd.DataFrame(columns=Export_dataset.columns)

tmp = pd.DataFrame(columns=Export_dataset.columns)



#for loop to fill na values country wise with median of that country

for i in df['Country'].unique():

    for cmdty in Export_dataset[Export_dataset['country'] == i]['Commodity'].unique():

        median = Export_dataset[(Export_dataset['country'] == i) & (Export_dataset['Commodity'] == cmdty)].value.median()

        tmp1 = Export_dataset[(Export_dataset['country'] == i) & (Export_dataset['Commodity'] == cmdty)].fillna(median)

        tmp = pd.concat([tmp, tmp1], axis=0)

    

exp_d_1 = tmp

exp_d_1.head()
#Filling null values for rest countries

filled_countries = exp_d_1['country'].unique()

all_countries = Export_dataset['country'].unique()

rest_countries=list(set(filled_countries)^set(all_countries))





#defing a temporary dataframe similar to original dataframe

tmp = pd.DataFrame(columns=Export_dataset.columns)



#for loop to fill na values country wise with median of that country

for i in rest_countries:

    median = Export_dataset[Export_dataset['country'] == i].value.median()

    tmp1 = Export_dataset[Export_dataset['country'] == i].fillna(median)

    tmp = pd.concat([tmp, tmp1], axis=0)

    

exp_d_2 = tmp

exp_d_2.head()
#Combining the two DataFrame for a complete Export DataFrame



Export = pd.concat([exp_d_1, exp_d_2])

Export.sort_values(["country", "Commodity", "year"], ascending=[True, True, True], inplace=True)
Export.drop(Export[Export['country'] == 'UNSPECIFIED'].index, inplace=True)
Export_data = Export

Export.isnull().sum()
#Generating Import Data

imp_d = Import

country = imp_d['country'].unique()

year = imp_d['year'].unique()

tmp_cntry = pd.DataFrame(columns=imp_d.columns)

tmp_yer = pd.DataFrame(columns=imp_d.columns)

country_list = []

year_list = []

value_list = []

#Loop for segrigating country wise data

for cntry in country:

    tmp_cntry = imp_d[imp_d['country'] == cntry]

    #Loop for segrigating Contry wise data further into year wise

    for yer in year:

        tmp_yer =tmp_cntry[tmp_cntry['year'] == yer]

        tmp_total_year_value = tmp_yer['value'].sum()

        tmp_country = cntry

        tmp_year = yer

        year_list.append(tmp_year)

        country_list.append(tmp_country)

        value_list.append(tmp_total_year_value)

        

Consolidated_data = {'Country' : country_list, 'Year' : year_list ,'ImportValue' : value_list}

imp_yearly_Consolidated = pd.DataFrame(Consolidated_data)

imp_yearly_Consolidated.sort_values(['Country', 'Year'], inplace=True)
#Generating Export Data

exp_d = Export

country = exp_d['country'].unique()

year = exp_d['year'].unique()

tmp_cntry = pd.DataFrame(columns=exp_d.columns)

tmp_yer = pd.DataFrame(columns=exp_d.columns)

country_list = []

year_list = []

value_list = []

#Loop for segrigating country wise data

for cntry in country:

    tmp_cntry = exp_d[exp_d['country'] == cntry]

    #Loop for segrigating Contry wise data further into year wise

    for yer in year:

        tmp_yer =tmp_cntry[tmp_cntry['year'] == yer]

        tmp_total_year_value = tmp_yer['value'].sum()

        tmp_country = cntry

        tmp_year = yer

        year_list.append(tmp_year)

        country_list.append(tmp_country)

        value_list.append(tmp_total_year_value)

        

Consolidated_data = {'Country' : country_list, 'Year' : year_list ,'ExportValue' : value_list}

exp_yearly_Consolidated = pd.DataFrame(Consolidated_data)

exp_yearly_Consolidated.sort_values(['Country', 'Year'], inplace=True)



imp_yearly_Consolidated.set_index('Country')         #Setting Index as Country

exp_yearly_Consolidated.set_index('Country')         #Setting Index as Country

India_imp_exp_data = pd.merge(imp_yearly_Consolidated, exp_yearly_Consolidated, how='outer')

India_imp_exp_data.head()
# The new dataframe India_imp_exp_data can have null values because it is not necessary that from a country we have exported and imported simultaneously

#So we can fill those null values with 0

India_imp_exp_data.fillna(0, inplace=True)

#Checking for null values

India_imp_exp_data.isnull().sum()
#Creating a new column of Trade deficit in India_imp_exp_data

India_imp_exp_data['TradeDeficit'] = India_imp_exp_data['ImportValue'] - India_imp_exp_data['ExportValue'] 

India_imp_exp_data.head()

#Consolidating data Year Wise

Year = India_imp_exp_data['Year'].unique()

year = []

import_value = []

export_value = []

exp_imp_diff = []

for yer in Year:

    tmp = India_imp_exp_data[India_imp_exp_data['Year'] == yer]

    total_import_in_year = tmp['ImportValue'].values.sum()

    total_export_in_year = tmp['ExportValue'].values.sum()

    total_exp_imp_diff = tmp['TradeDeficit'].values.sum()

    year.append(yer)

    import_value.append(total_import_in_year)

    export_value.append(total_export_in_year)

    exp_imp_diff.append(total_exp_imp_diff)

        

Yearly_Imp_Exp = pd.DataFrame({'Year' : year, 'Import': import_value, 'Export': export_value, 'TradeDeficit' : exp_imp_diff })

Yearly_Imp_Exp.sort_values("Year", inplace=True)

Yearly_Imp_Exp
#Import Export Growth Chart



plt.plot('Year', 'Import', data=Yearly_Imp_Exp,marker='o', label='Import')

plt.plot('Year', 'Export', data=Yearly_Imp_Exp,marker='^', label='Export')

plt.plot('Year', 'TradeDeficit', data=Yearly_Imp_Exp,marker='|', label='TradeDeficit')

plt.legend()

plt.title('Yearly Import & Export and Trade Defict')

plt.xlabel('Year')

plt.ylabel('Value')

plt.show()
#Bar Plot

Year = Yearly_Imp_Exp['Year'].values

Import = Yearly_Imp_Exp['Import'].values

Export = Yearly_Imp_Exp['Export'].values

TradeDeficit = Yearly_Imp_Exp['TradeDeficit'].values

index = np.arange(len(Year))

width = 0.25

plt.bar(index, Import, width, label='Import')

plt.bar(index + width, Export, width,label='Export')

plt.bar(index + width + width, TradeDeficit, width,label='Deficit')



plt.ylabel('Value')

plt.title('Import & Export')



plt.xticks(index + width / 2, Year)

plt.legend(loc='upper center')

plt.show()

    
#Seaborn

Yearly_top_5_import = pd.DataFrame(columns=India_imp_exp_data.columns)

Yearly_top_5_export = pd.DataFrame(columns=India_imp_exp_data.columns)

Yearly_top_5_deficit = pd.DataFrame(columns=India_imp_exp_data.columns)

for yer in India_imp_exp_data['Year'].unique():

    temp = India_imp_exp_data[India_imp_exp_data['Year'] == yer]

    top5_import = temp.sort_values('ImportValue', ascending=False).iloc[:5, :]

    Yearly_top_5_import = pd.concat([Yearly_top_5_import, top5_import])

    top5_export = temp.sort_values('ExportValue', ascending=False).iloc[:5, :]

    Yearly_top_5_export = pd.concat([Yearly_top_5_export, top5_export])

    top5_deficit = temp.sort_values('TradeDeficit', ascending=False).iloc[:5, :]

    Yearly_top_5_deficit = pd.concat([Yearly_top_5_deficit, top5_deficit])

    



sns.relplot(x='Year', y='ImportValue', data=Yearly_top_5_import.sort_values("Year"), hue='Country', size='ImportValue', sizes=(50,200))

sns.relplot(x='Year', y='ExportValue', data=Yearly_top_5_export.sort_values("Year"), hue='Country', size='ExportValue', sizes=(50,200))

sns.relplot(x='Year', y='TradeDeficit', data=Yearly_top_5_deficit.sort_values("Year"), hue='Country', size='TradeDeficit', sizes=(50,200))
#Pie Chart

# Ploting top 6 countries from which India Imported Year on Year

Year = India_imp_exp_data['Year'].unique()

top_6 = pd.DataFrame(columns=India_imp_exp_data.columns)

Final_Import = pd.DataFrame(columns=India_imp_exp_data.columns)

fig, ax = plt.subplots(3,3, constrained_layout=True)   #Defining Figure 3X3 Subplot

fig.set_size_inches(18.5, 10.5)     #Setting Figure Size

fig.suptitle('Yearly % Import from top 6 Countries',  fontsize=16, y=1.02) #Setting Main Title

i = 0

j = 0

for yer in Year:

    tmp = India_imp_exp_data[India_imp_exp_data['Year'] == yer]

    top_5 = tmp.sort_values("ImportValue",ascending=False).iloc[:6, :]

    temp = tmp.sort_values("ImportValue",ascending=False).iloc[6:, :]

    other_Valuesum = temp['ImportValue'].values.sum()

    sixth = pd.DataFrame({'Country': "Other", 'Year': yer , 'ImportValue': other_Valuesum}, index=['6'])

    top_6 = pd.concat([top_5,sixth], sort=True)

    Final_Import = pd.concat([Final_Import,top_6], sort=True)

    #top_6.plot.pie(y='ImportValue')

    label_name = top_6['Country'].values.tolist()

    sizes = top_6['ImportValue'].values

    ax[i,j].pie(sizes,labels=label_name, autopct='%1.1f%%',shadow=True, startangle=90,wedgeprops = {'linewidth': 8}, counterclock = False, labeldistance = 1.2)

    ax[i,j].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax[i,j].title.set_text(yer)

    j = j + 1

    if j > 2:

        i = i + 1

        j = 0

plt.show() 

#Final_Import.drop(['ExportValue'], axis=1, inplace=True)

#Pie Chart

# Ploting top 6 countries to which India Exported Year on Year



Year = India_imp_exp_data['Year'].unique()

top_6 = pd.DataFrame(columns=India_imp_exp_data.columns)

Final_Export = pd.DataFrame(columns=India_imp_exp_data.columns)

fig, ax = plt.subplots(3,3, constrained_layout=True)   #Defining Figure 3X3 Subplot

fig.set_size_inches(18.5, 10.5)     #Setting Figure Size

fig.suptitle('Yearly % Export to top 6 Countries', fontsize=16, y=1.02) #Setting Main Title

i = 0

j = 0

for yer in Year:

    tmp = India_imp_exp_data[India_imp_exp_data['Year'] == yer]

    top_5 = tmp.sort_values("ExportValue",ascending=False).iloc[:6, :]

    temp = tmp.sort_values("ExportValue",ascending=False).iloc[6:, :]

    other_Valuesum = temp['ExportValue'].values.sum()

    sixth = pd.DataFrame({'Country': "Other", 'Year': yer , 'ExportValue': other_Valuesum}, index=['6'])

    top_6 = pd.concat([top_5,sixth], sort=True)

    Final_Export = pd.concat([Final_Export,top_6], sort = True)

    #top_6.plot.pie(y='ImportValue')

    label_name = top_6['Country'].values.tolist()

    sizes = top_6['ExportValue'].values

    ax[i,j].pie(sizes,labels=label_name, autopct='%1.1f%%',shadow=True, startangle=90,wedgeprops = {'linewidth': 8}, counterclock = False, labeldistance = 1.2)

    ax[i,j].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax[i,j].title.set_text(yer)

    j = j + 1

    if j > 2:

        i = i + 1

        j = 0

plt.show()

#Final_Export.drop(['ImportValue'], axis=1, inplace=True)   

    



def top_x_import_in_year(no_of_country, year):

    data = India_imp_exp_data[India_imp_exp_data['Year'] == year ].sort_values("ImportValue", ascending=False).iloc[:no_of_country, [0,2]].values

    return pd.DataFrame({'Country' : data[:, 0], 'ImportValue' : data[:, 1] })

 

fig, ax = plt.subplots(3,3)   #Defining Figure 3X3 Subplot

fig.set_size_inches(18.5, 24.5)     #Setting Figure Size

fig.suptitle('Yearly Import from top 10 Countries '  , fontsize=16, y=1.02) #Setting Main Title

i = 0

j = 0 

for Year in India_imp_exp_data['Year'].unique():

    df = top_x_import_in_year(10, Year)

    Country = df['Country'].values

    Import = df['ImportValue'].values

    Index = np.arange(len(Import))

    yr = Year

    ax[i,j].bar(Index, Import)

    ax[i,j].title.set_text(yr)

    ax[i,j].set_xlabel('Country', fontsize=10)

    ax[i,j].set_xticks(Index)

    ax[i,j].set_xticklabels(Country,fontsize=10, rotation=30)

    ax[i,j].set_ylabel('Import Value', fontsize=10)

    j = j + 1

    if j > 2:

        i = i + 1

        j = 0

    

#fig.delaxes(ax[4, 1]) 

plt.tight_layout()

plt.show()
def top_x_export_in_year(no_of_country, year):

    data = India_imp_exp_data[India_imp_exp_data['Year'] == year ].sort_values("ExportValue", ascending=False).iloc[:no_of_country, [0,3]].values

    return pd.DataFrame({'Country' : data[:, 0], 'ExportValue' : data[:, 1] })

 

fig, ax = plt.subplots(3,3)   #Defining Figure 3X3 Subplot

fig.set_size_inches(18.5, 24.5)     #Setting Figure Size

fig.suptitle('Yearly Export to top 10 Countries '  , fontsize=16, y=1.02) #Setting Main Title

i = 0

j = 0 

for Year in India_imp_exp_data['Year'].unique():

    df = top_x_export_in_year(10, Year)

    Country = df['Country'].values

    Export = df['ExportValue'].values

    Index = np.arange(len(Import))

    yr = Year

    ax[i,j].bar(Index, Export)

    ax[i,j].title.set_text(yr)

    ax[i,j].set_xlabel('Country', fontsize=10)

    ax[i,j].set_xticks(Index)

    ax[i,j].set_xticklabels(Country,fontsize=10, rotation=30)

    ax[i,j].set_ylabel('Export Value', fontsize=10)

    j = j + 1

    if j > 2:

        i = i + 1

        j = 0

    

#fig.delaxes(ax[4, 1]) 

plt.tight_layout()

plt.show()



    



def top_x_import_in_year(no_of_country, year):

    data = India_imp_exp_data[India_imp_exp_data['Year'] == year ].sort_values("ImportValue", ascending=False).iloc[:no_of_country, [0,2]].values

    return pd.DataFrame({'Year' : year, 'Country' : data[:, 0], 'ImportValue' : data[:, 1] })

def bottom_sum(indices, year):

    index = np.arange(indices)

    export_array = df[df.Year == year].ImportValue.values

    return export_array[index].sum()

def color_selection(country):

    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:olive', 'tab:cyan', 'black', 'grey', 'lime','lightgray','teal' ]

    Country = df['Country'].unique().tolist()

    l = np.arange(len(Country))

    index = Country.index(country)

    return color[index]

df = pd.DataFrame(columns=India_imp_exp_data.columns)

#df.drop('ExportValue', axis=1, inplace=True)

for year in India_imp_exp_data['Year'].unique():

    df = pd.concat([df, top_x_import_in_year(5, year)], sort=True)

countries = df['Country'].unique()

year_df = pd.DataFrame(columns=df.columns)

for year in df['Year'].unique():

    Country = []

    Export = []

    Year = []

    temp = df[df['Year'] == year]

    temp_cntry = countries

    not_in_list = list(set(temp_cntry)^set(df[df['Year'] == year].Country))

    for cntry in not_in_list:

        Country.append(cntry)

        Export.append(1.00)

        Year.append(year)

    year_df = pd.DataFrame({'Country': Country, 'ImportValue': Export, 'Year': Year})

    df = pd.concat([df, year_df], axis=0, sort=True)

    

df.sort_values(['Year','ImportValue'], ascending=[True, False], inplace=True)

fig = plt.figure(figsize=(20, 12))

index = np.arange(len(df['Year'].unique()))

width = 0.25

for year in df['Year'].unique():

    for position in np.arange(len(df['Country'].unique().tolist())):

        if position == 0:

            plt.bar(index[df['Year'].unique().tolist().index(year)], df[df.Year == year].ImportValue.values[position], width, color=color_selection(df[df.Year == year].Country.values[position]))

        if position > 0:

            plt.bar(index[df['Year'].unique().tolist().index(year)], df[df.Year == year].ImportValue.values[position], width, bottom=bottom_sum(position, year), color=color_selection(df[df.Year == year].Country.values[position]))

    

plt.ylabel('Value')

plt.title('Yearly Top 5  Importer')

plt.xticks(index + width / 2, df['Year'].unique())     

plt.legend(df['Country'].unique(),loc=2)

plt.show()
def top_x_export_in_year(no_of_country, year):

    data = India_imp_exp_data[India_imp_exp_data['Year'] == year ].sort_values("ExportValue", ascending=False).iloc[:no_of_country, [0,3]].values

    return pd.DataFrame({'Year' : year, 'Country' : data[:, 0], 'ExportValue' : data[:, 1] })

def bottom_sum(indices, year):

    index = np.arange(indices)

    export_array = df[df.Year == year].ExportValue.values

    return export_array[index].sum()

def color_selection(country):

    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:olive', 'tab:cyan']

    Country = df['Country'].unique().tolist()

    l = np.arange(len(Country))

    index = Country.index(country)

    return color[index]

#top_x_export_in_year(5,2016)

df = pd.DataFrame(columns=India_imp_exp_data.columns)

#df.drop('ImportValue', axis=1, inplace=True)

for year in India_imp_exp_data['Year'].unique():

    df = pd.concat([df, top_x_export_in_year(5, year)], sort=True)

countries = df['Country'].unique()

year_df = pd.DataFrame(columns=df.columns)

for year in df['Year'].unique():

    Country = []

    Export = []

    Year = []

    temp = df[df['Year'] == year]

    temp_cntry = countries

    not_in_list = list(set(temp_cntry)^set(df[df['Year'] == year].Country))

    for cntry in not_in_list:

        Country.append(cntry)

        Export.append(1.00)

        Year.append(year)

    year_df = pd.DataFrame({'Country': Country, 'ExportValue': Export, 'Year': Year})

    df = pd.concat([df, year_df], axis=0, sort=True)

    

df.sort_values(['Year','ExportValue'], ascending=[True, False], inplace=True)

fig = plt.figure(figsize=(20, 10))

index = np.arange(len(df['Year'].unique()))

width = 0.25

for year in df['Year'].unique():

    for position in np.arange(len(df['Country'].unique().tolist())):

        if position == 0:

            plt.bar(index[df['Year'].unique().tolist().index(year)], df[df.Year == year].ExportValue.values[position], width, color=color_selection(df[df.Year == year].Country.values[position]))

        if position > 0:

            plt.bar(index[df['Year'].unique().tolist().index(year)], df[df.Year == year].ExportValue.values[position], width, bottom=bottom_sum(position, year), color=color_selection(df[df.Year == year].Country.values[position]))

    

plt.ylabel('Value')

plt.title('Yearly Top 5  Exporter')

plt.xticks(index + width / 2, df['Year'].unique())     

plt.legend(df['Country'].unique(),loc=2)

plt.show()
#Generating Import Commodity wise data

Import_list = []

commodity_list = []

year_list = []

for cmdty in Import_data['Commodity'].unique():

    for year in Import_data['year'].unique():

        Import_value = Import_data[(Import_data['Commodity'] == cmdty) & (Import_data['year'] == year)].value.sum()

        commodity_list.append(cmdty)

        year_list.append(year)

        Import_list.append(Import_value)

Import_Commodity = pd.DataFrame({'Year': year_list, 'Commodity' : commodity_list, 'Import' : Import_list})

Import_Commodity.sort_values([ "Commodity", "Year"], inplace=True)

Import_Commodity.head()
#Generating Import Commodity wise data

Export_list = []

commodity_list = []

year_list = []

for cmdty in Export_data['Commodity'].unique():

    for year in Export_data['year'].unique():

        Export_value = Export_data[(Export_data['Commodity'] == cmdty) & (Export_data['year'] == year)].value.sum()

        commodity_list.append(cmdty)

        year_list.append(year)

        Export_list.append(Export_value)

Export_commodity = pd.DataFrame({'Year': year_list, 'Commodity' : commodity_list, 'Export' : Export_list})

Export_commodity.sort_values([ "Commodity", "Year"], inplace=True)

Export_commodity.head()
#Merging Import and Export data into a common dataframe

Import_Commodity.set_index('Commodity')

Export_commodity.set_index('Commodity')



Imp_Exp_Commodity_data = pd.merge(Import_Commodity, Export_commodity, how='outer')
Imp_Exp_Commodity_data.head()
yearly_top10_import = pd.DataFrame(columns=Imp_Exp_Commodity_data.columns)

for year in Imp_Exp_Commodity_data['Year'].unique():

    temp = Imp_Exp_Commodity_data[Imp_Exp_Commodity_data['Year'] == year]

    top_10_import = temp.sort_values('Import', ascending=False).iloc[:10, :]

    yearly_top10_import = pd.concat([yearly_top10_import, top_10_import])



def top_10_commodity_in_year( year):

    return yearly_top10_import[yearly_top10_import['Year'] == year]



fig, ax = plt.subplots(9,1)   #Defining Figure 3X3 Subplot

fig.set_size_inches(9.5, 50.5)     #Setting Figure Size

fig.suptitle('Top 10 Commodiities Imported Yearly'  , fontsize=16) #Setting Main Title

j = 0 

for Year in yearly_top10_import['Year'].unique():

    df = top_10_commodity_in_year(Year)

    Commodity = df['Commodity'].values

    Import = df['Import'].values

    Index = np.arange(len(Import))

    yr = Year

    ax[j].barh(Index, Import)

    ax[j].title.set_text(yr)

    ax[j].set_ylabel('Commodity', fontsize=10)

    ax[j].set_yticks(Index)

    ax[j].set_yticklabels(Commodity,fontsize=10)

    j = j + 1

plt.show()
yearly_top10_export = pd.DataFrame(columns=Imp_Exp_Commodity_data.columns)

for year in Imp_Exp_Commodity_data['Year'].unique():

    temp = Imp_Exp_Commodity_data[Imp_Exp_Commodity_data['Year'] == year]

    top_10_export = temp.sort_values('Import', ascending=False).iloc[:10, :]

    yearly_top10_export = pd.concat([yearly_top10_export, top_10_export])
def top_10_commodity_in_year( year):

    return yearly_top10_export[yearly_top10_export['Year'] == year]



fig, ax = plt.subplots(9,1)   #Defining Figure 3X3 Subplot

fig.set_size_inches(9.5, 50.5)     #Setting Figure Size

fig.suptitle('Top 10 Commodiities Exported Yearly'  , fontsize=16) #Setting Main Title

j = 0 

for Year in yearly_top10_import['Year'].unique():

    df = top_10_commodity_in_year(Year)

    Commodity = df['Commodity'].values

    Export = df['Export'].values

    Index = np.arange(len(Export))

    yr = Year

    ax[j].barh(Index, Export)

    ax[j].title.set_text(yr)

    ax[j].set_ylabel('Commodity', fontsize=10)

    ax[j].set_yticks(Index)

    ax[j].set_yticklabels(Commodity,fontsize=10)

    j = j + 1

plt.show()
#Data Only for 2018

hscode = [27, 71, 84]

Final = pd.DataFrame(columns=['HSCode', 'Commodity', 'Country', 'Import', 'Export'])

for hs in hscode:

    I_temp = Import_data[(Import_data['HSCode'] == hs) & (Import_data['year'] == 2018)].sort_values("value", ascending=False).head(10)

    I_temp = I_temp[['HSCode', 'Commodity' , 'country', 'value']]

    I_temp.columns = ['HSCode', 'Commodity', 'Country', 'Import']

    I_temp.set_index('Country')

    E_temp = Export_data[(Export_data['HSCode'] == hs) & (Export_data['year'] == 2018)].sort_values("value", ascending=False).head(10)

    E_temp = E_temp[['HSCode' , 'Commodity','country', 'value']]

    E_temp.columns = ['HSCode', 'Commodity', 'Country', 'Export']

    E_temp.set_index('Country')

    data_temp = pd.merge(I_temp, E_temp, how='outer')

    Final = pd.concat([Final , data_temp])

    

Final.fillna(0, inplace=True)

Final[(Final['Import'] > 0) & (Final['Export'] > 0)]

country = ['CHINA P RP', 'U S A']

Top_2= pd.DataFrame(columns=['HSCode', 'Commodity', 'Country', 'Import', 'Export'])

for cntry in country:

    I_temp = Import_data[(Import_data['country'] == cntry ) & (Import_data['year'] == 2018) 

                         & (Import_data['value'] >  500)].sort_values('value', ascending=False)

    I_temp = I_temp[['HSCode', 'Commodity' , 'country', 'value']]

    I_temp.columns = ['HSCode', 'Commodity', 'Country', 'Import']

    I_temp.set_index('Country')

    E_temp = Export_data[(Export_data['country'] == cntry ) & (Export_data['year'] == 2018) 

                         & (Export_data['value'] >  300)].sort_values('value', ascending=False)

    E_temp = E_temp[['HSCode' , 'Commodity','country', 'value']]

    E_temp.columns = ['HSCode', 'Commodity', 'Country', 'Export']

    E_temp.set_index('Country')

    data_temp = pd.merge(I_temp, E_temp, how='outer')

    Top_2 = pd.concat([Top_2 , data_temp])

    

Top_2.fillna(0, inplace=True)

Top_2

#Final[(Final['Import'] > 0) & (Final['Export'] > 0)]