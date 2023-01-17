import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#load CSV files and create Dataframes

export_data=pd.read_csv('../input/india-trade-data/2018-2010_export.csv')

import_data=pd.read_csv('../input/india-trade-data/2018-2010_import.csv')
# get metadata information of the export data

print(export_data.info())

print(export_data.head())
# let us get rid off null rows from export data

export_data=export_data.dropna(how='any')
print(import_data.info())

print(import_data.head())
# let us get rid off null rows from import data

import_data=import_data.dropna(how='any')


country_from_india= export_data.groupby(['country'])['value'].sum()

country_from_india=country_from_india.reset_index()

country_from_india=country_from_india.sort_values('value',ascending=False).head(15)



print('The following are top 15 countries who have imported from India')

print('\t')

print(country_from_india)



fig1=plt.figure()

fig1.set_figheight(5)

fig1.set_figwidth(20)



a1=fig1.add_subplot(1,2,1)

sns.barplot(x=country_from_india.value,y=country_from_india.country, data=country_from_india, palette='Spectral')



b1=fig1.add_subplot(1,2,2)

explode=(0.3,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0)

plt.pie(country_from_india.value,labels=country_from_india.country,shadow=True, startangle=90,explode=explode)

plt.title('Most exported Coutries',fontsize=15)

top_years_of_export=export_data.groupby(['year'])['value'].sum()

top_years_of_export=top_years_of_export.reset_index().sort_values('value',ascending=False).head(7)



print('The following are top 7 years where India exported the most')

print('\t')

print(top_years_of_export)



fig2=plt.figure()

fig2.set_figheight(5)

fig2.set_figwidth(12)



a2=fig2.add_subplot(1,2,1)

sns.barplot(x=top_years_of_export.year,y=top_years_of_export.value, data=top_years_of_export, palette='Spectral')



b2=fig2.add_subplot(1,2,2)

explode=(0.3,0.2,0,0,0,0,0)

plt.pie(top_years_of_export.value,labels=top_years_of_export.year,shadow=True, startangle=90,explode=explode)

plt.title('TOP Export Years',fontsize=15)
product_from_india=export_data.groupby(['Commodity'])['value'].sum()

product_from_india=product_from_india.reset_index().sort_values('value',ascending=False).head(15)



print('The following are top 15 products from India which yeided the most')

print('\t')

print(product_from_india)



fig3=plt.figure()

fig3.set_figheight(5)

fig3.set_figwidth(15)



a3=fig3.add_subplot(1,2,1)

sns.barplot(x=product_from_india.value,y=product_from_india.Commodity, data=product_from_india, palette='Spectral').set_title('Top 15 exported Commodities')
country_product_from_india=export_data.groupby(['country','Commodity'])['value'].sum()

country_product_from_india=country_product_from_india.reset_index().sort_values('value',ascending=False)



print('The following are top 15 products from India exported to respective countries which yeided the most')

print('\t')

print(country_product_from_india.head(15))
year_lst=[2010,2011,2012,2013,2014,2015,2016,2017]

US_China_UAE=export_data.groupby(['year'])



for i in year_lst:

    gt_grp=US_China_UAE.get_group(i)

    gt_grp=gt_grp.reset_index().sort_values('value',ascending=False).head(5)

    print('The following are top 5 Commodities of the year', i)

    print('\t')

    print(gt_grp[['Commodity','value','country']])
most_imported=import_data.sort_values('value',ascending=False).head(10)

print('The following are top 10 Commodities which India Imported the Most from a single Nation in a calender Year')

print('\t')

print(most_imported) 
item_value= import_data.groupby(['Commodity'])['value'].sum()

item_value=item_value.reset_index().sort_values(['value'],ascending=False).head(10)

print('The following are top 10 Commodities which India Imported the Most of all time')

print('\t')

print(item_value)



fig4=plt.figure()

fig4.set_figheight(5)

fig4.set_figwidth(20)



a4=fig4.add_subplot(1,2,1)

sns.barplot(x=item_value.value,y=item_value.Commodity, data=item_value, palette='Spectral').set_title('Top 10 Imported Commodities of all Time')
Plastic=import_data.loc[import_data['Commodity']=='PLASTIC AND ARTICLES THEREOF.']



Plastic_country=Plastic.groupby(['country']) ['value'].sum()

Plastic_country=Plastic_country.reset_index().sort_values(['value'],ascending=False).head(10)



fig5=plt.figure()

fig5.set_figheight(5)

fig5.set_figwidth(15)



a5=fig5.add_subplot(1,2,1)

sns.barplot(x=Plastic_country.value,y=Plastic_country.country, data=Plastic_country, palette='Spectral').set_title('Top 10 countries who export Plastic to India')
Plastic_year=Plastic.groupby(['year']) ['value'].sum()

Plastic_year=Plastic_year.reset_index().sort_values(['value'],ascending=False).head(10)



fig6=plt.figure()

fig6.set_figheight(5)

fig6.set_figwidth(20)



a6=fig6.add_subplot(1,2,1)

explode=(0.3,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1)

plt.pie(Plastic_year.value,labels=Plastic_year.year,shadow=True, startangle=90,explode=explode)

plt.title('Yearly Plastic Import Value',fontsize=15)