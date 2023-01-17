import sqlite3  #connection with the database



import pandas as pd  #query data stored in pandas

pd.options.display.float_format = "{:.2f}".format  #reformat the pandas dataframe output in order to show the plain values

import numpy as np  #pandas built upon numpy, so it is necessary



import matplotlib.pyplot as plt  #plotting module

import matplotlib.patches as mpatches  #create my own legends and colors on the figures

import matplotlib.ticker as ticker  #provides broader customization options on the axes tickers



import seaborn as sns  #built upon matplotlib and pandas, used for visualization



import datetime  #SQL query date parsing
sql_database = '/kaggle/input/secondhand-car-market-data-parsing-dataset-v1/kaggle_sqlite'

conn = sqlite3.connect(sql_database)  #conn object with the SQLite DB



car_data_df = pd.DataFrame()  #empty DF

car_data_df = pd.read_sql("""

                            SELECT 

                                brand_name as 'brand',

                                strftime('%m', upload_date) as 'upload_month',

                                ad_price as 'price',

                                mileage as 'mileage'

                            FROM advertisements

                            JOIN brand ON advertisements.brand_id = brand.brand_id

                            WHERE brand_name IN ('BMW','AUDI', 'MERCEDES-BENZ');

                            """,

                            conn)  #SQL query; used pandas 'read_sql' method

print('Dataframe basic info:\n')

print(car_data_df.info())

print('\n\nDataframe data (top 5 rows):\n')

print(car_data_df.head())

print('\n\nDataframe description:\n')

print(car_data_df[['mileage','price']].describe())
fig, axes = plt.subplots(2,2, figsize = (25,15))  #initialize the figure and axes

sns.set(style = 'darkgrid')  #modifying the style of the figure



#grouped dataframes

ax00b_grouped_df = car_data_df.groupby(by=['upload_month']).count().reset_index()  #three brands combined

car_data_df_grouped = car_data_df.groupby(by=['upload_month', 'brand']).count().reset_index()  #separated by brands



"""--------------------------------------------------------------------------"""



#creating the barplot for all three brands combined; subplot[0,0]

ax00a = sns.barplot(

                ax=axes[0,0],

                x=car_data_df.upload_month,

                y=car_data_df.price,

                palette = "GnBu_d",

                errcolor = '#FF5511',

                capsize = 0.2)



ax00a.set_xlabel('Month', fontsize=15.0)  #x label

ax00a.set_ylabel('Price (in million HUF)', fontsize=15.0)  #y label



#secondary axis; lineplot

ax00b = ax00a.twinx()  #creating the secondary y-axis for the lineplot; subplot[0,0]

ax00b = sns.lineplot(

        x=ax00b_grouped_df.upload_month,

        y=ax00b_grouped_df.price,

        linewidth = 1.5,

        color = '#FF5511',

        marker = 'x',

        markersize = 15.0,

        markeredgecolor = '#FF5511',

        markeredgewidth = 3.0)



ax00b.set_ylabel('Total number of advertisements (count)', fontsize=15.0)



"""--------------------------------------------------------------------------"""



#creating the barplot for AUDI brand; subplot[0,1]

ax01a = sns.barplot(

        ax=axes[0,1],

        x=car_data_df.upload_month.loc[car_data_df['brand']=='AUDI'],

        y=car_data_df.price.loc[car_data_df['brand']=='AUDI'],

        palette = "Blues",

        errcolor = '#BBC400',

        capsize = 0.2)



ax01a.set_xlabel('Month', fontsize=15.0)

ax01a.set_ylabel('Price (in million HUF)', fontsize=15.0)

ax01a.set_title('AUDI')



#secondary axis; lineplot

ax01b = ax01a.twinx() #creating the secondary y-axis for the lineplot; subplot[0,1]

ax01b = sns.lineplot(

        x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='AUDI'],

        y=car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='AUDI'],

        linewidth = 0,

        color = '#BBC400',

        marker = 'o',

        markersize = 15.0,

        markeredgecolor = '#BBC400',

        markeredgewidth = 3.0)



ax01b.set_ylabel('Total number of advertisements (count)', fontsize=15.0)



"""--------------------------------------------------------------------------"""



#creating the barplot for BMW brand; subplot[1,0]

ax10a = sns.barplot(

        ax=axes[1,0],

        x=car_data_df.upload_month.loc[car_data_df['brand']=='BMW'],

        y=car_data_df.price.loc[car_data_df['brand']=='BMW'],

        palette = "Greys",

        errcolor = '#0068C4',

        capsize = 0.2)



ax10a.set_xlabel('Month', fontsize=15.0)

ax10a.set_ylabel('Price (in million HUF)', fontsize=15.0)

ax10a.set_title('BMW')



#secondary axis; lineplot

ax10b = ax10a.twinx() #creating the secondary y-axis for the lineplot; subplot[0,1]

ax10b = sns.lineplot(

        x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='BMW'],

        y=car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='BMW'],

        linewidth = 0,

        color = '#0068C4',

        marker = 'o',

        markersize = 15.0,

        markeredgecolor = '#0068C4',

        markeredgewidth = 3.0)



ax10b.set_ylabel('Total number of advertisements (count)', fontsize=15.0)



"""--------------------------------------------------------------------------"""



#creating the barplot for MERCEDES-BENZ brand; subplot[1,1]

ax11a = sns.barplot(

        ax=axes[1,1],

        x=car_data_df.upload_month.loc[car_data_df['brand']=='MERCEDES-BENZ'],

        y=car_data_df.price.loc[car_data_df['brand']=='MERCEDES-BENZ'],

        palette = "Greens",

        errcolor = '#000000',

        capsize = 0.2)



ax11a.set_xlabel('Month', fontsize=15.0)

ax11a.set_ylabel('Price (in million HUF)', fontsize=15.0)

ax11a.set_title('MERCEDES-BENZ')



#secondary axis; lineplot

ax11b = ax11a.twinx()  #creating the secondary y-axis for the lineplot; subplot[1,1]

ax11b = sns.lineplot(

        x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='MERCEDES-BENZ'],

        y=car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='MERCEDES-BENZ'],

        linewidth = 0,

        color = '#000000',

        marker = 'o',

        markersize = 15.0,

        markeredgecolor = '#000000',

        markeredgewidth = 3.0)



ax11b.set_ylabel('Total number of advertisements (count)', fontsize=15.0)



"""--------------------------------------------------------------------------"""



fig.savefig('month_price_fig1.png')
fig2, axes2 = plt.subplots(2,figsize = (25,20), sharex = True)  #initialize the figure and axes

fig2.subplots_adjust(hspace = 0.01)  #reducing the space between the subplots

sns.set(style = 'darkgrid')  #modifying the style



#visualization parameters

palette = {'AUDI':'#BBC400', 'BMW':'#0068C4', 'MERCEDES-BENZ':'#000000'}

bmw_patch = mpatches.Patch(color='#0068C4', label='BMW')

audi_patch = mpatches.Patch(color='#BBC400', label='AUDI')

merc_patch = mpatches.Patch(color='#000000', label='MERCEDES-BENZ')



"""--------------------------------------------------------------------------"""

#creating the lineplot; subplot[0]

ax00 = sns.lineplot(

                x=car_data_df_grouped.upload_month,

                y=car_data_df_grouped.price,

                hue = car_data_df_grouped.brand,

                palette = palette,

                linewidth = 2.5,

                ax=axes2[0])



ax00.set_ylabel('Count of advertisements', fontsize = 20.0)

ax00.tick_params(axis='y', labelsize=15.0)



ax00.legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)  #creating the legend manually



"""--------------------------------------------------------------------------"""



#creating the height values of the stacked bars

top_bmw = car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='BMW'].copy()

top_bmw.reset_index(drop=True, inplace=True)



top_audi = car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='AUDI'].copy()

top_audi.reset_index(drop=True, inplace=True)



top_merc = car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='MERCEDES-BENZ'].copy()

top_merc.reset_index(drop=True, inplace=True)



#creating the lineplot; subplot[1]

ax01 = sns.barplot(

                x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='MERCEDES-BENZ'].reset_index(drop=True),

                y=top_bmw + top_audi + top_merc,

                color = '#000000')



ax01 = sns.barplot(

                x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='AUDI'].reset_index(drop=True),

                y=top_bmw + top_audi,

                color = '#BBC400')



ax01 = sns.barplot(

                x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='BMW'].reset_index(drop=True),

                y=top_bmw,

                color = '#0068C4')



#creating the data labels for the columns

for month in np.sort(car_data_df_grouped.upload_month.unique()):

    month=int(month)

    plt.text(

        int(month-1)-0.1,

        int(top_bmw.iloc[month-1] + top_audi.iloc[month-1] + top_merc.iloc[month-1]) +10,

        str(top_bmw.iloc[month-1] + top_audi.iloc[month-1] + top_merc.iloc[month-1]),

        fontsize = 'large',

        fontstyle = 'normal')





ax01.set_ylabel('Count of advertisements', fontsize = 20.0)

ax01.tick_params(axis='y', labelsize=15.0)



ax01.set_xlabel('Month', fontsize = 15.0)

ax01.tick_params(axis='x', labelsize=15.0)



ax01.legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0) #creating the legend manually



"""--------------------------------------------------------------------------"""



fig2.savefig("month_count_fig2.png")
#trim the extreme values from the dataset

#removing rows with price less than 100 000 HUFs, more than 20 000 000 HUFs and milage less than 1500 kms and higher than 400 000 kms

car_data_df_trimmed = car_data_df.copy().drop(car_data_df.loc[(car_data_df.mileage>400000)|(car_data_df.mileage<1500)|(car_data_df.price<100000)|(car_data_df.price>20000000)].index, inplace = False)



fig3, ax3 = plt.subplots(1, figsize = (25,20))

ax3= sns.scatterplot(

            x=car_data_df_trimmed.price,

            y=car_data_df_trimmed.mileage,

            alpha = 0.2,

            hue = car_data_df_trimmed.brand,

            palette = {'AUDI':'#BBC400','BMW':'#0068C4','MERCEDES-BENZ':'#000000'})



ax3.ticklabel_format(style='plain', axis='y')  #y-axis scientific notation turned off

ax3.tick_params(axis='y', labelsize=20.0)

ax3.set_ylabel('Mileage', fontsize = 25.0)



ax3.ticklabel_format(style='sci', axis='x', scilimits=(6,6))  #x-axis scientific notation turned off

ax3.tick_params(axis='x', labelsize=20.0, labelrotation=45)

ax3.set_xlabel('Price (in million HUF)', fontsize = 25.0)



ax3.legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)



fig3.savefig('mileage_price_fig3.png')

fig4, ax4 = plt.subplots(2, figsize = (25,20), sharex=False)

ax00 = sns.distplot(

                a=car_data_df.price.loc[(car_data_df.brand=='AUDI')],

                bins=100,

                color = '#BBC400',

                hist=True,

                kde=False,

                kde_kws={'shade': True, 'linewidth': 3},

                ax=ax4[0])

ax00 = sns.distplot(

                a=car_data_df.price.loc[(car_data_df.brand=='BMW')],

                bins=100,

                color = '#0068C4',

                hist=True,

                kde=False,

                kde_kws={'shade': True, 'linewidth': 3},

                ax=ax4[0])



ax00 = sns.distplot(

                a=car_data_df.price.loc[(car_data_df.brand=='MERCEDES-BENZ')],

                bins=100,

                color = '#000000',

                hist=True,

                kde=False,

                kde_kws={'shade': True, 'linewidth': 3},

                ax=ax4[0])



ax4[0].ticklabel_format(style='plain', axis='y')

ax4[0].tick_params(axis='y', labelsize=20.0)

ax4[0].set_ylabel('Count of advertisements', fontsize = 25.0)



ax4[0].ticklabel_format(style='sci', axis='x', scilimits=(6,6))

ax4[0].tick_params(axis='x', labelsize=20.0, labelrotation=45)

ax4[0].set_xlabel('Price (in million HUF)', fontsize = 25.0)



ax4[0].legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)



"""--------------------------------------------------------------------------"""



car_data_df_price_trimmed = car_data_df.copy().drop(car_data_df.loc[(car_data_df.price<100000)|(car_data_df.price>20000000)].index, inplace = False)



ax01 = sns.distplot(

                a=car_data_df_price_trimmed.price.loc[(car_data_df_price_trimmed.brand=='AUDI')],

                bins=100,

                color = '#BBC400',

                hist=True,

                kde=False,

                kde_kws={'shade': True, 'linewidth': 3},

                ax=ax4[1])

ax01 = sns.distplot(

                a=car_data_df_price_trimmed.price.loc[(car_data_df_price_trimmed.brand=='BMW')],

                bins=100,

                color = '#0068C4',

                hist=True,

                kde=False,

                kde_kws={'shade': True, 'linewidth': 3},

                ax=ax4[1])



ax01 = sns.distplot(

                a=car_data_df_price_trimmed.price.loc[(car_data_df_price_trimmed.brand=='MERCEDES-BENZ')],

                bins=100,

                color = '#000000',

                hist=True,

                kde=False,

                kde_kws={'shade': True, 'linewidth': 3},

                ax=ax4[1])



ax4[1].ticklabel_format(style='plain', axis='y')  #y-axis scientific notation turned off

ax4[1].tick_params(axis='y', labelsize=20.0)

ax4[1].set_ylabel('Count of advertisements', fontsize = 25.0)



ax4[1].ticklabel_format(style='sci', axis='x', scilimits=(6,6))  #x-axis scientific notation turned off

ax4[1].tick_params(axis='x', labelsize=20.0, labelrotation=45)

ax4[1].set_xlabel('Price (in million HUF); trimmed price: 100 000< price < 20 000 000', fontsize = 25.0)

#ax4[1].xaxis.set_major_locator(ticker.MaxNLocator(30))

#ax4[1].xaxis.set_minor_locator(ticker.MaxNLocator(30))



ax4[1].legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)



fig4.savefig('price_dist_fig4.png')
fig5, ax5 = plt.subplots(1, figsize = (25,20))



ax00=sns.boxplot(

                x='upload_month',

                y='price',

                data=car_data_df,

                hue = 'brand',

                whis=[10, 90],

                sym="",

                palette = {'AUDI':'#BBC400','BMW':'#0068C4','MERCEDES-BENZ':'#FFFFFF'},

                )



ax5.ticklabel_format(style='sci', axis='y', scilimits=(6,6))  #y-axis scientific notation turned off

ax5.tick_params(axis='y', labelsize=20.0)

ax5.set_ylabel('Price in million HUF (from 10th to 90th percentiles)', fontsize = 25.0)





ax5.tick_params(axis='x', labelsize=20.0)

ax5.set_xlabel('Month', fontsize = 25.0)



merc_patch = mpatches.Patch(color='#FFFFFF', label='MERCEDES-BENZ')



ax5.legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)



fig5.savefig('boxplot_fig5.png')