import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
df_io=pd.read_csv('../input/NFA 2018.csv')

df_io.head(10)
df_io.info()
df_io.columns
df_io.rename(index=str, columns={'Percapita GDP (2010 USD)': 'Percapita_GDP_2010_USD'}, inplace=True)

df_io.columns
df_io['record'].unique()
df_io['record'].nunique()
df_io['year'].min()
df_io['year'].max()
def num_nan(df):

    return df.shape[0]-df.count()



land_types=['crop_land', 'grazing_land', 'forest_land', 'fishing_ground','built_up_land', 'carbon', 'total']



(df_io[['record']+land_types]

    .groupby('record')

    .agg(num_nan)

)
record_nan_countries = (df_io[['country']+['record']+land_types]

                            .set_index  ('country')

                            .isna       ()

                            .sum        (axis=1)

                            .loc        [lambda x: x>0]

                            .index

                            .unique     ()

                        )

record_nan_countries
(df_io[['country','Percapita_GDP_2010_USD','population']]

            .groupby     ('country')

            .agg         (num_nan)

            .sort_values (['Percapita_GDP_2010_USD','population'], ascending=[0, 0])  

            .query       ('Percapita_GDP_2010_USD>0')

)
GDP_nan_countries= (df_io[['country','Percapita_GDP_2010_USD','population']]

                        .groupby     ('country')

                        .agg         (num_nan)

                        .sort_values (['Percapita_GDP_2010_USD','population'], ascending=[0, 0])  

                        .query       ('Percapita_GDP_2010_USD>0')

                        .index

                    )

GDP_nan_countries
df_io['country'].unique()
df_io['country'].nunique()
world = df_io[df_io['country']=='World']

world.head()
world.info()
def year_range(s): #calulates the year range for the countries

    return s.max()-s.min()+1
df_all_countries= df_io[df_io['country']!='World']

data_year_range=(df_all_countries

                        .groupby           ('country')['year']

                        .agg               ([np.min,np.max, year_range,'nunique'])

                        .sort_values       (['year_range'], ascending = False)

                 )



data_year_range
data_year_range.query("year_range!=nunique")

#if no rows are returned, the year ranges and the unique numbers of years match for all countries, thus the data is consecutive
(data_year_range['year_range']

                         .value_counts()

                         .sort_index(ascending=False)

)
(data_year_range

             .query("year_range==54")['amin']

             .unique()

)             
(data_year_range

             .query("year_range==54")['amax']

             .unique()

)   
data_year_range.query("year_range!=54")
#slicing out NZ data from out df_io

df_NZ_new= (df_all_countries

                     .loc[df_all_countries['country']=='New Zealand'] 

                     .loc[df_all_countries['year']==2013]

            )     

df_NZ_new['year']=2014

df_NZ_new[land_types+['Percapita_GDP_2010_USD','population']]=np.nan

df_NZ_new
#concatinate with existing data

df_NZ_exist= df_all_countries.loc[df_all_countries['country']=='New Zealand']



df_NZ=pd.concat([df_NZ_exist,df_NZ_new])



df_NZ
inter_df=df_NZ[df_NZ['country']=='A']

inter_df

df_NZ[df_NZ['record']=='EFImportsPerCap']
inter_df=df_NZ[df_NZ['country']=='A']



for r in df_NZ['record'].unique():

    inter=df_NZ[df_NZ['record']==r].interpolate()

    inter_df=pd.concat([inter_df,inter[inter['year']==2014]])

    

inter_df
df_all_countries = pd.concat([df_all_countries,inter_df])

df_all_countries

df_all_countries[df_all_countries['country']=='New Zealand']
##slicing out NZ data from out df_io, then concatinating to df_io with year 2014

#df_NZ_new= (df_all_countries

#                     .loc[df_all_countries['country']=='New Zealand'] 

#                     .loc[df_all_countries['year']==2013]

#            )     

#df_NZ_new['year']=2014

#df_all_countries = pd.concat([df_all_countries,inter_df])

#df_all_countries

#df_all_countries[df_all_countries['country']=='New Zealand']
data_year_range1=(df_all_countries

                                .groupby           ('country')['year']

                                .agg               ([np.min,np.max, year_range,'nunique'])

                                .sort_values       (['year_range'], ascending = False)

                 )

data_year_range1.loc['New Zealand']
drop_countries=sorted(list(data_year_range1[(data_year_range1.amin>1973) | (data_year_range1.amax!=2014)].index))

drop_countries
df_all_clean= df_all_countries.drop(df_all_countries[df_all_countries.country.isin(drop_countries)].index)

#since some the remaining countries have data for years before 1975, we also drop those years from our working DF

df_all_clean.drop(df_all_clean[df_all_clean['year']<1975].index,inplace = True)    

df_all_clean.head()
df_all_clean.info()
(df_all_clean

             .groupby('UN_region')['country']

             .nunique()

             .sort_values(ascending=False)

)
(df_all_clean

             .groupby('UN_region')['country']

             .nunique()

             .plot

             .pie()

)
(df_all_clean

             .groupby('UN_subregion')['country']

             .nunique()

             .sort_values(ascending=False)

)
pt = (pd

        .pivot_table(df_all_clean,values = 'total',index=['UN_region', 'year'],columns=['record'],aggfunc='sum')[['BiocapTotGHA','EFConsTotGHA']]

        .reset_index()

        .set_index('UN_region')

     )

pt2=(pd.pivot_table(df_all_clean,values = 'population',index=['UN_region', 'year'],columns=['record'],aggfunc='sum')[['BiocapTotGHA']]

        .rename(index=str, columns={'BiocapTotGHA': 'population'})

        .reset_index()

        .drop(['year'],axis=1)

        .set_index('UN_region')

     )  

    
result_pt = pd.concat([pt, pt2], axis=1, join='inner')

result_pt['BiocapPerCap_region']=result_pt['BiocapTotGHA']/result_pt['population']

result_pt['EFConsPerCap_region']=result_pt['EFConsTotGHA']/result_pt['population']

result_pt2 = (result_pt[['year','BiocapPerCap_region','EFConsPerCap_region']]

                .reset_index()

                .set_index(['UN_region','year'])

            )

result_pt2
result_pt2.max()
N = 3

fig = plt.figure(figsize=(15, 15))

for i,k in enumerate(df_all_clean.UN_region.unique()):   

    ax_num = fig.add_subplot(N, N, i+1)

    ax_num.set_title (k)

    ax_num.set_ylim ((0,20))

    result_pt2.loc[k].plot(ax=ax_num)

    

    

fig.tight_layout()

plt.show()
world.head()
pt_w_CAP = pd.pivot_table(world,values = 'total',index=['year'],columns=['record'],aggfunc='sum')[['BiocapPerCap','EFConsPerCap']]

(pt_w_CAP.plot()

)
pt_w_EF_by_land=(world[world['record']=='EFConsTotGHA'][['year']+land_types[:-1]]

                    .set_index(['year'])

                )



pt_w_EF_by_land.plot.area(figsize=(12, 12))