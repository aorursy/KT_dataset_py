import glob

import pandas as pd

import numpy as np

import seaborn as sns



cm = sns.light_palette("grey", as_cmap=True)
path =r'../input/global-top-250-retailers/' 

allFiles = glob.glob(path + "/cos*.csv") 

allFiles = np.sort(allFiles)



frame = pd.DataFrame()

list_ = []

cols = [ 'name', 

        'country_of_origin', 

        'retail_revenue', 

        'dominant_operational_format', 

        'countries_of_operation', 

        'retail_revenue_cagr'

       ]

i = 1



for file_ in allFiles:

    df = pd.read_csv(file_,index_col=None, sep=";", usecols=cols, verbose=False) 

    df["year"] = 2016+i

    list_.append(df)

    del df

    i = i + 1



df = pd.concat(list_)

del list_



#df = df[df["year"].isin([2018, 2019])]

df.sample(10)
table_country = pd.pivot_table(df, 

                               values='retail_revenue', 

                               index=['country_of_origin'], 

                               columns=['year'], 

                               aggfunc=np.sum

                              )

table_country = table_country.reset_index()

table_country.columns = ["country_of_origin", "revenue_2017", "revenue_2018", "revenue_2019"]

sum_2019 = table_country['revenue_2019'].sum()

table_country = table_country.sort_values(by="revenue_2019", ascending=False).head(10)



table_country
print(f"In 2019 ${np.round(sum_2019.astype(int)/1e6, 1)} Trillion revenue was generated by top 250 retailers; {np.round(100*table_country['revenue_2019'].sum()/sum_2019)}% of these originate from {len(table_country)} countries")
df_us = df[df["country_of_origin"].isin(["US"]) & (df["year"]==2019)]

cols = ['name', 

        'retail_revenue',

        'dominant_operational_format', 

        'countries_of_operation',

        'retail_revenue_cagr']

df_us = df_us[cols].sort_values(by="retail_revenue", ascending=False).reset_index(drop=True)



df_us.head(10)
agg_dict = {'retail_revenue':['sum', 'median'], 

            'retail_revenue_cagr':['median'],

            'name': lambda x: len(x.unique()),

           }



us_agg = df_us.groupby(["dominant_operational_format"]).agg(agg_dict).reset_index()

us_agg.columns = [' '.join(col).strip() for col in us_agg.columns.values]

us_agg.columns = us_agg.columns.str.replace(' ','_')

us_agg.rename({us_agg.columns[-1]: "n_firms"}, axis=1, inplace=True)



us_agg = us_agg.sort_values(by="retail_revenue_sum", ascending=False).reset_index(drop=True)



us_agg["retail_revenue_sum"] /= 1e3

us_agg["retail_revenue_median"] /= 1e3



rename_dict = {

    "dominant_operational_format": "Retail format",

    "retail_revenue_sum": "Total revenue",

    "retail_revenue_median": "Median revenue",

    "retail_revenue_cagr_median": "Median CAGR",

    "n_firms": "# Firms"

}

format_dict = {

    'Total revenue':'${0:,.0f}', 

    'Median revenue':'${0:,.0f}',

    'Median CAGR': '{:.1%}'          

}



#us_agg

(

    us_agg.rename(rename_dict, axis=1)

    .style

    .bar(subset=['Median CAGR'], align='mid', color=['#d65f5f', '#5fba7d'])

    .background_gradient(subset=['Total revenue', '# Firms'], cmap=cm)

    .format(format_dict).hide_index()

    .bar(color='#5fba7d', vmin=0, subset=['Median revenue'], align='zero')

    .set_caption('2019 United States top retailers ($Billion)')

)
agg_dict = {'retail_revenue':['sum', 'median'], 

            'retail_revenue_cagr':['median'],

            'name': lambda x: len(x.unique()),

           }



us_agg = df[df["year"]==2019].groupby(["dominant_operational_format"]).agg(agg_dict).reset_index()

us_agg.columns = [' '.join(col).strip() for col in us_agg.columns.values]

us_agg.columns = us_agg.columns.str.replace(' ','_')

us_agg.rename({us_agg.columns[-1]: "n_firms"}, axis=1, inplace=True)



us_agg = us_agg.sort_values(by="retail_revenue_sum", ascending=False).reset_index(drop=True)



us_agg["retail_revenue_sum"] /= 1e3

us_agg["retail_revenue_median"] /= 1e3



rename_dict = {

    "dominant_operational_format": "Retail format",

    "retail_revenue_sum": "Total revenue",

    "retail_revenue_median": "Median revenue",

    "retail_revenue_cagr_median": "Median CAGR",

    "n_firms": "# Firms"

}

format_dict = {

    'Total revenue':'${0:,.0f}', 

    'Median revenue':'${0:,.0f}',

    'Median CAGR': '{:.1%}'          

}



#us_agg

(

    us_agg.rename(rename_dict, axis=1)

    .style

    .bar(subset=['Median CAGR'], align='mid', color=['#d65f5f', '#5fba7d'])

    .background_gradient(subset=['Total revenue', '# Firms'], cmap=cm)

    .format(format_dict).hide_index()

    .bar(color='#5fba7d', vmin=0, subset=['Median revenue'], align='zero')

    .set_caption('2019 Global top retailers ($Billion)')

)