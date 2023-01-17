import numpy as np

import pandas as pd
water = pd.read_csv(

    '../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/RawData/Basic and safely managed drinking water services.csv'

)
water.head(2)
water.shape
water.columns
water.isnull().sum()
def parse_who_data(df):

    df = df.loc[:, [

        'GHO (DISPLAY)',

        'YEAR (DISPLAY)',

        'REGION (DISPLAY)',

        'COUNTRY (CODE)',

        'COUNTRY (DISPLAY)',

        'RESIDENCEAREATYPE (DISPLAY)',

        'Numeric'

    ]]

    

    df['Index'] = df['GHO (DISPLAY)'] + ' - ' + df['RESIDENCEAREATYPE (DISPLAY)']

    

    df = df.drop(columns=[

        'GHO (DISPLAY)',

        'RESIDENCEAREATYPE (DISPLAY)'

    ])



    df = df.rename(columns={

        'YEAR (DISPLAY)':'Year',

        'REGION (DISPLAY)':'Region',

        'COUNTRY (CODE)':'Country code',

        'COUNTRY (DISPLAY)':'Country',

        'Numeric':'Value'

    })

    

    df = df.pivot_table(index = ['Year', 'Region', 'Country code', 'Country'], columns='Index', values='Value').reset_index()

    df.columns = df.columns.rename('')

    

    return df
water = parse_who_data(water)
water.head()
sanitation = pd.read_csv(

    '../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/RawData/Basic and safely managed sanitation services.csv'

)
sanitation.shape
sanitation.head(2)
sanitation.columns
sanitation.isnull().sum()
sanitation = parse_who_data(sanitation)
sanitation.head()
hand = pd.read_csv(

    '../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/RawData/Handwashing with soap.csv'

)
hand.shape
hand.head(2)
hand.columns
hand.isnull().sum()
hand = parse_who_data(hand)
hand.head()
defec = pd.read_csv(

    '../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/RawData/Open defecation.csv'

)
defec.shape
defec.head(2)
defec.columns
defec.isnull().sum()
defec = parse_who_data(defec)
defec.head()
water.shape
sanitation.shape
hand.shape
defec.shape
join_columns = ['Year', 'Region', 'Country code', 'Country']

metrics = water.merge(sanitation, how='outer', left_on=join_columns, right_on=join_columns)

metrics = metrics.merge(hand, how='outer', left_on=join_columns, right_on=join_columns)

metrics = metrics.merge(defec, how='outer', left_on=join_columns, right_on=join_columns)
metrics.shape
metrics.head()
metrics.to_csv('/kaggle/working/indexes.csv', index=False)