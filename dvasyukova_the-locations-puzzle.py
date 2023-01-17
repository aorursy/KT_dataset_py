import pandas as pd
train = pd.read_csv('../input/train.csv', nrows=10)
train.columns
train = pd.read_csv('../input/train.csv', usecols = ['posa_continent', 
       'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance','hotel_continent', 
       'hotel_country', 'hotel_market'], nrows=1000000).dropna()
distaggs = (train.groupby(['user_location_country','hotel_country'])
            ['orig_destination_distance']
            .agg(['min','mean','max','count']))
distaggs.sort_values(by='min').head(20)
c66 = train[train.user_location_country==66]
c66.user_location_region.unique().shape
c66in = c66[c66.hotel_country==50]
(c66in.groupby(['user_location_region','hotel_market'])['orig_destination_distance']
      .agg(['min','mean','max','count'])
      .sort_values(by='max',ascending=False).head(20))
hawaii = (c66in[c66in.hotel_market == 212]
          .groupby(['user_location_region','user_location_city'])
          ['orig_destination_distance']
          .agg(['min','mean','max','count'])
          .sort_values(by='count',ascending=False))
hawaii.head(10)
fromny = (c66in[(c66in.user_location_region == 348) & 
                (c66in.user_location_city == 48862)]
          .groupby(['hotel_market'])
          ['orig_destination_distance']
          .agg(['min','mean','max','count'])
          .sort_values(by='count',ascending=False))
fromny.head(10)
(c66in[(c66in.hotel_market==365) & 
       (c66in.user_location_region==174) & 
       (c66in.user_location_city==24103)]
 ['orig_destination_distance'].describe())
tony = (train[(train.hotel_market == 675) & (train.user_location_country != 66)]
        .groupby(['user_location_country','user_location_region', 'user_location_city'])
        ['orig_destination_distance']
        .agg(['min','mean','max','count'])
        .sort_values(by='count',ascending=False))
tony.head(10)
fromny = (train[(train.hotel_country != 50) & 
                (train.user_location_country == 66) &
                (train.user_location_region == 348) &
                (train.user_location_city == 48862)]
        .groupby(['hotel_country','hotel_market'])
        ['orig_destination_distance']
        .agg(['min','mean','max','count'])
        .sort_values(by='count',ascending=False))
fromny.head(10)