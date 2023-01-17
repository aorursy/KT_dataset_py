%%time



import numpy as np

import pandas as pd



from IPython.display import display

import matplotlib.pyplot as plt

%matplotlib inline



input_dir = "../input/"
%%time



auctions = pd.read_csv(input_dir + 'auction.csv')

display(auctions.head(5))
%%time



count_type_item = pd.get_dummies(auctions.drop_duplicates(subset='auctionid'), columns=['item'])

count_type_item = count_type_item[['openbid', 'price', 'auction_type', 'item_Cartier wristwatch', 'item_Palm Pilot M515 PDA', 'item_Xbox game console']].groupby(by=['auction_type'])[['item_Cartier wristwatch', 'item_Palm Pilot M515 PDA', 'item_Xbox game console']].sum()



print(count_type_item.sum(axis=0))

display(count_type_item)

count_type_item.plot.bar()
%%time



auctions['bidderrate'].fillna(0, inplace=True)

print(auctions.isnull().sum())
%%time



pda = auctions[auctions.item == 'Palm Pilot M515 PDA'].drop(['bidder', 'item'], axis = 1)



pda_first = pda[['auctionid', 'openbid', 'price', 'auction_type']].drop_duplicates(subset='auctionid')



grouped = pda.groupby('auctionid')

pda_count = grouped.price.agg(['count']).reset_index()

pda_bid = grouped['bid'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bid_min", "amax": "bid_max", "mean": "bid_mean",})

pda_bidtime = grouped['bidtime'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bidtime_min", "amax": "bidtime_max", "mean": "bidtime_mean",})

pda_bidderrate = grouped['bidderrate'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bidderrate_min", "amax": "bidderrate_max", "mean": "bidderrate_mean",})



pda_all = pda_count.merge(pda_first, how='left', on='auctionid')

pda_all = pda_all.merge(pda_bid, how='left', on='auctionid')

pda_all = pda_all.merge(pda_bidtime, how='left', on='auctionid')

pda_all = pda_all.merge(pda_bidderrate, how='left', on='auctionid')



del pda, pda_first, pda_count, pda_bid, pda_bidtime, pda_bidderrate

display(pda_all.head(5))
%%time



pda_type = pda_all.groupby('auction_type')[['count', 'openbid', 'price', 'bidtime_max', 'bidderrate_mean']].agg(np.mean).reset_index()

display(pda_type)



auction_types = ['3 day auction', '5 day auction', '7 day auction']

for auction_type in auction_types:

    print('Number of', auction_type, 'is', pda_all[pda_all.auction_type == auction_type].shape[0])



y_index = ['price', 'count', 'bidtime_max', 'bidderrate_mean']

for y in y_index:

    ax = pda_all[pda_all.auction_type == '3 day auction'].plot.scatter(x='openbid', y=y, color='DarkBlue', label='3 day auction')

    pda_all[pda_all.auction_type == '5 day auction'].plot.scatter(x='openbid', y=y, color='DarkGreen', label='5 day auction', ax=ax)

    pda_all[pda_all.auction_type == '7 day auction'].plot.scatter(x='openbid', y=y, color='Red', label='7 day auction', ax=ax)

    if y == 'price':

        ax.set_ylim(0, 500)
%%time



watch = auctions[auctions.item == 'Cartier wristwatch'].drop(['bidder', 'item'], axis = 1)



watch_first = watch[['auctionid', 'openbid', 'price', 'auction_type']].drop_duplicates(subset='auctionid')



grouped = watch.groupby('auctionid')

watch_count = grouped.price.agg(['count']).reset_index()

watch_bid = grouped['bid'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bid_min", "amax": "bid_max", "mean": "bid_mean",})

watch_bidtime = grouped['bidtime'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bidtime_min", "amax": "bidtime_max", "mean": "bidtime_mean",})

watch_bidderrate = grouped['bidderrate'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bidderrate_min", "amax": "bidderrate_max", "mean": "bidderrate_mean",})



watch_all = watch_count.merge(watch_first, how='left', on='auctionid')

watch_all = watch_all.merge(watch_bid, how='left', on='auctionid')

watch_all = watch_all.merge(watch_bidtime, how='left', on='auctionid')

watch_all = watch_all.merge(watch_bidderrate, how='left', on='auctionid')



del watch, watch_first, watch_count, watch_bid, watch_bidtime, watch_bidderrate

display(watch_all.head(5))
%%time



watch_type = watch_all.groupby('auction_type')[['count', 'openbid', 'price', 'bidtime_max', 'bidderrate_mean']].agg(np.mean).reset_index()

display(watch_type.head(10))



auction_types = ['3 day auction', '5 day auction', '7 day auction']

for auction_type in auction_types:

    print('Number of', auction_type, 'is', watch_all[watch_all.auction_type == auction_type].shape[0])



y_index = ['price', 'count', 'bidtime_max', 'bidderrate_mean']

for y in y_index:

    ax = watch_all[watch_all.auction_type == '3 day auction'].plot.scatter(x='openbid', y=y, color='DarkBlue', label='3 day auction')

    watch_all[watch_all.auction_type == '5 day auction'].plot.scatter(x='openbid', y=y, color='DarkGreen', label='5 day auction', ax=ax)

    watch_all[watch_all.auction_type == '7 day auction'].plot.scatter(x='openbid', y=y, color='Red', label='7 day auction', ax=ax)
%%time



xbox = auctions[auctions.item == 'Xbox game console'].drop(['bidder', 'item'], axis = 1)

display(xbox.head(5))



xbox_first = xbox[['auctionid', 'openbid', 'price', 'auction_type']].drop_duplicates(subset='auctionid')



grouped = xbox.groupby('auctionid')

xbox_count = grouped.price.agg(['count']).reset_index()

xbox_bid = grouped['bid'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bid_min", "amax": "bid_max", "mean": "bid_mean",})

xbox_bidtime = grouped['bidtime'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bidtime_min", "amax": "bidtime_max", "mean": "bidtime_mean",})

xbox_bidderrate = grouped['bidderrate'].agg([np.min, np.max, np.mean]).reset_index().rename(columns={"amin": "bidderrate_min", "amax": "bidderrate_max", "mean": "bidderrate_mean",})



xbox_all = xbox_count.merge(xbox_first, how='left', on='auctionid')

xbox_all = xbox_all.merge(xbox_bid, how='left', on='auctionid')

xbox_all = xbox_all.merge(xbox_bidtime, how='left', on='auctionid')

xbox_all = xbox_all.merge(xbox_bidderrate, how='left', on='auctionid')



del xbox, xbox_first, xbox_count, xbox_bid, xbox_bidtime, xbox_bidderrate

display(xbox_all.head(5))
%%time



xbox_type = xbox_all.groupby('auction_type')[['count', 'openbid', 'price', 'bidtime_max', 'bidderrate_mean']].agg(np.mean).reset_index()



display(xbox_type.head(10))
%%time



auction_types = ['3 day auction', '5 day auction', '7 day auction']

for auction_type in auction_types:

    print('Number of', auction_type, 'is', xbox_all[xbox_all.auction_type == auction_type].shape[0])



y_index = ['price', 'count', 'bidtime_max', 'bidderrate_mean']

for y in y_index:

    ax = xbox_all[xbox_all.auction_type == '3 day auction'].plot.scatter(x='openbid', y=y, color='DarkBlue', label='3 day auction')

    xbox_all[xbox_all.auction_type == '5 day auction'].plot.scatter(x='openbid', y=y, color='DarkGreen', label='5 day auction', ax=ax)

    xbox_all[xbox_all.auction_type == '7 day auction'].plot.scatter(x='openbid', y=y, color='Red', label='7 day auction', ax=ax)