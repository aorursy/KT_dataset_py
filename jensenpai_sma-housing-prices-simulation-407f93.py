%%time

# installing ffmpeg so that animation module can work

!apt-get -y install ffmpeg > /dev/null



import secrets  # python 3.6 necessary

import random

import numpy as np

import pandas as pd  # we try not to depend on pandas, to better translate later?

from copy import deepcopy

from IPython.display import display, HTML

from tqdm import tqdm_notebook

import matplotlib.animation as animation

import matplotlib.pyplot as plt  # for viz

import matplotlib.path as path  # for histogram

import matplotlib.patches as patches  # for histogram





pd.set_option('display.max_rows', 100)
# defining utility functions which forms the basis of housing valuation

def utility_general(house):

    '''

    Every person considers a house to have a certain utility.

    This is not based on personal perferences.

    '''

    utility_due_to_location = 2/(1 + (house["location"][0] - 5.3)**2 

                                   + (house["location"][1] - 5.3)**2)

    return utility_due_to_location + house["amenities"]["fengshui"]



def utility_function(person, house):

    '''

    A person considers each house to have a different utility.

    This assigns an additional utility of each house based on personal preferences.

    '''

    utility_due_to_person = 1/(1 + (house["location"][0] - person["idio"]["preferred_location"][0])**2 

                                 + (house["location"][1] - person["idio"]["preferred_location"][1])**2)

    return utility_general(house) + utility_due_to_person



### Weets' Vectorised Utility Functions (works with pd.Series) ###

# mere translation of above functions

# its quite hardcoded so not comfortable rofl



def utility_general_vectorised(house):

    '''

    Every person considers a house to have a certain utility.

    This is not based on personal perferences.

    '''

    utility_due_to_location = 2/(1 + (house["location"].apply(lambda tup: tup[0]) - 5.3)**2 

                                   + (house["location"].apply(lambda tup: tup[1]) - 5.3)**2)

    return utility_due_to_location + house["amenities"].apply(lambda amen_dt: amen_dt["fengshui"])



def utility_function_vectorised(person, house):

    '''

    A person considers each house to have a different utility.

    This assigns an additional utility of each house based on personal preferences.

    Input

        person: a dict or pandas df row

    '''

    # print(house["location"])

    # print(house["location"].apply(lambda tup: tup[0]))

    

    xloc = (house["location"].apply(lambda tup: tup[0]) - person["idio"]["preferred_location"][0])

    yloc = (house["location"].apply(lambda tup: tup[1]) - person["idio"]["preferred_location"][1])

    

    utility_due_to_person = 1/(1 + xloc**2 + yloc**2)

    return utility_general_vectorised(house) + utility_due_to_person
# params



# demographic

migration_base_age = 20

DEATH_RATE = 0.05 #deprecated? cant see where it was called

death_coef = 0.2

birth_n, birth_p = 10, 0.2



# geographic

city_x, city_y = 7, 9



# economic

income_mu, income_sigma = 10, 5

wealth_mu, wealth_sigma = 300, 10



# houses

initialization_price_mu, initialization_price_sigma = 300, 10

fengshui_mu, fengshui_sigma = 1, .1



# transactions

PROBA_SELL, PROBA_BUY  = 0.4, 0.8

# defining a template person and generate persons



def generate_person():

    person = {

        "age": migration_base_age,

        "income": np.random.normal(wealth_mu,wealth_sigma,1)[0],

        "wealth": np.random.normal(wealth_mu,wealth_sigma,1)[0],

        "house_staying": np.NaN,

        "house_selling": np.NaN,

        "utility": 0, # WEETS: utility here is person's 'score'. Every decision person makes must immediately result in increase of 0 or more, never decrease.

        "idio": {"preferred_location": (np.random.normal(city_x, 0.1, 1)[0],np.random.normal(city_y, 0.1, 1)[0])}

    }

    return person



persons = {}

for _ in range(10):

    persons[secrets.token_hex(4)] = generate_person()

persons = pd.DataFrame.from_dict(persons, orient='index')



persons['house_staying'] = persons['house_staying'].astype(object)

persons['house_selling'] = persons['house_selling'].astype(object)



persons.head()
# defining a template house and generate houses

houses = {}

for x in range(10):

    for y in range(10):

        houses[(x,y)] = {

            "location": (x,y),  # also the key 

            "last_bought_price": np.random.normal(initialization_price_mu, initialization_price_sigma, 1)[0],

            "status": "empty",  # "empty", "occupied", "selling" 

            "amenities": {"fengshui" : np.random.normal(fengshui_mu, fengshui_sigma, 1)[0]},

            "occupant": np.NaN,

            "last_updated": 0

        }

        houses[(x,y)]["market_price"] = houses[(x,y)]["last_bought_price"]



houses = pd.DataFrame.from_dict(houses, orient='index')



def status_to_float(status):

    if status == "empty": return 0 

    if status == "occupied": return 1 

    if status == "selling": return 2

    

houses.head()
def aging(verbose = False): # change this a function of age

    persons["age"] += 1

    persons["wealth"] += persons["income"]

    houses["last_updated"] += 1
def dying_prob_function(age):

    return 1./(1.+np.exp(-(death_coef*(age-50))))

plt.figure(figsize = (14,2))

plt.plot([dying_prob_function(age) for age in np.arange(100)])

plt.title("death probability over age")

plt.show()
def dying(verbose = False): # change this a function of age

    persons_id_dead = []

    for person_id in persons.index:

        if np.random.uniform() < dying_prob_function(persons.loc[person_id,"age"]):

            if verbose: print(person_id, " died")

            dead_person = persons.loc[person_id]

            if dead_person["house_staying"] != None:

                if verbose: print("vacated ", dead_person["house_staying"])

                houses.loc[dead_person["house_staying"],"status"] = "empty"

                houses.loc[dead_person["house_staying"],"occupant"] = None

                houses.loc[dead_person["house_selling"],"status"] = "empty"

                houses.loc[dead_person["house_selling"],"occupant"] = None

            persons_id_dead.append(person_id)

    persons.drop(persons_id_dead, inplace=True)
def birth(verbose = False):

    born = np.random.binomial(birth_n, birth_p)

    for _ in range(born):

        persons.loc[secrets.token_hex(4)] = generate_person()
persons
from collections import defaultdict

history = defaultdict(list)



def update_history(verbose = False):

    history["popn_with_zero_house"].append((persons.house_staying.values != persons.house_staying.values).sum())

    history["popn_with_one_house"].append((persons.house_selling.values != persons.house_selling.values).sum())

    history["popn_with_two_house"].append((persons.house_selling.values == persons.house_selling.values).sum())

    history["average_wealth"].append(np.mean(persons["wealth"]))

    history["total_houses_empty"].append((houses.status == "empty").sum())

    history["total_houses_occupied"].append((houses.status == "occupied").sum())

    history["total_houses_selling"].append((houses.status == "selling").sum())

    return None
### Phase 2: ASK -> BID -> MATCH -> UPDATE

# this is meant to be ran just once at the start

ask_df = pd.DataFrame(columns = ['location','occupant_id','amenities', 'ask_price']) # init empty ask_df with col

            

def gen_asks():

    ''' phase 2 bid-ask

    1. Refresh ask_df pd.DataFrame()

    2. Add empty houses from `houses` to ask_df

    3. Add more listings from persons who can and want to sell houses

    '''

    global ask_df # may not be necessary

    ask_df_columns = ask_df.columns.to_list() # ['house_pos','current_occupant_id','amenities', 'ask_price']

    

    # 1. Refresh ask_df pd.DataFrame()

    ask_df.drop(ask_df.index, inplace=True) # drops all rows

    

    # 2. Add empty houses from `houses` to ask_df

    empty_houses = houses[houses['status']=='empty']

    

    ## 2.1 Rename, reorder into ask_df column mold

    ## ask_df column order: ['house_pos','current_occupant_id','amenities', 'ask_price']

    empty_houses_listing = empty_houses.rename(columns={

        'occupant':'occupant_id',

        'last_bought_price':'ask_price',

    })

    empty_houses_listing = empty_houses_listing[ask_df_columns] # reorder

    

    ask_df = ask_df.append(empty_houses_listing, ignore_index=True) # TODO: optimise

    

    # 3. Add more listings from `persons` who can and want to sell houses

    ## 3.1 get sub df of persons who have a second house to sell

    COND_have_house_selling = persons['house_selling'] != None

    potential_sellers = persons[COND_have_house_selling] # a persons sub df

    

    ## 3.2 Get sellable houses that have market price >= cost price

    potential_house_selling_loc = potential_sellers['house_selling']

    potential_house_selling = houses[houses['location'].isin(potential_house_selling_loc.values)]

    COND_market_greater_or_equal_cost_price = potential_house_selling['market_price'] >= potential_house_selling['last_bought_price'] 

    no_loss_house_selling = potential_house_selling[COND_market_greater_or_equal_cost_price] # a houses subdf

    

    ## 3.3 Random decide if want to sell or not

    # arbitrary threshold; TODO: turn into adjustable param

    COND_want_sell = no_loss_house_selling['status'].apply(lambda runif: np.random.uniform()) <= PROBA_SELL

    want_sell_houses = no_loss_house_selling[COND_want_sell]

    want_sell_houses_loc = want_sell_houses['location']

    actual_house_selling = potential_house_selling[potential_house_selling['location'].isin(want_sell_houses_loc.values)]

    

    ## 3.4 Rename, reorder actual_house_selling into ask_df column mold

    ## ask_df column order: ['house_pos','current_occupant_id','amenities', 'ask_price']

    main_listing = actual_house_selling.rename(columns={'market_price':'ask_price',

                                               'occupant':'occupant_id'})

    main_listing = main_listing[ask_df_columns]

    

    ask_df = ask_df.append(main_listing, ignore_index=True)

    

    # strangely, there's a row with nan value in location appearing

    # this chunk fixes that

    if any(ask_df['location'].apply(lambda loc: type(loc)!=tuple)):

        # print('Missing location in ask_df, applying fix')

        ori_len = len(ask_df)

        ask_df = ask_df[~ask_df['location'].isna()]

        # print('Change in len', len(ask_df)-ori_len)

    

    

# test run

# gen_asks()

# ask_df.sample(10)
### Phase 2: ASK -> BID -> MATCH -> UPDATE

# init empty ask_df with col

bid_df = pd.DataFrame(columns = ['location', 'bidder_id', 'utility_to_buyer', 'max_bid_price', 'bid_price'])

    

def gen_bids():

    ''' phase 2 bid-ask

    1. Refresh bid_df pd.DataFrame()

    2. Generate subdf of persons who can and want to buy houses

    3. For each eligible person, iterate over ask, grow person_bids list of dict

    4. Merge 

    '''

    global bid_df # may not be necessary

    bid_df_columns = bid_df.columns.to_list() # ['location', 'bidder_id', 'utility', 'bid_price']

    

    # 1. Refresh bid_df pd.DataFrame()

    bid_df.drop(bid_df.index, inplace=True) # drops all rows

    

    # 2. Screen viable bidders

    ## 2.1 Does not own a second house (can have 1 or 0 houses)

    COND_only_one_house = persons['house_selling'].isna() # NOTE: do not use `persons['house_selling'] == None` to check

    potential_buyers = persons[COND_only_one_house]

    

    ## 2.2 Random decide if want to seek or not

     # arbitrary threshold; TODO: turn into adjustable param

    COND_want_buy = potential_buyers['age'].apply(lambda runif: np.random.uniform()) <= PROBA_BUY

    eligible_and_seeking_buyers = potential_buyers[COND_want_buy] # these are the eligible people who want to buy houses

    

    # 3. Each eligible buyer makes a bid for each house on sale

    list_of_bid_sets = [] # to be populated with df corr. to each person's bids

    

    ## 3.1 Define helper fn

    def _gen_bid_price(listing_row):

        max_bid_price = listing_row['max_bid_price']

        ask_price = listing_row['ask_price']

        if  max_bid_price >= ask_price:

            surplus = max_bid_price - ask_price

            return ask_price + np.random.uniform() * surplus

        else:

            return max_bid_price

    

    ## 3.2 Iterate over buyers

    for idx, buyer in eligible_and_seeking_buyers.iterrows():

        buyer_view_of_ask_df = ask_df.copy()

        

        ###  3.2.1 Calculate each listing's utility to buyer

        buyer_view_of_ask_df['bidder_id'] = idx

        buyer_view_of_ask_df['utility_to_buyer'] = utility_function_vectorised(buyer, buyer_view_of_ask_df)

        # NOTE: utility_to_buyer is partial -- it only consider's a houses's general and locational utility and buyer idio

        

        ### 3.2.2 Calculate bid_price

        buyer_view_of_ask_df['max_bid_price'] = buyer['wealth'] - buyer['utility'] + buyer_view_of_ask_df['utility_to_buyer'] # TODO: double check if this is a good rule

        # NOTE: WEETS suspects above formula may be wrong since it does not compare the differential in utility DUE TO HOUSE only

        # test: what if utility to buyer is negative? Would you still bid (and spend money)?

        buyer_view_of_ask_df['max_bid_price'] = buyer_view_of_ask_df['max_bid_price'].apply(lambda mbp: min(mbp, buyer['wealth']))

        # mbp must be capped at buyer's wealth

        buyer_view_of_ask_df['max_bid_price'] = buyer_view_of_ask_df['max_bid_price'].apply(lambda mbp: max(0,mbp))

        # mbp must be non-negative

        

        bid_price = buyer_view_of_ask_df.apply(_gen_bid_price, axis=1)

        buyer_view_of_ask_df['bid_price'] = bid_price

        

        ### 3.2.3 Append specific columns of buyer_view_of_ask_df to list_of_bid_sets

        select_columns = ['location', 'bidder_id', 'utility_to_buyer', 'max_bid_price', 'bid_price']

        list_of_bid_sets.append(buyer_view_of_ask_df[select_columns])

    

    # 4. Concatenate list of dataframes into one dataframe

    if list_of_bid_sets: # possible that no bids take place

        bid_df = pd.concat(list_of_bid_sets)

    return bid_df



# bid_df = gen_bids()

# # print(bid_df['bidder_id'].nunique())

# bid_df.head()
### Phase 2: ASK -> BID -> MATCH -> UPDATE



def match_ask_bid():

    '''

    1. Create a container list to store dicts of info relating to bidding for each listing

    2. Iterate over listings in ask_df, find best bid - is successful match

    3. For each successful match

        1. Create and append dict of info relating to bids for the listing

        2. Remove all bids for same listing

        3. Remove all other bids by same bidder

        4. Update asker and bidder

    4. For each unsuccessful match

        1. Create and append dict of info relating to bids for the listing

        2. Remove all bids for same listing

    5. Make match_df

    '''

    global bid_df, persons, houses

    # 1. Create a container list to store dicts of info relating to bidding for each listing

    list_of_matches = [] # contains info on winning bid

    

    # 2. Iterate over listings in ask_df, find best bid - is successful match

    for idx, listing in ask_df.iterrows():

        match_info_dict = {} # stats for each listing match

        

        ## 2.1 Get general data

        listing_loc = listing['location']

        match_info_dict['location'] = listing_loc

        

        match_info_dict['ask_price'] = listing['ask_price']

        

        relevant_bids = bid_df[bid_df['location']==listing_loc]

        match_info_dict['num_bids'] = len(relevant_bids) # expect 0 or more

        

        highest_bid_value = relevant_bids['bid_price'].max()

        match_info_dict['highest_bid_value'] = highest_bid_value

        

        match_info_dict['mean_bid_value'] = relevant_bids['bid_price'].mean()

        

        # 3. Found winning bid(s)

        if highest_bid_value >= listing['ask_price']: # there exists a successful match

            ## 3.1 Create and append dict of info relating to bids for the listing

            ### 3.1.1 Check for ties among highest bid

            highest_bids = relevant_bids[relevant_bids['bid_price']==highest_bid_value]

            num_highest_bid = len(highest_bids) # expect at least 1, rarely more

            assert num_highest_bid >= 1, 'ERR: num_highest_bid must be >= 1'

            

            ### 3.1.2 Get the winner

            winning_bid = highest_bids.sample(1) # tie-breaker: randomly choose one highest bidder to win

            

            winning_bidder_id = winning_bid['bidder_id'].iloc[0]

            match_info_dict['winning_bidder_id'] = winning_bidder_id

            match_info_dict['winning_bid_value'] = highest_bid_value # obviously; stated explicitly as highest_bid_value may not win for the `else` case

            

            ### 3.1.3 Append match info

            list_of_matches.append(match_info_dict)

            

            ## 3.2 Remove all corresponding bids, 3.3 Remove all other bids by same bidder

            bid_df = bid_df.drop(relevant_bids.index, axis=0)

            bid_df = bid_df[~(bid_df['bidder_id']==winning_bidder_id)]

            

            ## 3.4 Update asker and bidder

            asker_id = listing['occupant_id']

            

            ### 3.4.1 Update asker

            if type(asker_id) is str: # if str, then not empty house

                persons['wealth'].loc[asker_id] += highest_bid_value

                persons['house_selling'].iloc[asker_id] = np.NaN # potential problem here?

                # TODO: check where to update 'utility' (person's simulation score) -- here or elsewhere?

                # ENSURE: asker['utility'] increase or stay the same

                

            ### 3.4.2 Update bidder

            winning_bidder = persons.loc[winning_bidder_id]

            persons['wealth'].loc[winning_bidder_id] -= highest_bid_value

            

            #### Additional updates for bidder if second house buyer

            if type(winning_bidder['house_staying']) is tuple: # first house exists, buyer is buying second house

                persons['house_selling'].loc[winning_bidder_id] = winning_bidder['house_staying'] # set current house_staying to be house_selling

                houses['status'].loc[winning_bidder['house_staying']] = 'selling' # set that same current house to 'selling' status

            persons['house_staying'].loc[winning_bidder_id] = listing_loc

            # TODO: check where to update 'utility' (person's simulation score) -- here or elsewhere?

            # ENSURE: asker['utility'] increase or stay the same

            

            ### 3.4.3 Update house

            houses['last_bought_price'].loc[listing_loc] = highest_bid_value

            houses['status'].loc[listing_loc] = 'occupied'

            # Note: for second house buyers, their first house's status has already been updated

            houses['occupant'].loc[listing_loc] = winning_bidder_id

            houses['last_updated'].loc[listing_loc] = 0

            # TODO: update houses['market_price'] at the end of each time step, somewhere else perhaps

            

        # 4. No successful match   

        else:

            ## 4.1 Create and append dict of info relating to bids for the listing

            match_info_dict['winning_bidder_id'] = np.NaN

            match_info_dict['winning_bid_value'] = np.NaN

            list_of_matches.append(match_info_dict)

            

            ## 4.2 Remove all bids for same listing

            bid_df = bid_df.drop(relevant_bids.index, axis=0)

            

    # 5. Make match_df

    match_df = pd.DataFrame(list_of_matches)

    return match_df



# gen_asks()

# bid_df = gen_bids()

# match_df = match_ask_bid() # Note: changes bid_df each time it is called

# match_df.head(10)
persons
houses
%%time

def simulate():

    aging()

    birth()

    dying()

    gen_asks()

    gen_bids()

    match_ask_bid()

for _ in tqdm_notebook(range(10)):

    simulate()
%%time

fig, ax = plt.subplots(2,5,figsize=(12,7))

plt.subplots_adjust(wspace=0.4)

im0 = ax[0,0].imshow(np.random.randn(10,10), vmin=0, vmax=2)

im1 = ax[0,1].imshow(np.random.randn(10,10), vmin=0, vmax=400)

im2 = ax[0,2].imshow(np.random.randn(10,10), vmin=0, vmax=400)

im3 = ax[0,3].imshow(np.random.randn(10,10), vmin=0, vmax=100)

im4 = ax[0,4].imshow(np.random.randn(10,10), vmin=0, vmax=2)

ax[0,0].set_title("utility_general")

ax[0,1].set_title("market_price")

ax[0,2].set_title("last_bought_price")

ax[0,3].set_title("last_updated")

ax[0,4].set_title("occupancy status")



line_pop_0, = ax[1,0].plot([], lw=3)

line_pop_1, = ax[1,0].plot([], lw=3)

line_pop_2, = ax[1,0].plot([], lw=3)

ax[1,0].set_ylim((0, 60))

ax[1,0].set_xlim((-20, 0))

ax[1,0].set_title("population")



line_wealth_0, = ax[1,1].plot([], lw=3)

ax[1,1].set_ylim((0, 600))

ax[1,1].set_xlim((-20, 0))

ax[1,1].set_title("average wealth")



scat_income_age = ax[1,2].scatter([], [], s=20)

ax[1,2].set_ylim((0, 600))

ax[1,2].set_xlim((20, 60))

ax[1,2].set_title("wealth against age")



# animating histogram is quite troublesome

# https://matplotlib.org/gallery/animation/animated_histogram.html

bins_bdrs = np.arange(0,600,50)  # defining boundaries

n, bins = np.histogram(np.random.randn(1000), bins=bins_bdrs)

left, right = np.array(bins[:-1]), np.array(bins[1:])

top, bottom = np.zeros(len(left)), np.zeros(len(left))

nrects = len(left)  # defining rectangles

nverts = nrects * (1 + 3 + 1)

verts = np.zeros((nverts, 2))

codes = np.ones(nverts, int) * path.Path.LINETO

codes[0::5], codes[4::5] = path.Path.MOVETO, path.Path.CLOSEPOLY

verts[0::5, 0], verts[0::5, 1] = left, bottom

verts[1::5, 0], verts[1::5, 1] = left, top

verts[2::5, 0], verts[2::5, 1] = right, top

verts[3::5, 0], verts[3::5, 1] = right, bottom

barpath = path.Path(verts, codes)  # defining graphs

hist_patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)

ax[1,3].add_patch(hist_patch)

ax[1,3].set_ylim((0, 20))

ax[1,3].set_xlim((0, 600))

ax[1,3].set_title("histogram of wealth")



line_house_0, = ax[1,4].plot([], lw=3)

line_house_1, = ax[1,4].plot([], lw=3)

line_house_2, = ax[1,4].plot([], lw=3)

ax[1,4].set_ylim((0, 100))

ax[1,4].set_xlim((-20, 0))

ax[1,4].set_title("housing occupancy")



patch = [im0, im1, im2, im3, im4, hist_patch]



a1 = np.random.randn(10,10)

a2 = np.random.randn(10,10)

a3 = np.random.randn(10,10)

a4 = np.random.randn(10,10)

a5 = np.random.randn(10,10)



def update_plot():

    for x in range(10):

        for y in range(10):

            a1[x,y] = utility_general(houses.loc[(x,y)])

            a2[x,y] = houses.loc[(x,y),"market_price"]

            a3[x,y] = houses.loc[(x,y),"last_bought_price"]

            a4[x,y] = houses.loc[(x,y),"last_updated"]

            a5[x,y] = status_to_float(houses.loc[(x,y),"status"])

#     for index in list(houses.index):

#         a1[index[0], index[1]] = utility_general(houses.loc[(x,y)])

#         a2[index[0], index[1]] = houses.loc[(x,y),"market_price"]

#         a3[index[0], index[1]] = houses.loc[(x,y),"last_bought_price"]

#         a4[index[0], index[1]] = houses.loc[(x,y),"last_updated"]

#         a5[index[0], index[1]] = status_to_float(houses.loc[(x,y),"status"])

    im0.set_data(a1)

    im1.set_data(a2)

    im2.set_data(a3)

    im3.set_data(a4)

    im4.set_data(a5)

    line_pop_0.set_data(range(0,-len(history["popn_with_zero_house"][-20:]),-1), 

                        history["popn_with_zero_house"][-20:][::-1])

    line_pop_1.set_data(range(0,-len(history["popn_with_one_house"][-20:]),-1), 

                        history["popn_with_one_house"][-20:][::-1])

    line_pop_2.set_data(range(0,-len(history["popn_with_two_house"][-20:]),-1), 

                        history["popn_with_two_house"][-20:][::-1])

    

    line_house_0.set_data(range(0,-len(history["total_houses_empty"][-20:]),-1), 

                          history["total_houses_empty"][-20:][::-1])

    line_house_1.set_data(range(0,-len(history["total_houses_occupied"][-20:]),-1), 

                          history["total_houses_occupied"][-20:][::-1])

    line_house_2.set_data(range(0,-len(history["total_houses_selling"][-20:]),-1), 

                          history["total_houses_selling"][-20:][::-1])

    

    line_wealth_0.set_data(range(0,-len(history["average_wealth"][-20:]),-1), 

                           history["average_wealth"][-20:][::-1])

    scat_income_age.set_offsets(np.transpose((persons["age"], persons["wealth"])))

    

    # histogram

    height, bins = np.histogram(persons["wealth"], bins=bins_bdrs)

    verts[1::5, 1], verts[2::5, 1] = height, height



def init():

    return patch



def animate(i):

    simulate()

    update_history()

    update_plot()

    return patch



# call the animator. blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=1000, interval=100, blit=True)



vid = anim.to_html5_video()

plt.close()
HTML(vid)
persons
houses
# somehow NaN is introduced to the dataframe index

# this converts the index to a numpy float which I need to reconvert back

if np.NaN in houses.index:

    houses = houses.drop([np.NaN])



def convert_to_tuple_int(index):

    return (int(index[0]), int(index[1]))

houses.index = houses.index.map(convert_to_tuple_int)
houses
import plotly.figure_factory as ff

# reference https://plot.ly/python/annotated_heatmap/#custom-hovertext



z = np.empty([10, 10])

hover = [[None for _ in range(10)] for _ in range(10)]

ann = [["" for _ in range(10)] for _ in range(10)]



for addr in houses.index:

    z[addr] = status_to_float(houses.loc[addr,"status"])

    displayed = "<br>".join(["{}: {}".format(k,v) for k,v in houses.loc[addr].to_dict().items()])

    occupant = houses.loc[addr,"occupant"]

    if occupant:

        try:

            displayed += "<br><br>Tenant information<br>"

            displayed += "<br>".join(["{}: {}".format(k,v) for k,v in persons.loc[occupant].to_dict().items()])

        except:

            pass

#             print(occupant, " not found")

    hover[addr[0]][addr[1]] = displayed



colorscale=[[0.0, 'rgb(255,255,255)'],[0.2, 'rgb(255, 255, 153)'],

            [0.4, 'rgb(153,255,204)'],[0.6, 'rgb(179, 217, 255)'],

            [0.8, 'rgb(240,179,255)'],[1.0, 'rgb(255,  77, 148)']]



# Make Annotated Heatmap

fig = ff.create_annotated_heatmap(z, text=hover, hoverinfo='text', annotation_text=ann,

                                  colorscale=colorscale, font_colors=['black'])

fig.update_layout(title_text='Occupancy Status')

fig.show()