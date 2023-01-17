%reset -sf
%%time

# to make animation module work

!apt-get -y install ffmpeg > /dev/null



import secrets  # python 3.6 necessary

import random

import numpy as np

import pandas as pd  # we try not to depend on pandas, to better translate later?

from copy import deepcopy

from IPython.display import display, HTML

from tqdm import tqdm_notebook

import matplotlib

matplotlib.use('Agg')

import matplotlib.animation as animation

import matplotlib.colors as colors

import matplotlib.pyplot as plt  # for viz

import matplotlib.path as path  # for histogram

import matplotlib.patches as patches  # for histogram

from sklearn.linear_model import LinearRegression

def displayer(df): display(HTML(df.head(2).to_html()))



pd.set_option('display.max_rows', 100)

pd.options.mode.chained_assignment = None
# city coefficients

CITY_X, CITY_Y = 5.3,5.3

AMENITIES_COEF = 500

LOC_COEF = 500

INITIAL_PRICE = lambda: 1

INTIAL_AMENITIES = lambda: np.random.uniform() 



# individual perferences

PREFERRED_LOCATION_X = lambda: 10*np.random.uniform()

PREFERRED_LOCATION_Y = lambda: 10*np.random.uniform()



# initialisation variables for people

INITIAL_AGE = lambda: 20

INITIAL_WEALTH = lambda: 100 + 100*np.random.uniform()

INCOME = lambda: 10

STARTING_POPULATION = 10



# random variable of number of people born per timeframe

NUM_BORN = lambda: np.random.binomial(10, 0.2)



# death probability function

DYING_PROB_FUNCTION = lambda age: 1./(1.+np.exp(-(0.2*(age-50))))



# individual preferences

IDIO_COEF = 0.2



# market transaction model variables

PROBA_BUY = 0.8

PROBA_SELL_NO_LOSS = 0.8 # update(weets, 191128)

PROBA_SELL_WITH_LOSS = 0.4 # update(weets, 191128)

MIN_TXN_VOL = 5 # update(weets, 191202)



# current affairs # update(weets,191202)

PROBA_CA_GOOD = 0.025

PROBA_CA_BAD = 0.1

FAIR_UB = 1.05

FAIR_LB = 0.95

CA_MULTIPLIER_FAIR = lambda: np.random.uniform(FAIR_LB,FAIR_UB)

CA_MULTIPLIER_GOOD = lambda: np.random.uniform(FAIR_UB,1.20)

CA_MULTIPLIER_BAD = lambda: np.random.uniform(0.75,FAIR_LB)



# plotting variables

NUM_FRAMES = 2000

MILLISECS_PER_FRAME = 50
# defining utility functions which forms the basis of housing valuation

def utility_general(house):

    '''

    Every person considers a house to have a certain utility.

    This is not based on personal perferences.

    '''

    utility_due_to_location = 2/(1 + (house["location"][0] - CITY_X)**2 

                                   + (house["location"][1] - CITY_Y)**2)



    return utility_due_to_location + house["amenities"] # UPDATE(weets, 191125)



def utility_function(person, house):

    '''

    A person considers each house to have a different utility.

    This assigns an additional utility of each house based on personal preferences.

    '''

    

    # UPDATE(weets, 191125) - made preferred_location a column var instead of nesting in idio dict

    utility_due_to_person = 1/(1 + (house["location"][0] - person["preferred_location"][0])**2 

                                 + (house["location"][1] - person["preferred_location"][1])**2)

    return utility_general(house) + utility_due_to_person



### Weets' Vectorised Utility Functions (works with pd.Series) ###

# mere translation of above functions

# its quite hardcoded so not comfortable rofl



def utility_general_vectorised(house):

    '''

    Every person considers a house to have a certain utility.

    This is not based on personal perferences.

    '''

    utility_due_to_location = 2/(1 + (house["location"].apply(lambda tup: tup[0]) - CITY_X)**2 

                                   + (house["location"].apply(lambda tup: tup[1]) - CITY_Y)**2)

    

    global AMENITIES_COEF

    AMENITIES_COEF = AMENITIES_COEF

    global LOC_COEF

    LOC_COEF = LOC_COEF # UPDATE(weets, 191126)

    return LOC_COEF*utility_due_to_location + AMENITIES_COEF*house["amenities"] # UPDATE(weets, 191125)



def utility_function_vectorised(person, house):

    '''

    A person considers each house to have a different utility.

    This assigns an additional utility of each house based on personal preferences.

    Input

        person: a dict or pandas df row

    '''

    

    # UPDATE(weets, 191125) - made preferred_location_? a column var instead of nesting in idio dict

    xloc = (house["location"].apply(lambda tup: tup[0]) - person["preferred_location_x"])

    yloc = (house["location"].apply(lambda tup: tup[1]) - person["preferred_location_y"])

    

    utility_due_to_person = 1/(1 + xloc**2 + yloc**2)

    global IDIO_COEF

    IDIO_COEF = IDIO_COEF # UPDATE(weets, 191126)

    return utility_general_vectorised(house) + IDIO_COEF * utility_due_to_person
# defining a template person and generate persons

def generate_person():

    person = {

        "age": INITIAL_AGE(),

        "income": INCOME(),

        "wealth": INITIAL_WEALTH(),

        "house_staying": np.NaN,

        "house_selling": np.NaN,

        "utility": 0, # WEETS: utility here is the utility of the current staying house to the person. It will be swapped for the utility of the new house if this person buys new house.

        # the true 'score' of a person is wealth + utility

        'preferred_location_x': PREFERRED_LOCATION_X(), # UPDATE(weets, 191125)

        'preferred_location_y': PREFERRED_LOCATION_Y() # UPDATE(weets, 191125)

    }

    return person



persons = None



def initialise_persons():

    global STARTING_POPULATION, persons

    persons = {}

    for _ in range(STARTING_POPULATION):

        persons[secrets.token_hex(4)] = generate_person()

    persons = pd.DataFrame.from_dict(persons, orient='index')



initialise_persons()



persons['house_staying'] = persons['house_staying'].astype(object)

persons['house_selling'] = persons['house_selling'].astype(object)



persons.head()
# defining a template house and generate houses

houses = None

def initialise_houses():

    global houses

    houses = {}

    for x in range(10):

        for y in range(10):

            houses[(x,y)] = {

                "location": (x,y),  # also the key 

                "last_bought_price": INITIAL_PRICE(),

                "status": "empty",  # "empty", "occupied", "selling" 

                'amenities': INTIAL_AMENITIES(), # UPDATE(weets, 191125)

                "occupant": np.NaN,

                "last_updated": 0,

                "distance_to_city":((x-CITY_X)**2+(y-CITY_Y)**2)**(0.5) # update(weets,191202) -- euclidean dist

            }

            houses[(x,y)]["market_price"] = houses[(x,y)]["last_bought_price"]



    houses = pd.DataFrame.from_dict(houses, orient='index')

    houses["utility_general"] = utility_general_vectorised(houses)



def status_to_float(status):  # for visualisation

    if status == "empty": return 0 

    if status == "occupied": return 1 

    if status == "selling": return 2

    

initialise_houses()

houses.head()
def aging(verbose = False): # change this a function of age

    persons["age"] += 1

    persons["wealth"] += persons["income"]

    houses["last_updated"] += 1
plt.figure(figsize = (14,2))

plt.plot([DYING_PROB_FUNCTION(age) for age in np.arange(100)])

plt.title("death probability over age")

plt.show()
def dying(verbose = False): # change this a function of age

    persons_id_dead = []

    for person_id in persons.index:

        if np.random.uniform() < DYING_PROB_FUNCTION(persons.loc[person_id,"age"]):

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

    born = NUM_BORN()

    for _ in range(born):

        persons.loc[secrets.token_hex(4)] = generate_person()
from collections import defaultdict

history = defaultdict(list)



def update_history(verbose = False):

    history["popn_with_zero_house"].append((persons.house_staying.values != persons.house_staying.values).sum())

    history["popn_with_one_house"].append((persons.house_selling.values != persons.house_selling.values).sum())

    history["popn_with_two_house"].append((persons.house_selling.values == persons.house_selling.values).sum())

    history["total_houses_empty"].append((houses.status == "empty").sum())

    history["total_houses_occupied"].append((houses.status == "occupied").sum())

    history["total_houses_selling"].append((houses.status == "selling").sum())

    history["average_wealth"].append(np.mean(persons["wealth"]))

    history["average_utility"].append(np.mean(persons["utility"]))

    history["average_price"].append(np.mean(houses["last_bought_price"]))

    history["average_market"].append(np.mean(houses["market_price"]))

    return None
### Phase 2: ASK -> BID -> MATCH -> UPDATE

# this is meant to be ran just once at the start

ask_df = pd.DataFrame(columns = ['location','occupant_id','amenities', 'distance_to_city', 'ask_price']) # init empty ask_df with col

            

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

    empty_houses_listing = empty_houses_listing[ask_df_columns] # reorder and subset columns

    

    ask_df = ask_df.append(empty_houses_listing, ignore_index=True) # TODO: optimise

    

    # 3. Add more listings from `persons` who can and want to sell houses

    ## 3.1 get sub df of persons who have a second house to sell

    COND_have_house_selling = persons['house_selling'] != None

    potential_sellers = persons[COND_have_house_selling] # a persons sub df

    

    ## 3.2 Get potential sellable houses

    potential_house_selling_loc = potential_sellers['house_selling']

    potential_house_selling = houses[houses['location'].isin(potential_house_selling_loc.values)]

    

    ## 3.3 Random decide if want to sell or not given market_price vs last_bought_price

    global PROBA_SELL_NO_LOSS, PROBA_SELL_WITH_LOSS # update(weets,191128)

    

    ### 3.3.1 Build conditionals to identify houses poorly affected by market but want to sell anyway

    COND_poor_market = potential_house_selling['market_price'] < potential_house_selling['last_bought_price'] # expect loss

    COND_want_sell_with_loss = potential_house_selling['status'].apply(lambda runif: np.random.uniform()) <= PROBA_SELL_WITH_LOSS # lower proba of selling

    

    ### 3.3.2 Build conditionals to identify houses well affected by market and want to sell anyway

    COND_good_market = potential_house_selling['market_price'] >= potential_house_selling['last_bought_price'] # no loss

    COND_want_sell_no_loss = potential_house_selling['status'].apply(lambda runif: np.random.uniform()) <= PROBA_SELL_NO_LOSS # higher proba of selling

    

    ### 3.3.3 Get subdf of actual houses to be listed

    actual_house_selling = potential_house_selling[(COND_poor_market & COND_want_sell_with_loss) | (COND_good_market & COND_want_sell_no_loss)]

    

    ## 3.4 Rename, reorder actual_house_selling into ask_df column mold

    ## ask_df column order: ['house_pos','current_occupant_id','amenities', 'ask_price']

    main_listing = actual_house_selling.rename(columns={'market_price':'ask_price',

                                               'occupant':'occupant_id'})

    main_listing = main_listing[ask_df_columns]

    

    ask_df = ask_df.append(main_listing, ignore_index=True)

    

    # strangely, there's a row with nan value in location appearing

    # this chunk fixes that

    if any(ask_df['location'].apply(lambda loc: type(loc)!=tuple)):

        ori_len = len(ask_df)

        ask_df = ask_df[~ask_df['location'].isna()]

        # print('Change in len', len(ask_df)-ori_len)

    

    

# test run

# gen_asks()

# ask_df.sample(10)
### Phase 2: ASK -> BID -> MATCH -> UPDATE

# init empty ask_df with col

bid_df = pd.DataFrame(columns = ['location', 'bidder_id', 'utility_to_buyer', 'max_bid_price', 'bid_price','buying_second_house']) # not impt actually

    

def gen_bids():

    ''' phase 2 bid-ask

    1. Refresh bid_df pd.DataFrame()

    2. Generate subdf of persons who can and want to buy houses

    3. For each eligible person, iterate over ask, grow person_bids list of dict

    4. Merge 

    '''

    global bid_df # may not be necessary

    bid_df_columns = bid_df.columns.to_list() # ['location', 'bidder_id', 'utility_to_buyer', 'max_bid_price', 'bid_price']

    

    # 1. Refresh bid_df pd.DataFrame()

    bid_df.drop(bid_df.index, inplace=True) # drops all rows

    

    # 2. Screen viable bidders

    ## 2.1 Does not own a second house (can have 1 or 0 houses)

    COND_no_second_house = persons['house_selling'].isna() # NOTE: do not use `persons['house_selling'] == None` to check

    potential_buyers = persons[COND_no_second_house]

    

    ## 2.2 Random decide if want to seek or not

    global PROBA_BUY

    PROBA_BUY = PROBA_BUY # arbitrary threshold; TODO: turn into adjustable param

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

        buyer_view_of_ask_df['utility_to_buyer'] = utility_function_vectorised(buyer, buyer_view_of_ask_df) # person, house

        # NOTE: utility_to_buyer is partial -- it only consider's a houses's general and locational utility and buyer idio

        

        ### 3.2.2 Calculate bid_price

        buyer_view_of_ask_df['max_bid_price'] = buyer['wealth'] - buyer['utility'] + buyer_view_of_ask_df['utility_to_buyer'] # TODO: double check if this is a good rule

        # utility came from preceding iter(s). If utility fromp previous is very high, the buyer's max bid price will be lower, all else equal.

        # if bid is successful, the utility_to_buyer of new house will replace previous utility value (from old house)

        buyer_view_of_ask_df['max_bid_price'] = buyer_view_of_ask_df['max_bid_price'].apply(lambda mbp: min(mbp, buyer['wealth']))

        # mbp must be capped at buyer's wealth

        buyer_view_of_ask_df['max_bid_price'] = buyer_view_of_ask_df['max_bid_price'].apply(lambda mbp: max(0,mbp))

        # mbp must be non-negative

        

        bid_price = buyer_view_of_ask_df.apply(_gen_bid_price, axis=1)

        buyer_view_of_ask_df['bid_price'] = bid_price

        

        ### 3.2.3 Mark out second house buyers - updated(weets, 191125)

        buyer_view_of_ask_df['buying_second_house'] = type(buyer['house_staying']) == tuple # if have house_staying location tuple, then is buying second house

        

        ### 3.2.4 Append specific columns of buyer_view_of_ask_df to list_of_bid_sets

        select_columns = ['location', 'bidder_id', 'utility_to_buyer', 'max_bid_price', 'bid_price', 'buying_second_house']

        list_of_bid_sets.append(buyer_view_of_ask_df[select_columns])

    

    # 4. Concatenate list of dataframes into one dataframe

    if list_of_bid_sets: # possible that no bids take place

        bid_df = pd.concat(list_of_bid_sets)

    return bid_df



# bid_df = gen_bids()

# print(bid_df['bidder_id'].nunique())

# bid_df.head()
### Phase 2: ASK -> BID -> MATCH -> UPDATE

match_df = pd.DataFrame(columns = ['location','amenities','distance_to_city','ask_price','num_bids','highest_bid_value','mean_bid_value','winning_bidder_id','winning_bid_value'])

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

    global bid_df, persons, houses, match_df

    # 1. Create a container list to store dicts of info relating to bidding for each listing

    list_of_matches = [] # contains info on winning bid

    

    # 2. Iterate over listings in ask_df, find best bid - is successful match

    for idx, listing in ask_df.sample(frac=1).iterrows(): # shuffles ask_df

        match_info_dict = {} # stats for each listing match

        

        ## 2.1 Get general data

        listing_loc = listing['location']

        match_info_dict['location'] = listing_loc

        

        match_info_dict['amenities'] = listing['amenities']

        match_info_dict['distance_to_city'] = listing['distance_to_city']

        

        match_info_dict['ask_price'] = listing['ask_price']

        

        relevant_bids = bid_df[bid_df['location']==listing_loc]

        match_info_dict['num_bids'] = len(relevant_bids) # expect 0 or more

        

        highest_bid_value = relevant_bids['bid_price'].max() # might be NaN if len(relevant_bids) == 0

        match_info_dict['highest_bid_value'] = highest_bid_value

        

        match_info_dict['mean_bid_value'] = relevant_bids['bid_price'].mean() # might be NaN if len(relevant_bids) == 0

        

        # 3. Found winning bid(s)

        if highest_bid_value >= listing['ask_price']: # there exists a successful match; NaN compatible

            ## 3.1 Create and append dict of info relating to bids for the listing

            ### 3.1.1 Check for ties among highest bid

            highest_bids = relevant_bids[relevant_bids['bid_price']==highest_bid_value]

            num_highest_bid = len(highest_bids) # expect at least 1, unlikely but possibly more

            assert num_highest_bid >= 1, 'ERR: num_highest_bid must be >= 1'

            

            ### 3.1.2 Get the winner

            winning_bid = highest_bids.sample(1) # tie-breaker: randomly choose one highest bidder to win

            

            winning_bidder_id = winning_bid['bidder_id'].iloc[0]

            match_info_dict['winning_bidder_id'] = winning_bidder_id

            match_info_dict['winning_bid_value'] = highest_bid_value # obviously; but stated explicitly as highest_bid_value may not win for the `else` case

            

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

                persons['house_selling'].loc[asker_id] = np.NaN # check for error here

                # weets: asker utility does not increase on sale of house. Utility will be utility of the house staying, given the price the person bought it at.

                # thus, sales of second house does not affect asker's utility

                

            ### 3.4.2 Update bidder

            winning_bidder = persons.loc[winning_bidder_id]

            persons['wealth'].loc[winning_bidder_id] -= highest_bid_value

            

            #### Additional updates for bidder if second house buyer

            if winning_bid['buying_second_house'].iloc[0]: # first house exists, buyer is buying second house

            # if type(winning_bidder['house_staying']) is tuple: # first house exists, buyer is buying second house

                persons['house_selling'].loc[winning_bidder_id] = "random string to recast type"

                persons['house_selling'].loc[winning_bidder_id] = winning_bidder['house_staying'] # set current house_staying to be house_selling

                houses['status'].loc[winning_bidder['house_staying']] = 'selling' # set that same current house to 'selling' status

            persons['house_staying'].loc[winning_bidder_id] = "random string to recast type"

            persons['house_staying'].loc[winning_bidder_id] = listing_loc

            

            prev_utility =  persons['utility'].loc[winning_bidder_id] # get old

            persons['utility'].loc[winning_bidder_id] = winning_bid['utility_to_buyer'].iloc[0] # set new utility

            

            # print('Utility change:',persons['utility'].loc[winning_bidder_id] - prev_utility) # should be non-negative

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

    

    ## 5.1 Additional calculations

    def _get_dist(loc):

        return ((loc[0]-CITY_X)**2 + (loc[1]-CITY_Y)**2)**(0.5)

    

    match_df['distance_to_city'] = match_df['location'].apply(_get_dist)

    return match_df





# gen_asks()

# bid_df = gen_bids()

# match_df = match_ask_bid() # Note: changes bid_df each time it is called

# match_df.head(10)
def update_market_price():

    global houses, match_df, MIN_TXN_VOL

    

    # 1. Update market price using info from bid-ask-match data

    clean_matches = match_df[~match_df['highest_bid_value'].isna()]

    if len(clean_matches):

        ## 1.1 build linear regression model for market_price

        X = clean_matches[['amenities','distance_to_city']]

        

        Y = clean_matches['highest_bid_value'].values

        lm = LinearRegression().fit(X,Y)

        

        if DEBUG:

            print('score {} m {} c {}'.format(lm.score(X, Y),

                                           lm.coef_,

                                           lm.intercept_))

        

        ## 1.2 Build market pricer function 

        def cal_market_price(houses_df): # update(weets,191202)

            '''Applies linear predictor model

            '''

            X = houses_df[['amenities','distance_to_city']].values.reshape(1, -1)

            pred_market_price = max(lm.predict(X).item(),0)

            return pred_market_price

            

        def _choose_market_pricer():

            if len(clean_matches) >= MIN_TXN_VOL: # update(weets,191202)

                # if sufficient transactions occur, use linear model

                return cal_market_price # update(weets, 191202)

            else: return lambda _: clean_matches['highest_bid_value'].median() #  if not, just use median of highest bid values

        

        market_pricer = _choose_market_pricer()

            

        ## 1.3 Update market prices

        houses = houses[~houses['amenities'].isna()] # drops any rows from houses that do not have an amenities data

        houses['market_price'] = houses.apply(market_pricer, axis=1) # update(weets,191202)

    

    # 2. Update market price given current affairs

    def gen_current_affair_multiplier():

        global PROBA_CA_GOOD, PROBA_CA_BAD, CA_MULTIPLIER_GOOD, CA_MULTIPLIER_BAD, CA_MULTIPLIER_FAIR

        u = np.random.uniform(0,1)

        if u < PROBA_CA_GOOD: return CA_MULTIPLIER_GOOD()

        elif u < PROBA_CA_GOOD+PROBA_CA_BAD: return CA_MULTIPLIER_BAD()

        else: return CA_MULTIPLIER_FAIR()

    

    current_affairs_multipler = gen_current_affair_multiplier() # ranges from BAD_LB to GOOD_UB

    print('CAM', current_affairs_multipler)

    houses['market_price'] = houses['market_price'].apply(lambda mp: mp*current_affairs_multipler)

        

# update_market_price() 
DEBUG = False

debug_run = 100

if DEBUG:

    for i in range(debug_run):

        aging()

        birth()

        dying()

        print('persons',len(persons))

        print(persons['wealth'].max())

        gen_asks()

        print('ask',len(ask_df))

        print('min ask',ask_df['ask_price'].min())

        gen_bids()

        print('bid',len(bid_df))

        print(bid_df['bid_price'].max())

        match_ask_bid()

        print('successful match',len(match_df[~match_df['winning_bidder_id'].isna()]))

        update_market_price()
if DEBUG: match_df[~match_df['highest_bid_value'].isna()].plot(kind='scatter',x='amenities',y='highest_bid_value')
if DEBUG: match_df[~match_df['highest_bid_value'].isna()].plot(kind='scatter',x='distance_to_city',y='highest_bid_value')
if DEBUG: houses[~houses['market_price'].isna()].plot(kind='scatter',x='amenities',y='market_price')
if DEBUG: houses[~houses['market_price'].isna()].plot(kind='scatter',x='distance_to_city',y='market_price')
# hotfix as NaN row appears in houses

# somehow NaN is introduced to the dataframe index

# this converts the index to a numpy float which I need to reconvert back



def drop_NaN_row(df):

    if np.NaN in df.index:

        df = df.drop([np.NaN])

    return df



def convert_to_tuple_int(index):

    return (int(index[0]), int(index[1]))

houses = drop_NaN_row(houses)

houses.index = houses.index.map(convert_to_tuple_int)
%%time

snapshots = []

snapshot = {}

snapshot["houses"] = houses.copy()

snapshot["persons"] = persons.copy()

snapshot["ask_df"] = ask_df.copy()

snapshot["bid_df"] = bid_df.copy()

snapshot["match_df"] = match_df.copy()

snapshot["history"] = deepcopy(history)

snapshots.append(deepcopy(snapshot))



def simulate():

    aging()

    birth()

    dying()

    gen_asks()

    gen_bids()

    match_ask_bid()

    update_history()

    update_market_price()

    

    # hotfix as NaN row appears in houses

    global houses

    houses = drop_NaN_row(houses)

    houses.index = houses.index.map(convert_to_tuple_int)

    

for _ in tqdm_notebook(range(NUM_FRAMES)):

    simulate()

    snapshot["houses"] = houses.copy()

    snapshot["persons"] = persons.copy()

    snapshot["ask_df"] = ask_df.copy()

    snapshot["bid_df"] = bid_df.copy()

    snapshot["match_df"] = match_df.copy()

    snapshot["history"] = deepcopy(history)

    snapshots.append(deepcopy(snapshot))
%%time

# save data (and an example to load the data)

import pickle

with open('snapshots.pkl', 'wb') as handle: pickle.dump(snapshots, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('snapshots.pkl', 'rb') as handle: unserialized_data = pickle.load(handle)

print(str(snapshots) == str(unserialized_data))
%%time

# retrieve snapshots

houses = snapshots[0]["houses"]

persons = snapshots[0]["persons"]

history = snapshots[0]["history"]



zero_house = persons[persons["house_staying"]!=persons["house_staying"]]

two_houses = persons[persons["house_selling"]==persons["house_selling"]]

one_houses = persons[(persons["house_staying"]==persons["house_staying"]) & 

                     (persons["house_selling"]!=persons["house_selling"])]



house_empty = houses[houses["status"]=="empty"]

house_occu  = houses[houses["status"]=="occupied"]

house_sell  = houses[houses["status"]=="selling"]



# initialise graphs

fig, ax = plt.subplots(nrows=5, ncols=5,figsize=(12,12),

                       gridspec_kw={"height_ratios":[0.8, 0.05, 1.5, 1.5, 1.5]})

plt.subplots_adjust(wspace=0.4)
%%time

# colormaps

im0 = ax[0,0].imshow(np.random.randn(10,10), vmin=0, vmax=1000, cmap="inferno_r")

im1 = ax[0,1].imshow(np.random.randn(10,10), cmap="inferno_r",

                     norm=colors.LogNorm(vmin=10, vmax=1000))

im2 = ax[0,2].imshow(np.random.randn(10,10), cmap="inferno_r",

                     norm=colors.LogNorm(vmin=10, vmax=1000))

im3 = ax[0,3].imshow(np.random.randn(10,10), cmap="inferno_r", vmin=0, vmax=100)

im4 = ax[0,4].imshow(np.random.randn(10,10), cmap=colors.ListedColormap(['blue','orange','green']), 

                     vmin=0, vmax=2)

ax[0,0].set_title("utility_general")

ax[0,1].set_title("market_price")

ax[0,2].set_title("last_bought_price")

ax[0,3].set_title("last_updated")

ax[0,4].set_title("occupancy status")



fig.colorbar(im0, cax=ax[1,0], orientation="horizontal")

fig.colorbar(im1, cax=ax[1,1], orientation="horizontal")

fig.colorbar(im2, cax=ax[1,2], orientation="horizontal")

fig.colorbar(im3, cax=ax[1,3], orientation="horizontal")

fig.colorbar(im4, cax=ax[1,4], orientation="horizontal")

ax[1,4].set_xticklabels(['Empty', 'Occupied', 'Selling'])

pass
%%time

# plots on wealth

line_pop_0, = ax[2,0].plot([], lw=3, label="zero houses")

line_pop_1, = ax[2,0].plot([], lw=3, label="one house")

line_pop_2, = ax[2,0].plot([], lw=3, label="two houses")

max_plot_pop = 60

ax[2,0].set_ylim((0, max_plot_pop))

ax[2,0].set_xlim((-20, 0))

ax[2,0].set_title("population")

ax[2,0].legend(loc="upper right")



line_wealth_0, = ax[2,1].plot([], lw=3, label="average wealth")

max_plot_wealth = 600

ax[2,1].set_ylim((0, max_plot_wealth))

ax[2,1].set_xlim((-20, 0))

ax[2,1].set_title("average wealth")

ax[2,1].legend(loc="upper right")



scat_wealth_age_zero = ax[2,2].scatter([], [], s=20, label="zero houses", color="blue")

scat_wealth_age_one = ax[2,2].scatter([], [], s=20, label="one house", color="orange")

scat_wealth_age_two = ax[2,2].scatter([], [], s=20, label="two house", color="green")

ax[2,2].set_ylim((0, max_plot_wealth))

ax[2,2].set_xlim((20, 60))

ax[2,2].set_title("wealth against age")

ax[2,2].legend(loc="upper right")



ax[2,3].hist([], color="green")

ax[2,3].set_title("histogram of wealth")

ax[2,3].set_xlim((0, max_plot_wealth))

ax[2,3].set_ylim((0, 20))



line_house_0, = ax[2,4].plot([], lw=3, label="empty", color="blue")

line_house_1, = ax[2,4].plot([], lw=3, label="occupied", color="orange")

line_house_2, = ax[2,4].plot([], lw=3, label="selling", color="green")

ax[2,4].set_ylim((0, 100))

ax[2,4].set_xlim((-20, 0))

ax[2,4].set_title("housing occupancy")

ax[2,4].legend(loc="upper right")

pass
%%time

ax[3,0].set_axis_off()

# plots on utility

line_utility_0, = ax[3,1].plot([], lw=3, label="average utility")

max_plot_utility = 600

ax[3,1].set_ylim((0, max_plot_utility))

ax[3,1].set_xlim((-20, 0))

ax[3,1].set_title("average utility")

ax[3,1].legend(loc="upper right")



scat_utility_age_zero = ax[3,2].scatter([], [], s=20, label="zero houses", color="blue")

scat_utility_age_one = ax[3,2].scatter([], [], s=20, label="one house", color="orange")

scat_utility_age_two = ax[3,2].scatter([], [], s=20, label="two house", color="green")

ax[3,2].set_ylim((0, max_plot_utility))

ax[3,2].set_xlim((20, 60))

ax[3,2].set_title("utility against age")

ax[3,2].legend(loc="upper right")



scat_utility_wealth_zero = ax[3,4].scatter([], [], s=20, label="zero houses", color="blue")

scat_utility_wealth_one = ax[3,4].scatter([], [], s=20, label="one house", color="orange")

scat_utility_wealth_two = ax[3,4].scatter([], [], s=20, label="two house", color="green")

ax[3,4].set_ylim((0, max_plot_utility))

ax[3,4].set_xlim((0, max_plot_wealth))

ax[3,4].set_title("utility against wealth")

ax[3,4].legend(loc="upper right")



ax[3,3].hist([], color="green")

ax[3,3].set_title("histogram of utility")

ax[3,3].set_xlim((0, max_plot_wealth))

ax[3,3].set_ylim((0, 20))

pass
%%time

ax[4,0].set_axis_off()

# plots on housing

line_price_0, = ax[4,1].plot([], lw=3, label="price", color="red")

max_plot_price = 200

ax[4,1].set_ylim((0, max_plot_price))

ax[4,1].set_xlim((-20, 0))

ax[4,1].set_title("average last bought price")

ax[4,1].legend(loc="upper right")



scat_price_util_empty = ax[4,2].scatter([], [], s=20, marker="s", label="empty", color="blue")

scat_price_util_occu  = ax[4,2].scatter([], [], s=20, marker="s", label="occupied", color="orange")

scat_price_util_sell  = ax[4,2].scatter([], [], s=20, marker="s", label="selling", color="green")

ax[4,2].set_xlim((0, max(houses["utility_general"])))

ax[4,2].set_ylim((0, max_plot_price))

ax[4,2].set_title("last price against utility")

ax[4,2].legend(loc="upper right")



line_market_0, = ax[4,3].plot([], lw=3, label="price", color="red")

ax[4,3].set_ylim((0, max_plot_price))

ax[4,3].set_xlim((-20, 0))

ax[4,3].set_title("average market price")

ax[4,3].legend(loc="upper right")



scat_market_util_empty = ax[4,4].scatter([], [], s=20, marker="s", label="empty", color="blue")

scat_market_util_occu  = ax[4,4].scatter([], [], s=20, marker="s", label="occupied", color="orange")

scat_market_util_sell  = ax[4,4].scatter([], [], s=20, marker="s", label="selling", color="green")

ax[4,4].set_xlim((0, max(houses["utility_general"])))

ax[4,4].set_ylim((0, max_plot_price))

ax[4,4].set_title("market price against utility")

ax[4,4].legend(loc="upper right")

pass

# to add - ask_df and bid_df
%%time

plt.tight_layout()

patch = [im0, im1, im2, im3, im4]

a1,a2,a3,a4,a5 = tuple([np.empty((10,10,)) for _ in range(5)])



def update_plot():

    a1[:], a2[:], a3[:], a4[:], a5[:] = (np.NaN,)*5

    for index in list(houses.index):

        a1[index[0], index[1]] = houses.loc[index,"utility_general"]

        a2[index[0], index[1]] = houses.loc[index,"market_price"]

        a3[index[0], index[1]] = houses.loc[index,"last_bought_price"]

        a4[index[0], index[1]] = houses.loc[index,"last_updated"]

        a5[index[0], index[1]] = status_to_float(houses.loc[index,"status"])

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

    global max_plot_pop

    max_plot_pop = max(max_plot_pop, 

                       history["popn_with_zero_house"][-1], 

                       history["popn_with_one_house"][-1], 

                       history["popn_with_two_house"][-1])

    ax[2,0].set_ylim(0, max_plot_pop)

    

    line_house_0.set_data(range(0,-len(history["total_houses_empty"][-20:]),-1), 

                          history["total_houses_empty"][-20:][::-1])

    line_house_1.set_data(range(0,-len(history["total_houses_occupied"][-20:]),-1), 

                          history["total_houses_occupied"][-20:][::-1])

    line_house_2.set_data(range(0,-len(history["total_houses_selling"][-20:]),-1), 

                          history["total_houses_selling"][-20:][::-1])

    

    line_wealth_0.set_data(range(0,-len(history["average_wealth"][-20:]),-1), 

                           history["average_wealth"][-20:][::-1])

    line_utility_0.set_data(range(0,-len(history["average_utility"][-20:]),-1), 

                            history["average_utility"][-20:][::-1])

    line_price_0.set_data(range(0,-len(history["average_price"][-20:]),-1), 

                          history["average_price"][-20:][::-1])

    line_market_0.set_data(range(0,-len(history["average_market"][-20:]),-1), 

                          history["average_market"][-20:][::-1])

    

    zero_house = persons[persons["house_staying"]!=persons["house_staying"]]

    two_houses = persons[persons["house_selling"]==persons["house_selling"]]

    one_houses = persons[(persons["house_staying"]==persons["house_staying"]) & 

                         (persons["house_selling"]!=persons["house_selling"])]

    

    house_empty = houses[houses["status"]=="empty"]

    house_occu  = houses[houses["status"]=="occupied"]

    house_sell  = houses[houses["status"]=="selling"]



    

    global max_plot_wealth, max_plot_utility, max_plot_price

    max_plot_wealth = max(max_plot_wealth, max(persons["wealth"]))

    max_plot_utility = max(max_plot_utility, max(persons["utility"]))

    max_plot_price = max(max_plot_price, max(houses["last_bought_price"]))

    ax[2,1].set_ylim(0, max_plot_wealth)

    ax[2,2].set_ylim(0, max_plot_wealth)

    ax[3,1].set_ylim(0, max_plot_utility)

    ax[3,2].set_ylim(0, max_plot_utility)

    ax[3,4].set_xlim(0, max_plot_wealth)

    ax[3,4].set_ylim(0, max_plot_utility)

    ax[4,1].set_ylim(0, max_plot_price)

    ax[4,2].set_ylim(0, max_plot_price)

    

    scat_wealth_age_zero.set_offsets(np.transpose((zero_house["age"], zero_house["wealth"])))

    scat_wealth_age_one.set_offsets(np.transpose((one_houses["age"], one_houses["wealth"])))

    scat_wealth_age_two.set_offsets(np.transpose((two_houses["age"], two_houses["wealth"])))

    

    scat_utility_age_zero.set_offsets(np.transpose((zero_house["age"], zero_house["utility"])))

    scat_utility_age_one.set_offsets(np.transpose((one_houses["age"], one_houses["utility"])))

    scat_utility_age_two.set_offsets(np.transpose((two_houses["age"], two_houses["utility"])))



    scat_utility_wealth_zero.set_offsets(np.transpose((zero_house["wealth"], zero_house["utility"])))

    scat_utility_wealth_one.set_offsets(np.transpose((one_houses["wealth"], one_houses["utility"])))

    scat_utility_wealth_two.set_offsets(np.transpose((two_houses["wealth"], two_houses["utility"])))



    scat_price_util_empty.set_offsets(np.transpose((house_empty["utility_general"], house_empty["last_bought_price"])))

    scat_price_util_occu.set_offsets(np.transpose((house_occu["utility_general"], house_occu["last_bought_price"])))

    scat_price_util_sell.set_offsets(np.transpose((house_sell["utility_general"], house_sell["last_bought_price"])))

    

    scat_market_util_empty.set_offsets(np.transpose((house_empty["utility_general"], house_empty["market_price"])))

    scat_market_util_occu.set_offsets(np.transpose((house_occu["utility_general"], house_occu["market_price"])))

    scat_market_util_sell.set_offsets(np.transpose((house_sell["utility_general"], house_sell["market_price"])))

    

    # histograms

    ax[2,3].cla()

    ax[2,3].hist(persons["wealth"], bins=np.linspace(0,max_plot_wealth,15), color="green")

    ax[2,3].set_title("histogram of wealth")

    ax[2,3].set_xlim((0, max_plot_wealth))

    ax[2,3].set_ylim((0, 20))



    ax[3,3].cla()

    ax[3,3].hist(persons["utility"], bins=np.linspace(0,max_plot_utility,15), color="green")

    ax[3,3].set_title("histogram of utility")

    ax[3,3].set_xlim((0, max_plot_utility))

    ax[3,3].set_ylim((0, 20))

    

def init():

    return patch



def animate(i):

    global houses, persons, history

    houses  = snapshots[i+1]["houses"]

    persons = snapshots[i+1]["persons"]

    history = snapshots[i+1]["history"]

    update_plot()

    print("*", end="")

    return patch



# call the animator. blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=NUM_FRAMES, interval=MILLISECS_PER_FRAME, blit=True)



vid = anim.to_html5_video()

plt.close()

print()
HTML(vid)
persons
houses
%%time

import plotly.figure_factory as ff

# reference https://plot.ly/python/annotated_heatmap/#custom-hovertext



z = np.empty([10, 10])

z[:] = np.nan

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





colorscale=[[0.00, 'rgb{}'.format(matplotlib.cm.get_cmap('viridis', 12)(0.0)[:3])],

            [0.50, 'rgb{}'.format(matplotlib.cm.get_cmap('viridis', 12)(0.5)[:3])],

            [1.00, 'rgb{}'.format(matplotlib.cm.get_cmap('viridis', 12)(1.0)[:3])]]



# Make Annotated Heatmap

fig = ff.create_annotated_heatmap(z, text=hover, hoverinfo='text', annotation_text=ann,

                                  colorscale=colorscale, font_colors=['black'])

fig.update_layout(title_text='Occupancy Status',

                  yaxis = dict(scaleanchor = "x", scaleratio = 1),

                  template="plotly_white")

fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()
# results = []

# for i in range(0):

#     global NUM_BORN, NUM_FRAMES

#     snapshots = []

#     snapshot = {}

#     NUM_BORN = lambda: 1 + i*5

#     initialise_persons()

#     initialise_houses()

    

#     global history

#     history = defaultdict(list)

    

#     global ask_df, bid_df, match_df

#     ask_df = pd.DataFrame(columns = ['location','occupant_id','amenities', 'ask_price']) # init empty ask_df with col

#     bid_df = pd.DataFrame(columns = ['location', 'bidder_id', 'utility_to_buyer', 'max_bid_price', 'bid_price','buying_second_house']) # not impt actually

#     match_df = pd.DataFrame(columns = ['location','amenities','ask_price','num_bids','highest_bid_value','mean_bid_value','winning_bidder_id','winning_bid_value'])



#     for _ in tqdm_notebook(range(NUM_FRAMES)):

#         simulate()

#         snapshot["houses"] = houses.copy()

#         snapshot["persons"] = persons.copy()

#         snapshot["history"] = deepcopy(history)

#         snapshots.append(deepcopy(snapshot))

#     results.append(np.mean(history["popn_with_zero_house"]))
# plt.figure(figsize=(14,5))

# plt.plot(results)