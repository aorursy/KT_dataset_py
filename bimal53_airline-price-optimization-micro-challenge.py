import sys

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

demand_list = []

avrg_demand = demand_list



def avrg_calc(demand_level):

    avrg_demand.append(demand_level)

    return np.mean(avrg_demand)



def std_demand(demand_level):

    return np.std(avrg_demand)
def pricing_function(days_left, tickets_left, demand_level):

    average_demand = avrg_calc(demand_level)

    STD_demand = std_demand(demand_level)

    price = demand_level - round((tickets_left / days_left))

    if(average_demand > demand_level and (len(avrg_demand) != 0 ) and days_left > 1 ):

        price = demand_level - (tickets_left / days_left) + 3

    if( (average_demand - STD_demand) > demand_level and (len(avrg_demand) != 0 ) and days_left > 1 ):  

        price = demand_level - (tickets_left / days_left) + 6

    if( (demand_level  >= average_demand + (1.35* STD_demand)) and (len(avrg_demand) >= 5 )):  

        price = demand_level - (tickets_left / days_left) - 11

    if( (demand_level  <= average_demand - (2* STD_demand)) and (len(avrg_demand) >= 5 )):

        price = demand_level - (tickets_left / days_left) + 9   

    return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
demand_list = []

avrg_demand = demand_list

score_me(pricing_function)