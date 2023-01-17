import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):
    price = max(demand_level/2, demand_level-tickets_left)
    return price

def unif_cdf(val, low=100, high=200):
    val = floor(val)
    if(val > high):
        return 0
    elif(val<low):
        return 1
    else:
        return 1-((val-low + 1)/(high-low + 1))

from math import floor
import numpy as np


def expected(demand, tickets):
    l = []
    for i in range(0,int(tickets)+1):
        price_point = demand-i
        tickets_left_after_today = tickets - i
        revenue_today = i*price_point
        future_prob = unif_cdf(price_point + tickets_left_after_today)
        expected_price_tomorrow = ((price_point+tickets_left_after_today) + 200)/2 
        future_val = tickets_left_after_today * expected_price_tomorrow 
        l.append(revenue_today+future_prob*future_val)
    return l
    
    
def pricing_function(days_left, tickets_left, demand_level):
    if days_left == 1:
        price = max(demand_level/2, demand_level-tickets_left)
    else:
        to_sell = np.argmax(expected(floor(demand_level), tickets_left))
        price = demand_level-to_sell
    return price
from math import floor
import numpy as np
import scipy

def valuation(days, dem, tickets, valuation_point):
    prob = unif_cdf(valuation_point)
    n = days*prob
    v = tickets*(((200 + valuation_point)/2)-(tickets/n))
    return v
    
def expected(demand, tickets, days_left):
    l = []
    for i in range(0,int(tickets)+1):
        today = i*(demand-i)
        future_val = valuation(days_left,demand, tickets-i, demand-i)
        l.append(today + future_val)
    return l
    
    
def pricing_function(days_left, tickets_left, demand_level):
    if days_left == 1:
        price = max(demand_level/2, demand_level-tickets_left)
    else:
        to_sell = np.argmax(expected(demand_level, tickets_left, days_left))
        price = demand_level-to_sell
    return price

simulate_revenue(days_left=1, tickets_left=3, pricing_function=pricing_function, verbose=True)
%%timeit
score_me(pricing_function)
#7482, 2.07s +/- 35ms/loop
from math import floor
import numpy as np
import scipy

def valuation(days, dem, tickets, valuation_point):
    k = np.arange(1,days, dtype=np.dtype(np.int32))
    prob = unif_cdf(valuation_point)
    p = scipy.special.comb(days-1,k)*np.power(prob,k)*np.power((1-prob), days-k-1)
    v = tickets*(((200 + valuation_point)/2)-(tickets/k))
    vp = v*p
    
    return np.sum(v*p)
    
    
def expected(demand, tickets, days_left):
    l = []
    for i in range(0,int(tickets)+1):
        today = i*(demand-i)
        future_val = valuation(days_left,demand, tickets-i, demand-i)
        l.append(today + future_val)
        #print("{}: {} + {} = {}".format(i, today, future_val, today+future_val))
    return l
    
    
def pricing_function(days_left, tickets_left, demand_level):
    if days_left == 1:
        price = max(demand_level/2, demand_level-tickets_left)
    else:
        to_sell = np.argmax(expected(demand_level, tickets_left, days_left))
        price = demand_level-to_sell
    return price
%%timeit
score_me(pricing_function)
#7533, 50.4 s ± 172 ms/loop