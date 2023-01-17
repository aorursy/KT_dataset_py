import sys
import numpy as np
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    
    if days_left==1:
       price = demand_level-tickets_left 
    elif tickets_left >= demand_level/2 and demand_level >= 195:
        price = demand_level/2
    elif demand_level >= 150:
        price = demand_level-np.floor(tickets_left/days_left)
    else:
        price = demand_level-1
    return price
simulate_revenue(days_left=7, tickets_left=200, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)