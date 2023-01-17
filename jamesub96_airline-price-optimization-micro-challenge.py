import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

import pandas as pd
def max_profit(x):
    test = pd.DataFrame(columns=['price'])
    demand_level = int(x)
    for i in range(0,demand_level,1):
        test = test.append({'price': i}, ignore_index=True)
        
    test['profit'] = test['price'].apply(lambda x: (demand_level-x)*x)
    value = test['profit'].idxmax()
    return value
int(max_profit(175))
def pricing_function(days_left, tickets_left, demand_level):
    if days_left == 1:
        price = demand_level-tickets_left
        return price
    elif days_left >1:
        price = max_profit(demand_level)
        return price
def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    if days_left == 1:
        price = demand_level-tickets_left
        return price
    elif demand_level >= 190:
        price = demand_level - 20
        return price
    elif 180 <= demand_level < 190:
        price = demand_level - 10
        return price
    elif 150 <= demand_level <=180:
        price = demand_level - 5
        return price
    elif demand_level < 150:
        price = demand_level -0
        return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)