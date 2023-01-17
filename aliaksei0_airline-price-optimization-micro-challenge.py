import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

from math import floor

savedF = {}
savedG = {}

def getG(tickets, days, demand):
    if days <= 0 or tickets <= 0:
        return (0,0)
    if (tickets, days, demand) in savedG:
        return savedG[(tickets, days, demand)]
    topTotal = -1
    topPrice = -1
    for i in range(1, demand+1):
        res = min(demand-i, tickets)*i + (getF(tickets-(demand-i), days-1) if tickets-(demand-i) > 0 else 0)
        if res > topTotal:
            topTotal, topPrice = res, i
    savedG[(tickets, days, demand)] = (topTotal, topPrice)
    return (topTotal, topPrice)
                                       
        

def getF(tickets,days):
    if tickets<=0 or days <= 0:
        return 0
    if (tickets,days) in savedF:
        return savedF[(tickets,days)]
    res = 0.0
    #count math expectation 
    for i in range(100, 201):
        res += getG(tickets, days, i)[0]
    res /= 101
    
    savedF[(tickets,days)] = res
    return res

def pricing_function(days_left, tickets_left, demand_level):
    demand_level_floor = int(floor(demand_level))
    demand_delta = demand_level - demand_level_floor
    (topTotal, topPrice) = getG(tickets_left, days_left, demand_level_floor)
    return topPrice + demand_delta - 0.00001
simulate_revenue(days_left=50, tickets_left=100, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)