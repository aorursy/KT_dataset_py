import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

from math import floor

from flight_revenue_simulator import _tickets_sold



def pricing_function(days_left, tickets_left, demand_level):

    int_demand_level = int(floor(demand_level))

    maxRevenue = 0

    bestPrice = 0

    for price in range(1, int_demand_level + 1):

        tickets_sold = _tickets_sold(int_demand_level, price, tickets_left)

        revenue = price * tickets_sold + revenue_function(days_left - 1, tickets_left - tickets_sold)

        if maxRevenue < revenue:

            maxRevenue = revenue

            bestPrice = price    

    return bestPrice + demand_level - int_demand_level - 0.00001



cache_revenue = {}



def revenue_function(days_left, tickets_left):

    if days_left <= 0 or tickets_left <= 0:

        return 0

    elif (days_left, tickets_left) in cache_revenue:

        return cache_revenue[(days_left, tickets_left)]

    else:

        # evaluate all possibilities from 100 to 200 included

        revenue = 0

        for demand_level in range(100, 201):

            price = pricing_function(days_left, tickets_left, demand_level)

            tickets_sold = _tickets_sold(demand_level, price, tickets_left)

            revenue += price * tickets_sold + revenue_function(days_left - 1, tickets_left - tickets_sold)

        #average revenue

        revenue /= 101

        cache_revenue[(days_left, tickets_left)] = revenue

        return revenue

# Average revenue across all flights is $7596
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)