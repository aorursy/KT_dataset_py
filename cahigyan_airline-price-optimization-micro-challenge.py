import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    price = demand_level - 10

    return price
simulate_revenue(days_left=10, tickets_left=20, pricing_function=pricing_function, verbose=True)
def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    if days_left > 14:  

        if demand_level >= 190:

            price = demand_level - ((demand_level - 100)/ (10 + ((demand_level-190)/10)))

            return price

        elif demand_level >= 180:

            price = demand_level - ((demand_level - 100)/ (40 + ((demand_level-180)/2)))

            return price

        else:

            price = demand_level

            return price

    if days_left > 2 and days_left <=14:

        if demand_level >= 175:

            price = demand_level - ((demand_level - 100)/ (10 - ((demand_level-175)/4.6)))

            return price

        elif demand_level >= 150:

            price = demand_level - ((demand_level - 100)/ (16 + ((demand_level-150)/5)))

            return price

        else:

            price = demand_level

            return price

    if days_left == 2:

        price = 0.92*demand_level

        return price

    if days_left == 1:

        price = demand_level - tickets_left

        return price

    

score_me(pricing_function)