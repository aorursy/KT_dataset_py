import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    if days_left == 1:

        price = demand_level-tickets_left

    else:

        price = demand_level-tickets_left/days_left*demand_level**5/(150**5)

    return price
x = simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=False)
print(x)
for i in  range(80,100):

    def pricing_function(days_left, tickets_left, demand_level):

        """Sample pricing function"""

        if days_left == 1:

            price = demand_level-tickets_left

        else:

            price = demand_level-tickets_left/days_left*(demand_level/165)**(i/10)

        return price

    print(i/10)

    score_me(pricing_function)
def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    if days_left == 1:

        price = demand_level-tickets_left

    else:

        price = demand_level-tickets_left/days_left*(demand_level/165)**9.9

    return price

score_me(pricing_function)
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)