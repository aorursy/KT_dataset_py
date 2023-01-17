import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""

    if days_left == 1:
        price = (demand_level-tickets_left)
    else:
        probability_of_better_day = sum([(200-demand_level)/100 for x in range(days_left)])/days_left
        fraction_of_tickets_to_sell = (1-probability_of_better_day)
        days_left_penalty = pow(days_left,.8)
        fraction_of_tickets_to_sell = pow(fraction_of_tickets_to_sell,days_left_penalty)
        price = demand_level-tickets_left*fraction_of_tickets_to_sell
        



    
    
    return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)