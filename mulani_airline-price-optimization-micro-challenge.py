import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

pricing_function(2, 6, 184)
from math import ceil, pow



def pricing_function(days_left, tickets_left, demand_level):

    

    bucket_count = pow(2,days_left)-1

    

    bucket_number = ceil((bucket_count * (demand_level - 100.0))/100.0)

    

    return demand_level - bucket_number * (tickets_left / bucket_count)

simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)