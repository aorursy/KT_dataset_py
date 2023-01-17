import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

import numpy as np
def create_valuefunctions(remaining_days,remaining_tickets,min_demand_level,max_demand_level):

    demand_levels_n = max_demand_level - min_demand_level

    Q = np.zeros([remaining_days + 1,remaining_tickets + 1,demand_levels_n,remaining_tickets + 1])

    V = np.zeros([remaining_days + 1,remaining_tickets + 1])

    return Q,V



def make_base_step(Q,V,demand_range,remaining_tickets):

    for tickets_left in range(1,remaining_tickets + 1):

        for demand_level_i,demand_level  in enumerate(demand_range):

            for tickets_sold in range(1, tickets_left + 1):

                price = demand_level - tickets_sold

                immediate_reward = tickets_sold * price

                Q[1,tickets_left,demand_level_i,tickets_sold] = immediate_reward

                

        V[1,tickets_left] = Q[1,tickets_left,:,:].max(axis = 1).mean()

    



def dynamic_programming_algorithm(Q,V,remaining_days,remaining_tickets,demand_range):

    for days_left in range(2, remaining_days +1):

        for tickets_left in range(1,remaining_tickets+1):

            for demand_level_i,demand_level  in enumerate(demand_range):

                for tickets_sold in range(1, tickets_left + 1):

                    price = demand_level - tickets_sold

                    immediate_reward = tickets_sold * price

                    Q[days_left,tickets_left,demand_level_i,tickets_sold] = immediate_reward + V[days_left-1,tickets_left - tickets_sold]

                

                V[days_left,tickets_left] = Q[days_left,tickets_left,:,:].max(axis = 1).mean()

    

    return Q,V
remaining_days = 200

remaining_tickets = 150

min_demand_level = 100

max_demand_level = 200



demand_levels_n = max_demand_level - min_demand_level

demand_range = np.linspace(min_demand_level, max_demand_level,demand_levels_n , dtype = int)



Q,V = create_valuefunctions(remaining_days, remaining_tickets, min_demand_level, max_demand_level)

make_base_step(Q,V,demand_range,remaining_tickets)

Q,V = dynamic_programming_algorithm(Q,V,remaining_days,remaining_tickets,demand_range)

   
def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    demand_level_index = np.abs(demand_level - demand_range).argmin()

    demand_level_index = demand_level_index

    relevant_tickest_sold = Q[days_left, int(tickets_left), demand_level_index, :]

    best_tickets_sold = relevant_tickest_sold.argmax()# armax is equal to formula from step 5

    price = max(demand_level - best_tickets_sold,0)

    return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)