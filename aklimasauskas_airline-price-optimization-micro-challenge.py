import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me
import numpy as np





price = np.zeros((101, 101, 200), dtype='uint8') # The optimal price given the number of days left, tickets left and current demand level

profit = np.zeros((101, 101)) # Expected profit when pricing optimally given the number of days and tickets left



# Base case when there is 1 day left

for tickets in range(101):

    for demand in range(100, 200):

        q = min(demand // 2, tickets) # This is the optimal quantity to sell

        p = demand - q

        

        price[1, tickets, demand] = p

        profit[1, tickets] += (p*q / 100) # Average out the best profit through all possible demand levels (to get the expected profit)

        

# Fill in the tables

for day in range(2, 101):

    for tickets in range(101):

        for demand in range(100, 200):

            

            best_prof = 0

            best_price = 0

            

            for p in range(1, demand+1): # Go though all feasible prices

                q = min(demand-p, tickets) # This is the quantity that will be sold at price p

                    

                prof = p*q + profit[day-1, tickets-q] # This is the total expected profit if selling at price p

                

                if prof > best_prof: # Update best profit and price

                    best_prof = prof

                    best_price = p

            

            # Set the optimal price and profit values

            price[day, tickets, demand] = best_price

            profit[day, tickets] += (best_prof / 100) # Average out the best profit through all possible demand levels          
def pricing_function(days_left, tickets_left, demand_level):

    """Return the optimal price"""

    

    # These should be cast to an integer (and rounded down)

    tickets_left = int(tickets_left)

    demand_level = int(demand_level)

    

    return price[days_left, tickets_left, demand_level] # Return the precomputed values
# Simulate one run of the pricing

simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
# Get the overall score

score_me(pricing_function)