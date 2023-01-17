import matplotlib.pyplot as plt

import numpy as np



demand_level = 150

# only plot quantities up to demand_level. That's point where price becomes 0

revenue = [(demand_level - q) * q for q in range(demand_level)]



# Make primary plot

plt.plot(revenue, label="Revenue")

plt.xlabel("Seats Sold")

plt.ylabel("Dollars")

plt.ylim(ymin=0)

plt.xlim(xmin=0)



# Show the revenue maximizing quantity

optimal_q = np.argmax(revenue)

plt.vlines(x=optimal_q, ymin=0, ymax=max(revenue), color='green', linestyle="--", label="Optimal Quantity")



plt.legend(loc='lower right')

plt.show()
n_demand_levels = 100 # Instead of 11 levels, we will take 100 levels. Our granularity will increase

min_demand_level = 100

max_demand_level = 200

demand_levels = np.linspace(min_demand_level, max_demand_level, n_demand_levels)



max_tickets = 200

max_days = 150



# Q indices are: n_sold in day, tickets_left to start day, demand_level, days_left

Q = np.zeros([max_tickets, max_tickets, n_demand_levels, max_days])

# V indices are: n_left and n_days

V = np.zeros([max_tickets, max_days])

for tickets_left in range(max_tickets):

    for tickets_sold in range(tickets_left+1): # add 1 to offset 0 indexing. Allow selling all tickets

        for demand_index, demand_level in enumerate(demand_levels):

            # Never set negative prices

            price = max(demand_level - tickets_sold, 0)

            Q[tickets_sold, tickets_left, demand_index, 0] = price * tickets_sold

    # For each demand_level, choose the optimum number to sell. Output of this is array .of size n_demand_levels

    revenue_from_best_quantity_at_each_demand_level = Q[:, tickets_left, :, 0].max(axis=0)

    # take the average, since we don't know demand level ahead of time and all are equally likely

    V[tickets_left, 0] = revenue_from_best_quantity_at_each_demand_level.mean()
for days_left in range(1, max_days):

    for tickets_left in range(max_tickets):

        for tickets_sold in range(tickets_left):

            for demand_level in range(100,200):

                price = max(demand_level - tickets_sold, 0)

                rev_today = price * tickets_sold

                Q[tickets_sold, tickets_left, demand_level-100, days_left] = rev_today + V[tickets_left-tickets_sold, days_left-1]

        expected_total_rev_from_best_quantity_at_each_demand_level = Q[:, tickets_left, :, days_left].max(axis=0)

        V[tickets_left, days_left] = expected_total_rev_from_best_quantity_at_each_demand_level.mean()
def pricing_function(days_left, tickets_left, demand_level):

    demand_level_index = np.abs(demand_level - demand_levels).argmin()

    day_index = days_left - 1 # arrays are 0 indexed

    tickets_index = int(tickets_left)  # in case it comes in as float, but need to index with it

    relevant_Q_vals = Q[:, tickets_index, demand_level_index, day_index]

    desired_quantity = relevant_Q_vals.argmax()# offset 0 indexing

    price = demand_level - desired_quantity

    return price
import sys

sys.path.append("../input")

from flight_revenue_simulator import simulate_revenue, score_me



#simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)

score_me(pricing_function)