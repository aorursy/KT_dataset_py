import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me, _tickets_sold

%%time
import numpy as np

N = 100 #days_left = 0..N, tickets_left = 0..N, demand_level - N = 0..N
optimal_price = np.zeros((N+1,N+1,N+1), dtype=int) #(days_left, tickets_left, demand_level)
expected_sales = np.zeros((N+1,N+1), dtype=float) #(days_left, tickets_left)
for days_left in range(1,N+1):
    for tickets_left in range(1,N+1):
        last_best_price = 0 # For previous demand level (one lower)
        for demand_level in range(N,2*N+1):
            # Search over all prices to find the best price for the context: days_left, tickets_left, demand_level
            best_sales = best_price = 0
            # No need to search all prices in range(0, N+1):
            for price in range(max(demand_level - tickets_left, last_best_price), demand_level + 1):
                tickets_sold = demand_level - price
                sales = price * tickets_sold + expected_sales[days_left - 1, tickets_left - tickets_sold]
                if sales > best_sales: best_sales, best_price = sales, price
            expected_sales[days_left, tickets_left] += best_sales/(N+1)
            optimal_price[days_left, tickets_left, demand_level - N] = best_price
            last_best_price = best_price # For higher demand_levels, this best_price is the floor
# 100 days_left, 100 tickets_left, 200 demand_level:
optimal_price[100,100,200-N] # 190... not the most we could charge (199), but takes advantage of unusually high demand (200)
def pricing_function(days_left, tickets_left, demand_level):
    return optimal_price[days_left, int(tickets_left), int(demand_level - 100)]
score_me(pricing_function)
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)
