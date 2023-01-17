import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

import numpy as np



expectancies = lambda days: np.array([100*(1+i/(days+1)) for i in range(days, 0, -1)])

expected_revenue = lambda d, p: sum(np.maximum(0, np.floor(d - p)) * p)



def pricing_function(days_left, tickets_left, demand_level):

    if days_left == 1:

        price = max(demand_level / 2, demand_level - tickets_left)

    else:

        # What's the maximum (level) price at which we can expect to clear our inventory?

        expected_demand = np.concatenate([[demand_level], expectancies(days_left - 1)])

        price = max(expected_demand) - 1

        while(True):

            cleared_inventory = sum(np.maximum(0, np.floor(expected_demand - price)))

            if cleared_inventory < tickets_left: price -= 1

            else: break

        # Shift sales around to take advantage of different revenue sensitivities to price

        prices = np.ones(days_left) * price

        while(True):

            marginal_gains = np.maximum(0, expected_demand - prices - 1)

            source = marginal_gains.argmax()

            opportunity_costs = np.maximum(0, expected_demand - prices) + prices[source] - np.minimum(expected_demand, prices) + 1

            target = opportunity_costs.argmin()

            adjusted_prices = prices.copy()

            adjusted_prices[source] += 1

            adjusted_prices[target] -= max(1, prices[target] - expected_demand[target] + 1)

            if expected_revenue(expected_demand, prices) >= expected_revenue(expected_demand, adjusted_prices): break

            else: prices = adjusted_prices.copy()

        price = prices[0]

    return price
score_me(pricing_function)
demand_levels = []



def hacking_function(days_left, tickets_left, demand_level):

    global demand_levels

    demand_levels.append(demand_level)

    return demand_level



score_me(hacking_function)
import matplotlib.pyplot as plt

from scipy.stats import chisquare



def freqs(bornes, to_check):

    lower, upper = np.ones(len(to_check)) * bornes[0], np.ones(len(to_check)) * bornes[1]

    return sum((lower <= to_check) * (to_check < upper))



plt.hist(demand_levels, 10)

slices = np.concatenate([np.arange(100,200,10), np.arange(110,210,10)]).reshape(2,10).T

fobs = np.apply_along_axis(freqs, 1, slices, demand_levels)

print("P-value: {}".format(chisquare(fobs)[1]))
import pandas as pd

from scipy.stats import norm



score_me_result, function = 7577, pricing_function

macro_iters, micro_iters = 100, 200

params = pd.DataFrame({'days': [100,14,2,1],

                       'tickets': [100,50,20,3],

                       'avg_revenue': np.zeros(4)})

results = np.array([])



for k in np.arange(macro_iters):

    for idx, row in params.iterrows():

        total_revenue = 0

        days, tickets = int(row.days), int(row.tickets)

        for i in np.arange(micro_iters):

            total_revenue += simulate_revenue(days_left=days, tickets_left=tickets, pricing_function=function, verbose=False)

        avg_this_run = total_revenue / micro_iters

        params.avg_revenue.iat[idx] = avg_this_run

    this_result = np.mean(params.avg_revenue)

    print(str(k) + ": $" + str(round(this_result, 2)), end=" ... ")

    results = np.append(results, this_result)

print("\nAverage over {} runs is ${}, compared to ${}".format(macro_iters, np.mean(results), score_me_result))



sd = np.var(results)**0.5

z = abs(score_me_result - np.mean(results)) / sd

p_value = (1 - norm.cdf(z)) * 2

print("The standard deviation of the result is {}, and the z-score is {}.".format(sd, z))

print("The probability of an outcome more extreme than the the score_me() outcome is {}%".format(round(p_value*100,4)))

print("Max result: ${}; {}% of test results >= score_me_result".format(round(max(results),2), round(100*sum(results>=score_me_result)/len(results),2)))
optimized_prices, start_idx, n_sim = {}, 0, 0



def optimizer_with_crystal_ball(sim_number):

    global optimized_prices, start_idx

    if sim_number < 200: ndays, tickets = 100, 100

    elif sim_number < 400: ndays, tickets = 14, 50

    elif sim_number < 600: ndays, tickets = 2, 20

    else: ndays, tickets = 1, 3

    demand = np.array(demand_levels[start_idx:(start_idx+ndays)])

    # Look for maximum price that will clear the inventory

    price = max(demand) - 1

    while(True):

        cleared_inventory = sum(np.maximum(0, np.floor(demand - price)))

        if cleared_inventory < tickets: price -= 1

        else: break

    prices = np.ones(ndays) * price

    # Make sure we don't leave a few pennies on the table on days we make sales

    prices = np.maximum(prices, demand - np.floor(demand - prices))

    # Shift sales around to maximize revenue

    while(True):

        marginal_gains = np.maximum(0, demand - prices - 1)

        source = marginal_gains.argmax()

        rel_opportunity_costs = np.maximum(0, demand - prices) + np.floor(prices[source] - np.minimum(demand, prices))

        target = rel_opportunity_costs.argmin()

        adjusted_prices = prices.copy()

        adjusted_prices[source] += 1

        adjusted_prices[target] = min(prices[target], demand[target]) - 1

        if expected_revenue(demand, prices) >= expected_revenue(demand, adjusted_prices): break

        else: prices = adjusted_prices.copy()

    optimized_prices[sim_number] = prices

    # The previous logic seems to contain an error, as it sometimes undershoots prices

    while sum(np.maximum(0, demand - prices)) > tickets:

        prices[prices.argmin()] += 1        

    # Since the random numbers are sequentially generated, we need to find how many days this simulation will last,

    # as redundant demand values get used at the beginning of the following scenario

    cumul = 0

    for i in range(0,ndays):

        cumul += max(0, np.floor(demand[i] - prices[i]))

        if cumul >= tickets:

            start_idx += i + 1

            break

            

def hacked_price(days_left, tickets_left, demand_level):

    global n_sim, prices

    prices = optimized_prices[n_sim][::-1]

    price = prices[days_left - 1]

    if (days_left == 1 or np.floor(demand_level - price) >= tickets_left):

        n_sim += 1

    return price



for i in range(800):

    optimizer_with_crystal_ball(i)

    

score_me(hacked_price)