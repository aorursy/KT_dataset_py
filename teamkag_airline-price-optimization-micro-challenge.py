import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):

    # if this is the last day, sell all the tickets at max value

    if (days_left==1):

        tickets_to_sell_today=tickets_left

    else:

        # first approach to confirm understanding of the problem

        # determine the average tickets per day if we were to sell an equal number every day regardless of price

        # tickets_avg_per_day = tickets_left / days_left    

        # tickets_to_sell_today = tickets_avg_per_day

        

        # second approach will use a simple formula

        # determine the demand percent as a percentage of the range 100-200, where 150 =0.50, 175=0.75, etc.

        demand_percent=(demand_level-100)/100

        

        # compute remaining tickets to sell today based on an demand_percent raised to the number of days_left

        # this, of course, does not consider overall profitability it is just a manual heuristic

        # note: the .8 exponent was found by manually doing a search of values between 0.1 and 4 

        tickets_to_sell_today = tickets_left * demand_percent ** (days_left ** .8)



    # set the price based on how many tickets_to_sell_today

    price=demand_level-tickets_to_sell_today

   

    return price

simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)