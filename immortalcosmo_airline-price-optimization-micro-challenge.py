import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me



def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    #The less days, the more we need to sell

    #The less tickets, the less we need to sell

    #The higher the demand, the more we should sell

    if(days_left==1):

        price = demand_level-tickets_left #Last day, must sell all tickets

    else:

        p = (demand_level-100)/100  #Favorability of demand_level, from 0 at 100 demand to 100 at 200 demand

        tentative_price = demand_level - p*tickets_left #Amount of remaining tickets I want sold based on favorability

        c =.008 #Constant that represents how much we value pressure, lower is higher value

        d = 2 #Constant that represents how much pressure we feel from days_left

        pressure = tickets_left/(days_left**d)

        #Low tickets_left = low pressure, high = high pressure. Low days_left is high pressure, high is low pressure.

        price = tentative_price * (1 + c/pressure)

    return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)