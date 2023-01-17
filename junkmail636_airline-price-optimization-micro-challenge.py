import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me
def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    price = demand_level - 10

    return price
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
import math

import numpy as np

NumDaysMax    =100 #index-size for number of days

NumTicketsMax =101 #index-size for number of tickets, 0-100 including 0 and 100

DemandLevelMin=100 

DemandLevelMax=200

NumDemandLevel=DemandLevelMax-DemandLevelMin+1 #index-size for DemandLevel, 100-200 includiong 100 and 200



OptimizedRevenueAvg=np.zeros((NumDaysMax,NumTicketsMax+1))                #stores best revenue averaged over all demand_levels

OptimizedPrice     =np.zeros((NumDaysMax,NumTicketsMax+1,NumDemandLevel)) #stores optimized price



print("preparing lookup tables for NumDaysMax=",NumDaysMax)

for ndays in np.arange(0,NumDaysMax):

    print('{0}/{1}\r'.format(ndays+1,NumDaysMax),end="")

    for ntickets in np.arange(0,NumTicketsMax):

        for demand_level in np.arange(DemandLevelMin,DemandLevelMax+1):

            #Should never sell more than demand_level/2 tickets

            num_tickets_MaxSell=math.floor(min(ntickets,demand_level/2))



            if ndays==0:

                best_price      =demand_level-num_tickets_MaxSell

                best_revenue    =best_price*num_tickets_MaxSell

            else:

                #find out what number of tickets sold today maximizes the expected-revenue

                num_tickets =(np.arange(0,num_tickets_MaxSell+1)).astype(int)

                price       =demand_level-num_tickets

                revenue     =price*num_tickets + OptimizedRevenueAvg[ndays-1][ntickets-num_tickets]



                best_num_tickets=np.argmax(revenue)

                best_price      =price  [best_num_tickets]

                best_revenue    =revenue[best_num_tickets]



            demand=demand_level-DemandLevelMin #index for demand level

            OptimizedPrice     [ndays][ntickets][demand] =best_price

            OptimizedRevenueAvg[ndays][ntickets]        +=best_revenue/NumDemandLevel

print("\nDone")



def pricing_function(days_left, tickets_left, demand_level):

    return OptimizedPrice[days_left-1][math.floor(tickets_left)][math.floor(demand_level-DemandLevelMin)]

score_me(pricing_function)