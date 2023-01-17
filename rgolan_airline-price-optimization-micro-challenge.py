import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me



import random 

import statistics



def pricing_function(days_left, tickets_left, demand_level):

    

    """Sample pricing function"""

    #under a KNOWN series of prices - what would be optimal? - lets try n scenarios and pick the median



    n=10

    #maxREV=[0]*n 

    P=[200]*n

    for sim_i in range(n):

        # 100 (n) price scenarios

        simD=[random.randrange(100, 200, 1) for i in range(days_left)] #randomly asigned future demand levels

        simD[0]=int(demand_level) # at 0 I place the actual known demand value of today

        #the best pricing with full information: calculate a margional revenue for a marginal ticket in each period and every time take the highest

        simT=[0]*days_left #a list of tickets optimally sold each day

        simMR=simD.copy() #marginal revenue from the ticket, if sold in a specific day

        #for d_i in range(days_left):

        #    simMR[d_i]=simD[d_i]-1

        for ticket_i in range(int(tickets_left)):

            if max(simMR)>0:

                #maxREV[sim_i]+=max(simMR)

                temp=simMR.index(max(simMR)) # on which day its best to sell this marginal ticket

                simMR[temp]=simMR[temp]-1-simT[temp] # update the marginal revenue for that day

                simT[temp]+=1



        P[sim_i]=simD[0]-simT[0]

    price=int(statistics.median(P))    

    #    price = demand_level - 10

    return price
simulate_revenue(days_left=1, tickets_left=5, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)