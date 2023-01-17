import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

def custom_sort(x):

    for item in x:

                return x[1]



def pricing_function(days_left, tickets_left, demand_level): 

    """Sample pricing function"""

    

    # sell all tickets day before flight

    if days_left == 1:

        price = demand_level - tickets_left

        return price

    

    sims = []

    

    revs = []

    

    risk_factor = 0.93 ** days_left

    

    risk_factor = min(1, risk_factor)

    

    P_x_list = []

    x = 100 # x is the price I am simulating selling my tickets for in the future

    while x < 200:

        my_list = list(range(1, days_left + 1)) # list of the ways in which I can sell all my tickets at x

        for item in my_list:

            

            

            PA = ((200 - (x + (tickets_left / item))) / 100)

            PA = max(0, PA)

            

            P = (PA ** item) * (days_left / item) # the probability of selling at x in one way x 

         

            P = max(0, P)



            P_x_list.append((P)) # list of the probabilities of selling x in each way

            

        P_x = sum(P_x_list) # probability of selling x is the sum of the probability of selling x in each fashion

        #print (x, P_x)

      

        if P_x < risk_factor:

            sims.append((x, P_x)) #want to find the max price guaranteed, which means P_x will equal 1

        

        

        

        P_x_list = []

        

        

        

        x += 1

        

    

    adjusted_price = sims[0][0]

    

    

    if adjusted_price >= demand_level:

        n = 0 

    else:

        premiums = list(range(1,(200 - adjusted_price)))

        

        for premium in premiums:                

            

            n = demand_level - (adjusted_price + premium)

            Erev = (n * (adjusted_price + premium)) + (adjusted_price * (tickets_left - n))           

            

            revs.append((n, Erev))

        

        revs.sort(key = custom_sort, reverse = True)

        

        n = revs[0][0] 

        

        revs = []

    

    price = demand_level - n

    

    return price

   

                 

    

pricing_function(100, 100, 199)    

    

    

    

#simulate_revenue(days_left= 14, tickets_left= 50, pricing_function=pricing_function, verbose=True)

score_me(pricing_function)