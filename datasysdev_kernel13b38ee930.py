import random

def daily_outcome():

    return random.randint(-200,250) / 100

for i in range(1,100):

    print (daily_outcome())
def cum_return(invest,days):

    if days > 0:

        a = daily_outcome()

        if (a > 0):

            daily = 10

        else:

            daily = 20

        invest = daily + invest + (invest * a / 100)

        #print(invest * volatility() / 100)

        #print(invest)

        invest = cum_return(invest, days - 1)

    else:

        print ('Final day:', invest + 10)

        
for i in range(1,100):

    (cum_return(30,260))

#cum_return(10,260)

print(10*260)