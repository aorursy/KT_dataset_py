import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

import random

def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    return random.randint(1,500)
def hasNumbers(inputString):

    return any(char.isdigit() for char in inputString)



def strip_string(output):

    data_from_string = list()

    lines = output.split('\n')

    for line in lines:

        if hasNumbers(line):

            days_left = line.split(' ')[0]

            started_with_seats = line.split('with ')[1].split(' seats.')[0]

            demand_level = line.split('level: ')[1].split('. Price set')[0]

            price_ticket = line.split('to $')[1].split('. Sold')[0]

            tickets_sold = line.split('Sold ')[1].split(' tickets.')[0]

            daily_revenue = line.split('Daily revenue is ')[1].split('. Total revenue')[0]

            total_rev = line.split('date is ')[1].split('. ')[0]

            seats_remaining = line.split('date is ')[1].split('. ')[1].split(' seats')[0]

            data_from_string.append([days_left,started_with_seats,demand_level,price_ticket,tickets_sold,daily_revenue,total_rev,seats_remaining])

        else:

            break

        

    return data_from_string

        
from contextlib import redirect_stdout

from io import StringIO

import pandas as pd

all_data_list = list()

days_numbers = [100,14,2,1]

tickets = [100,50,20,3]

for day_number in days_numbers:

    for ticket in tickets: 

        for i in range(500000):

            out_buffer = StringIO()

        #     sys.stdout.write("\rDoing thing %i" % i)

        #     sys.stdout.flush()

            with redirect_stdout(out_buffer):

                simulate_revenue(days_left=day_number, tickets_left=ticket, pricing_function=pricing_function, verbose=True)

            out_str = out_buffer.getvalue()

            # print(out_str)



            data = strip_string(out_str)

            all_data_list.append(data)



        all_data = list()

        for data in all_data_list:

            for day in data:

                all_data.append(day)



# print(all_data)

airline_price_data = pd.DataFrame(all_data, 

                                columns=['days_left', 'started_with_seats','demand_level','price_ticket','tickets_sold','daily_revenue','total_rev','seats_remaining'])



# airline_price_data = 0

# count = 0 

# for data in all_data:

#     for day in data:

#         if count != 0:

#             airline_price_data = airline_price_data.append(pd.DataFrame(data, 

#                                 columns=['days_left', 'started_with_seats','demand_level','price_ticket','tickets_sold','daily_revenue','total_rev','seats_remaining']))

# #             count = count + 1

#         else:

#             airline_price_data = pd.DataFrame(data, 

#                                 columns=['days_left', 'started_with_seats','demand_level','price_ticket','tickets_sold','daily_revenue','total_rev','seats_remaining'])

#             count = count + 1

airline_price_data.to_csv('airline_price_week.csv',index=False)

print(airline_price_data.shape)
score_me(pricing_function)