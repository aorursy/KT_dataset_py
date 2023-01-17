import sys

sys.path.append('../input')

from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):

    """Sample pricing function"""

    price = demand_level - 10

    return price
import numpy as np



class pricing_function_type:

    """ lookup-table based pricing function"""

    

    def __init__(self, 

                 max_days_left     = 100,

                 max_tickets_left  = 100,

                 min_ticket_demand = 100, 

                 max_ticket_demand = 200

                ):

        

        self.min_days_left     = 0

        self.max_days_left     = max_days_left

        self.n_days_left       = self.max_days_left - self.min_days_left + 1

        self.days_left         = np.arange(self.min_days_left, self.max_days_left + 1)



        self.min_tickets_left  = 0

        self.max_tickets_left  = max_tickets_left

        self.n_tickets_left    = self.max_tickets_left - self.min_tickets_left + 1

        self.tickets_left      = np.arange(self.min_tickets_left, self.max_tickets_left + 1)



        self.min_ticket_demand = min_ticket_demand

        self.max_ticket_demand = max_ticket_demand

        self.n_ticket_demand   = self.max_ticket_demand - self.min_ticket_demand + 1

        self.ticket_demand     = np.arange(self.min_ticket_demand, self.max_ticket_demand + 1)



        self.min_ticket_price  = min_ticket_demand // 2 + min_ticket_demand % 2

        self.max_ticket_price  = max_ticket_demand

        self.n_ticket_price    = self.max_ticket_price - self.min_ticket_price + 1

        self.ticket_price      = np.arange(self.min_ticket_price, self.max_ticket_price + 1)



        self.best_ticket_price      = np.zeros((self.n_days_left, self.n_tickets_left, self.n_ticket_demand), dtype = int)

        self.expected_total_revenue = np.zeros((self.n_days_left, self.n_tickets_left))



        # # uncomment, if one can still sell tickets when days_left == 0 

        # self.best_ticket_price     [0] = self.get_best_ticket_price_single_day     (self.tickets_left[..., None], self.ticket_demand)

        # self.expected_total_revenue[0] = self.get_expected_total_revenue_single_day(self.tickets_left)



        for d in range(1, self.n_days_left):



            revenue_today          = self.get_revenue_single_day     (self.tickets_left[..., None, None], self.ticket_demand[...,None], self.ticket_price)

            tickets_left_tomorrow  = self.get_tickets_left_single_day(self.tickets_left[..., None, None], self.ticket_demand[...,None], self.ticket_price)

            total_revenue_today    = revenue_today + self.expected_total_revenue[d - 1, tickets_left_tomorrow]



            best_i                    = np.argmax(total_revenue_today, axis = 2)

            self.best_ticket_price[d] = self.ticket_price[best_i]



            best_total_revenue_today       = np.amax(total_revenue_today, axis = 2)

            self.expected_total_revenue[d] = np.average(best_total_revenue_today, axis = 1)



    

    def get_tickets_sold_single_day(self, tickets_left, ticket_demand, ticket_price):

        return np.clip(ticket_demand - ticket_price, 0, tickets_left)  



    

    def get_tickets_left_single_day(self, tickets_left, ticket_demand, ticket_price):

        return np.clip(tickets_left - ticket_demand + ticket_price, 0, tickets_left)  



    

    def get_revenue_single_day(self, tickets_left, ticket_demand, ticket_price):

        return ticket_price * np.clip(ticket_demand - ticket_price, 0, tickets_left)



    

    def get_best_ticket_price_single_day(self, tickets_left, ticket_demand):

        return np.maximum(ticket_demand // 2 + ticket_demand % 2, ticket_demand - tickets_left)

    



    def get_best_revenue_single_day(self, tickets_left, ticket_demand):

        p = self.get_best_ticket_price_single_day(tickets_left, ticket_demand)

        return p * (ticket_demand - p)





    def get_expected_revenue_single_day(self, tickets_left):

        revenue = get_best_revenue_single_day(np.expand_dims(tickets_left, -1), self.ticket_demand)

        expected_revenue = np.average(revenue, axis = -1)

        return expected_revenue

    

    

    def get_best_ticket_price(self, days_left, tickets_left, ticket_demand):

        

        # print('days_left =', days_left)

        # print('tickets_left =', tickets_left)

        # print('ticket_demand =', ticket_demand)

        # 

        # print('type(days_left) =', type(days_left))

        # print('type(tickets_left) =', type(tickets_left))

        # print('type(ticket_demand) =', type(ticket_demand))

        

        # was expecting ticket_demand to be integer, but the sim. environment sometimes provides floats

        if isinstance(days_left, float):

            days_left = int(days_left)

        if isinstance(tickets_left, float):

            tickets_left = int(tickets_left)

        if isinstance(ticket_demand, float):

            ticket_demand = int(ticket_demand)

        

        i_day           = days_left     - self.min_days_left

        i_tickets_left  = tickets_left  - self.min_tickets_left

        i_ticket_demand = ticket_demand - self.min_ticket_demand

        return self.best_ticket_price[i_day, i_tickets_left, i_ticket_demand]

    

    

    def get_expected_total_revenue(self, days_left, tickets_left):

        i_day           = days_left     - self.min_days_left

        i_tickets_left  = tickets_left  - self.min_tickets_left

        return self.expected_total_revenue[i_day, i_tickets_left]



    

    def get_best_tickets_sold(self, days_left, tickets_left, ticket_demand):

        best_ticket_price = self.get_best_ticket_price(days_left, tickets_left, ticket_demand)

        best_tickets_sold = self.get_tickets_sold_single_day(tickets_left, ticket_demand, best_ticket_price)

        return best_tickets_sold



    

    def __call__(self, days_left, tickets_left, ticket_demand):

        return self.get_best_ticket_price(days_left, tickets_left, ticket_demand)





my_pricing_function = pricing_function_type()    

simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(my_pricing_function)