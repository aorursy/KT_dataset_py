import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

from math import floor

def how_many_ticket(days_left, tickets_left, demand_level):

    for ticket_to_sell in range(0,int(tickets_left)+1):    
        curr_price = demand_level - ticket_to_sell
        better_gap = max(0,199-curr_price)
        avg_better_expence_token_number = better_gap*(better_gap+1)/2 /101 # expectect number of ticket that I could sell tomorrow at higher price
        avg_better_expence_token_number_total = avg_better_expence_token_number * (days_left-1) # expectect number of ticket that I could sell on the left days at higher price.

        if avg_better_expence_token_number_total > tickets_left:
            return ticket_to_sell # It's better to sell my ticket other days
        
        best_number_to_sell = tickets_left-round(avg_better_expence_token_number_total) # number of ticket that It's better to sell now
        avail_number_to_sell = demand_level-curr_price # number of ticket that I can sell now
        
        if avail_number_to_sell > best_number_to_sell:
            return ticket_to_sell # I have enought ticket to sell at that price. Otherwise It's better to decrease the price and sell one ticket more (Spoiler: this will decrease the best_number _to_sell)
    return ticket_to_sell
#        max_sell_tiket = min(demand_level-curr_price, tickets_left-avg_better_expence_token_number_total)

                
def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
        
    ticket_to_sell = how_many_ticket(days_left, tickets_left, demand_level)
    price = demand_level - ticket_to_sell - 0.00001

    
    if days_left == 1:
        price = demand_level - tickets_left
    return price
simulate_revenue(days_left=100, tickets_left=100, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)