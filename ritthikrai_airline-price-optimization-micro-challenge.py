import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    if days_left == 1:
        price = demand_level - tickets_left
    elif days_left == 2:
        if demand_level > 179:
            price = demand_level - tickets_left
        else:
            price = demand_level - tickets_left/2
    else:
        if days_left > 12:
            if demand_level > 186:
                price = demand_level - 8
            else:
                price = demand_level + 1
        elif days_left > 3:
            if tickets_left/days_left > 2.5:
                if demand_level > 169:
                    price = demand_level - 17
                else:
                    price = demand_level + 1
            else:
                if demand_level > 180:
                    price = demand_level - 20
                else:
                    price = demand_level + 1
        else:
            if demand_level > 169:
                price = demand_level - 13
            else:
                #price = demand_level - tickets_left/days_left
                price = demand_level - 1
    return price

simulate_revenue(days_left=14, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)