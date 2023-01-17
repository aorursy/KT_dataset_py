import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    price = demand_level - 10
    return price
import sys
loop_range=range(-1,1)
max_result = sys.float_info.min
best_params = {'x1': 0, 'x2':0, 'x3':0, 'x4':0}
for x1 in loop_range:
    for x2 in loop_range:
        for x3 in loop_range:
            for x4 in loop_range:
                def get_pricing_function():
                    def pricing_function(days_left, tickets_left, demand_level):
                        """Sample pricing function"""
                        price = x1*days_left + x2*tickets_left + x3*demand_level + x4
                        return price
                    return pricing_function
                result=simulate_revenue(days_left=7, tickets_left=50, pricing_function=get_pricing_function(), verbose=True)
                if (result > max_result):
                    max_result = result
                    best_params = {'x1': x1, 'x2':x2, 'x3':x3, 'x4':x4}
                print("result = %s" % result)
print('max_result = %f' % max_result)               
print('best_params = %s' % best_params)
simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)