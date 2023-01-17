import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me

import numpy as np
def expected_first_revenues(days_left, tickets_left, Matrix):
    lowerbound = 100
    upperbound = 200
    total = 0
    count = 0
    interval = .1 #It seems like when it's very small, it converges to 7610
    
    if days_left > 1:
      for demand in np.arange(lowerbound, upperbound, interval):
        tickets_sold = 0
        expected_revs = 0
        while tickets_sold <= demand and tickets_sold <= tickets_left:
          q = tickets_sold
          sumly = q  * (demand - q) + Matrix[days_left - 1][tickets_left - q]
          if sumly > expected_revs:
            expected_revs = sumly
          tickets_sold += 1
        total += expected_revs
        count += 1
      Matrix[days_left][tickets_left] = total / count
      return total / count
    if days_left == 1:
      for demand in np.arange(lowerbound, upperbound, interval):
        q = min(demand / 2, tickets_left)
        expected_revs = q  * (demand - q)
        total += expected_revs
        count += 1
      Matrix[days_left][tickets_left] = total / count
      return total / count


def test(days_left, tickets_left, demand_level=0):
  w, h = tickets_left+1, days_left+1;
  Matrix = [[0 for x in range(w)] for y in range(h)] 
  
  for j in range(days_left+1):    
        for i in range(tickets_left+1):
          expected_first_revenues(j, i, Matrix)
  return Matrix #[days_left][tickets_left]

Matrix = test(100,100)

def pricing_function(days_left, tickets_left, demand_level, Matrix = Matrix):
    demand = demand_level
    tickets_sold = 0
    expected_revs = 0
    p = 0
    while tickets_sold <= demand and tickets_sold <= tickets_left:
      q = tickets_sold
      sumly = q  * (demand - q) + Matrix[int(days_left - 1)][int(tickets_left - q)]
      if sumly > expected_revs:
        expected_revs = sumly
        p = demand - q
      tickets_sold += 1
    return p


simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)
score_me(pricing_function)