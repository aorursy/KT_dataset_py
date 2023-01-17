import numpy as np
import matplotlib.pyplot as plt
import math
percent_increase_yr=800 # a recent Tesla metric
no_months=12
starting_value=1.00
value_increase=starting_value+percent_increase_yr/100
exponential=value_increase**(1/no_months)
months=range(0,no_months)
costs=[exponential**month for month in months]
earnings=[value_increase-cost for cost in costs]
print("Costs \n{}".format(np.round(costs,2)))
print("Earnings \n{}".format(np.round(earnings,2)))
total_cost=np.sum(costs)
total_earnings=np.sum(earnings)
print(" Costs ${:0.2f}".format(total_cost))
print(" Earnings ${:0.2f}".format(total_earnings))
print(" Total Value ${:0.2f}".format(total_cost+total_earnings))
roi=total_earnings/total_cost
print("ROI is {:0.2f}%".format(roi*100))
costs_plt=plt.bar(months,costs)
earn_plt=plt.bar(months,earnings,bottom=costs)
plt.legend((costs_plt[0], earn_plt[0]), ('Costs', 'Earnings'))
plt.show()