import numpy as np

import matplotlib.pyplot as plt
protection_cost = 5_000_000



cumulative_protection_cost = [0 for i in range(0,9)]

years_span = 55

for i in range(9,years_span):

    if i%5 == 0:

        cumulative_protection_cost.append(cumulative_protection_cost[i-1]+protection_cost)

    else:

        cumulative_protection_cost.append(cumulative_protection_cost[i-1])
plt.ylabel('Cost')

plt.xlabel('Year')

plt.plot(cumulative_protection_cost, label="Protection Cost")

plt.legend()
flood_cost_protected = 100_000

cumulative_protection_flood_cost = cumulative_protection_cost



for i in range(0,years_span,10):

    cumulative_protection_flood_cost[i] += flood_cost_protected
plt.ylabel('Cost')

plt.xlabel('Year')

plt.plot(cumulative_protection_cost, label="Protection Cost")

plt.plot(cumulative_protection_flood_cost, label="Protection + Flood Cost")

plt.legend()
flood_cost = 10_000_000

cumulative_flood_cost = [0]

for i in range(1,years_span):

    if i%10 == 0:

        cumulative_flood_cost.append(cumulative_flood_cost[i-1]+flood_cost)

    else:

        cumulative_flood_cost.append(cumulative_flood_cost[i-1])
plt.ylabel('Cost')

plt.xlabel('Year')

plt.plot(cumulative_protection_flood_cost, label="Policy 1)")

plt.plot(cumulative_flood_cost, label="Policy 2)")

plt.legend()
cumulative_protection_flood_trade_cost = [0 for i in range(0,8)]

for i in range(8,years_span):

    if (i+2)%10 == 0:

        cumulative_protection_flood_trade_cost.append(cumulative_protection_flood_trade_cost[i-1]+protection_cost)

    else:

        cumulative_protection_flood_trade_cost.append(cumulative_protection_flood_trade_cost[i-1])

    if i%10 == 0:

        cumulative_protection_flood_trade_cost[i] += flood_cost_protected
plt.ylabel('Cost')

plt.xlabel('Year')

plt.plot(cumulative_protection_flood_cost, label="Policy 1)")

plt.plot(cumulative_flood_cost, label="Policy 2)")

plt.plot(cumulative_protection_flood_trade_cost, label="Policy 3)")

plt.legend()
savings_p2_p3 = cumulative_flood_cost[-1] - cumulative_protection_flood_trade_cost[-1]

savings_p1_p3 = cumulative_protection_flood_cost[-1] - cumulative_protection_flood_trade_cost[-1]

print(savings_p1_p3)

print(savings_p2_p3)