import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt
data = pd.read_csv("../input/cost_revenue_clean.csv")
data.describe()
X = DataFrame(data, columns=['production_budget_usd'])

y = DataFrame(data, columns=['wordwide_gross_usd'])
plt.figure(figsize=(10,6))

plt.scatter(X,y)

plt.title('Film Cost vs Global Revenue')

plt.xlabel('Production Budet $')

plt.ylabel('Worldwide Gross $')

plt.show()