import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
register_matplotlib_converters()
date_cols = ["observation_date"]
oil_data = pd.read_excel(r'../input/uscorporateextractionprofits-saudiarabiaproduction/BOOTCAMP_oil_profits_before_tax.xls', parse_dates=date_cols)
oil_data
x = oil_data['observation_date']
y = oil_data['USA Corporate Profits Before Tax']

plt.scatter(x, y)


plt.xlabel('Year')
plt.ylabel('USA Corporate Profits Before Tax')

plt.title('USA Corporate Profits Before Tax by Year')

plt.savefig('USA Corporate Profits Before Tax by Year.png')

plt.show()
x = oil_data['observation_date']
y = oil_data['Saudi_Arabia_Oil_Production']

plt.scatter(x, y)

plt.xlabel('Year')
plt.ylabel('Saudi Arabia Oil Production')

plt.title('Saudi Arabia Oil Production by Year')
plt.savefig('Saudi Arabia Oil Production by Year.png')
plt.show()

x = oil_data['Saudi_Arabia_Oil_Production']
y = oil_data['USA Corporate Profits Before Tax']

plt.scatter(x, y)

plt.xlabel('Saudi Arabia Oil Production')
plt.ylabel('USA Corporate Profits Before Tax')

plt.title('USA Corporate Profits Before Tax by Saudi Arabia Oil Production')

plt.savefig('USA Corporate Profits Before Tax by Saudi Arabia Oil Production.png')

plt.show()
x = oil_data['Saudi_Arabia_Oil_Production']
y = oil_data['USA Corporate Profits Before Tax']
const = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(print_model), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png')
