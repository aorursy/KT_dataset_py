#Varissara Tangsajjanuraksa ID:5811095
import pandas as pd

raw_csv = pd.read_csv('../input/Lemonade.csv')
print(raw_csv.shape)
#Show average sales
sales = raw_csv['Sales']
avg_sale = sales.mean()
print('Average Sales: ', str(avg_sale))
print(sales.shape)
#Show records whose sales are lower than average
lower_sales = raw_csv.loc[raw_csv['Sales'] < avg_sale]
lower_sales
import matplotlib.pyplot as plt

#Show a scatter plot of sales and temperature
temperatures = raw_csv['Temperature']
plt.scatter(sales, temperatures)
plt.xlabel("Sales")
plt.ylabel("Temperature")

plt.show()
sales_by_day = raw_csv.groupby(['Day'],as_index = False)['Sales'].mean()
print(sales_by_day)
print(sales_by_day.columns)
sales_by_day_ordered = [
    sales_by_day.loc[sales_by_day['Day']=='Sunday']['Sales'].values[0],
    sales_by_day.loc[sales_by_day['Day']=='Monday']['Sales'].values[0],
    sales_by_day.loc[sales_by_day['Day']=='Tuesday']['Sales'].values[0],
    sales_by_day.loc[sales_by_day['Day']=='Wednesday']['Sales'].values[0],
    sales_by_day.loc[sales_by_day['Day']=='Thursday']['Sales'].values[0],
    sales_by_day.loc[sales_by_day['Day']=='Friday']['Sales'].values[0],
    sales_by_day.loc[sales_by_day['Day']=='Saturday']['Sales'].values[0]
]
day_ordered = ('Sunday', 'Monday', 'Tuesday', 'Wednesday',
              'Thursday', 'Friday', 'Saturday')

print(sales_by_day_ordered)
#Show a chart of average sales on each day (Sunday to Saturday)..
fig, ax = plt.subplots(figsize=(10,6))
plt.xlabel("Day")
plt.ylabel("Average Sales")
plt.title("Average sales on each day (Sunday to Saturday)")
ax.bar(range(7), sales_by_day_ordered, width=0.5, align='center')
ax.set(xticks=range(7), xticklabels=day_ordered, ylim=[24.25,26])

plt.show()
