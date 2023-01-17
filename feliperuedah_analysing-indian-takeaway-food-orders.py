import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
r1 = pd.read_csv('../input/19560-indian-takeaway-orders/restaurant-1-orders.csv', parse_dates=['Order Date'])
r1.head()
r1.info()
print('Number of orders: ', len(r1['Order Number'].unique()))
item_freq = r1.groupby('Item Name').agg({'Quantity': 'sum'})
item_freq = item_freq.sort_values(by=['Quantity'])
top_20 = item_freq.tail(20)
top_20.plot(kind="barh", figsize=(16,8))
plt.title('Top 20 sold items')
# Lista de todos los items
all_items = list(r1['Item Name'].unique())

# Tabla de asociaciones
associations = pd.DataFrame(index=all_items, columns=all_items)
associations.fillna(0, inplace=True)
associations.iloc[:4,:4]
orders = r1.groupby('Order Number')['Item Name'].apply(lambda x: ','.join(x)).reset_index()
orders.rename(columns={'Item Name': 'Order'}, inplace=True)
orders['Order'] = orders['Order'].str.split(',')
orders.head(20)
# Popular la tabla
for order in orders['Order']:
    associations.loc[order, order] += 1
associations.iloc[:4, :4]
associations_top = associations.loc[list(top_20.index), list(top_20.index)]

for i in range(associations_top.shape[0]):
    for j in range(i, associations_top.shape[0]):
        associations_top.iloc[i, j] = 0
        
associations_top.iloc[:5, :5]
plt.figure(figsize=(12,8))
plt.title('Common sold together items')
sns.heatmap(associations_top, cmap="Greens", annot=False)
# Añadir columna con la hora de la orden
r1['hour'] = r1['Order Date'].dt.hour
r1.sample(5)
# Agregar columna con la fecha
r1['date'] = r1['Order Date'].dt.strftime('%y/%m/%d')
r1.sample(5)
def avg_hour(hour):
    by_hour = r1[r1['hour'] == hour]
    avg = len(by_hour['Order Number'].unique()) / len(r1['date'].unique())
    return avg

hours = pd.DataFrame(sorted(r1['hour'].unique()))
hours.rename(columns={0:'hour'}, inplace=True)
hours['Average orders'] = hours['hour'].apply(avg_hour)
hours.set_index('hour', inplace=True)
hours.head()
hours.plot.bar(figsize=(11,6), rot=0)
plt.xlabel('Hour')
plt.title('Average number of orders by hour of the day')
# Columna con el nombre del día
r1['day'] = r1['Order Date'].dt.day_name()
r1.sample(5)
def by_day(day):
    data_day = r1[r1['day'] == day]
    avg = len(data_day['Order Number'].unique()) / len(data_day['date'].unique())
    return(avg)

days = pd.DataFrame(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
days.rename(columns={0: 'day'}, inplace=True)
days['avg_orders'] = days['day'].apply(by_day)
days
plt.bar(days['day'], days['avg_orders'])
plt.xlabel('Day of week')
plt.ylabel('Average number of orders')
plt.title('Average orders by day of week')
plt.xticks(rotation=90)
print('Primera venta: ', r1['Order Date'].min())
print('última venta: ', r1['Order Date'].max())
import datetime

months = []

for year in range(2015, 2020):
    for month in range(1, 13):
        d = datetime.date(year, month, 1)
        months.append(d)

monthly = pd.DataFrame(months)
monthly.rename(columns={0: 'month'}, inplace=True)
monthly.head()
def sales_month(date):
    year_month = date.strftime('%y/%m')
    data = r1[r1['date'].str[:5] == year_month].copy()
    total = (data['Quantity'] * data['Product Price']).sum()
    return(total)

monthly['total'] = monthly['month'].apply(sales_month)
monthly.head()
plt.plot(monthly['month'], monthly['total'])
plt.xlabel('Date')
plt.ylabel('Total sales (USD)')
plt.title('Total monthly sales')
monthly[monthly['month'] >= datetime.date(2019, 1, 1)]
order_total = r1[['Order Number', 'Quantity', 'Product Price']].copy()
order_total['total'] = order_total['Quantity'] * order_total['Product Price']

# Agregar el precio de la orden
order_totals = order_total.groupby('Order Number').agg({'total': 'sum'})
plt.boxplot(order_totals['total'])
plt.title('Order price distribution')
p_95 = order_totals['total'].describe(percentiles=[0.95])['95%']
print('El 95% de las ordenes son menores o iguales a {percentile} USD'.format(percentile=p_95))
plt.boxplot(order_totals[order_totals['total'] < 63]['total'])
plt.title('Order total USD')
plt.ylabel('USD')
sns.distplot(order_totals[order_totals['total'] < 63], bins=20)
plt.title('Order price distribution')