import numpy as np # Linear algebra
import pandas as pd # For manipulating data
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For plotting
import os # For accessing system files
# Listing all the files
root_dir = '../input/sales-analysis-2019/'
files = [file for file in os.listdir(root_dir)]

# Creating an empty DataFrame for storing all the data
all_months_data = pd.DataFrame() 

# Merging all the CSV files
for file in files:
    df = pd.read_csv(root_dir + file)
    all_months_data = pd.concat([all_months_data, df])
    
# Saving the file to do further analysis
all_months_data.to_csv('all_data.csv', index = False)
all_data = pd.read_csv('all_data.csv')
all_data.head()
# Total NaN in columns
all_data.isnull().sum()
# Dropping the NaN
all_data.dropna(inplace=True)
# Add month column
all_data['Month'] = all_data['Order Date'].str[0:2]
try:
    all_data['Month'] = all_data['Month'].astype(np.int32)
except Exception as e:
    print(e)
# Finding the Or in Month column
temp_df = all_data[all_data['Order Date'].str[0:2] == 'Or']
temp_df.head()
# Removing Or data
all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']
# Againg adding the month column

all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype(np.int32)
# Converting price column to numeric column
all_data['Price Each'] = all_data['Price Each'].astype(np.float64)

# Converting Quantity column to numeric column
all_data['Quantity Ordered'] = all_data['Quantity Ordered'].astype(np.int64)

# Adding sales Columns
all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
# Checking the data type of every column
all_data.info()
# Freuency of the months in sales
results = all_data.groupby('Month').sum()
results
# plotting the sales result

months = range(1,13)
plt.bar(months, results['Sales'])
plt.title('Sales per Month')
plt.xticks(months)
plt.ylabel('Sales in USD')
plt.xlabel('Months')
plt.show();
results['Month'] = results.index

sns.lineplot(x='Month', y='Sales', data=results, color='red', markers='--')

plt.xticks(months)
plt.title('Sales per Month')
plt.xlabel('Months for Sales')
plt.ylabel('Sales in USD($)')
plt.show()
# Adding Cities column

all_data['City'] = all_data['Purchase Address']
all_data['State'] = all_data['Purchase Address']
def adding_city_and_state(a):
    return f"{a.split(',')[1][1:]} ({a.split(',')[2][1:3]})"
 
all_data['City'] = all_data['City'].apply(adding_city_and_state)
all_data['City'].head()
# Grouping via City
results = all_data.groupby('City').sum()
results[results['Sales'] == results['Sales'].max()]
# plotting the resulted sales

cities = [city for city, _ in all_data.groupby('City')]

plt.bar(cities, results['Sales'])
plt.xticks(cities, rotation='vertical')
plt.ylabel('Sales in USD')
plt.xlabel('City Name')
plt.show();
results['City'] = results.index

sns.lineplot(x='City', y='Sales', data=results, color='red')

plt.xticks(cities, rotation='vertical')
plt.title('Sales per City')
plt.xlabel('City for Sales')
plt.ylabel('Sales in USD($)')
plt.show()
# Converting Order Date to date time field using pandas to_datetime

all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])
all_data.head()
# Adding Hours column in our dataset
all_data['Hour'] = all_data['Order Date'].dt.hour

# Adding Minute column in our dataset
all_data['Minute'] = all_data['Order Date'].dt.minute

all_data.head()
hours = [hour for hour, _ in all_data.groupby('Hour')]

plt.plot(hours, all_data.groupby(['Hour']).count())
plt.xticks(hours)
plt.xlabel("Hours ----->")
plt.ylabel("Number of Orders ----->")
plt.grid()
plt.show();
# Capturing all the duplicate Order ID to find out which are boughts in pairs

temp = all_data[all_data['Order ID'].duplicated(keep=False)]
temp.head(10)
# adding Grouped column which will tell us what products are sold with the same Order Id

temp['Grouped'] = temp.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
temp.head()
# Removing the duplicates from the dataset

temp = temp[['Order ID', 'Grouped']].drop_duplicates()
temp.head()
# Counting the frequency of occurence of two products together

from itertools import combinations
from collections import Counter

count = Counter()

for row in temp['Grouped']:
    row_list = list(row.split(','))
    count.update(Counter(combinations(row_list, 2)))
items = []
items_count = []
for key, value in count.most_common(10):
    items.append(str(key))
    items_count.append(int(value))
    print(f"{str(key)[1:-1]} : {str(value)}")

item_len = range(1, len(items)+1)
plt.bar(item_len, items_count)
plt.xticks(item_len, rotation='vertical')
plt.show()
    
# Uploading my notebook to my jovian.ml account
!pip install jovian
import jovian
jovian.commit(project='eda_on_sales_analysis_2019')