# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import calendar
import random
import numpy
import uuid

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
products = {
  'Kievsky cake Roshen': [279.00, 120],
  'Golden Key cake': [154.00, 90],
  'Cake Prague': [119.00, 60],
  'Millennium White': [21.21, 11],
  'Svitoch White': [33.33, 5],
  'Milka White': [31.22, 13],
  'Candies Kiev Evening': [135.00, 47],
  'Candies Chocolate Cherry': [52.99, 190],
  'Assortment Elegant': [58.60, 180]
  }

columns = ['Order ID', 'Product', 'Quantity Ordered', 
           'Price Each', 'Order Date', 'Purchase Address']

def generate_random_time(month):
  day = generate_random_day(month)
  if random.random() < 0.5:
    date = datetime.datetime(2019, month, day,12,00)
  else:
    date = datetime.datetime(2019, month, day,20,00)
  time_offset = numpy.random.normal(loc=0.0, scale=180)
  final_date = date + datetime.timedelta(minutes=time_offset)
  return final_date.strftime("%m/%d/%y %H:%M")

def generate_random_day(month):
  day_range = calendar.monthrange(2019,month)[1]
  return random.randint(1,day_range)

def generate_random_address():
  street_names = ['Tarasa Shevchenka','Ivana Mazepy', 'Ivana Boguna']
  cities = ['Kyiv', 'Dnepr', 'Lviv']
  weights = [9,4,5]
  zips = ['01000', '49000', '79000']
  state = ['Kyivska', 'Dnipropetrovska', 'Lvivska']

  street = random.choice(street_names)
  index = random.choices(range(len(cities)), weights=weights)[0]

  return f"{random.randint(1,99)} {street} St, {cities[index]}, {state[index]} {zips[index]}"

def create_data_csv():
  pass

def write_row(order_number, product, order_date, address):
  product_price = products[product][0]
  quantity = numpy.random.geometric(p=1.0-(1.0/product_price), size=1)[0]
  output = [order_number, product, quantity, product_price, order_date, address]
  return output

if __name__ == '__main__':
  order_number = 141234
  for month in range(1,13):
    if month <= 10:
      orders_amount = int(numpy.random.normal(loc=1200, scale=400))
    elif month == 11:
      orders_amount = int(numpy.random.normal(loc=2000, scale=300))
    else: # month == 12
      orders_amount = int(numpy.random.normal(loc=2600, scale=300))

    product_list = [product for product in products]
    weights = [products[product][1] for product in products]

    df = pd.DataFrame(columns=columns)
    print(orders_amount)

    i = 0
    while orders_amount > 0:

      address = generate_random_address()
      order_date = generate_random_time(month)

      product_choice = random.choices(product_list, weights)[0]
      df.loc[i] = write_row(order_number, product_choice, order_date, address)
      i += 1

      # Add some items to orders with random chance
      if product_choice == 'Kievsky cake Roshen':
        if random.random() < 0.15:
          df.loc[i] = write_row(order_number, "Millennium White", order_date, address)
          i += 1
        if random.random() < 0.05:
          df.loc[i] = write_row(order_number, "Svitoch White", order_date, address)
          i += 1

        if random.random() < 0.07:
          df.loc[i] = write_row(order_number, "Milka White", order_date, address)
          i += 1 

      elif product_choice == "Golden Key cake" or product_choice == "Cake Prague":
        if random.random() < 0.18:
          df.loc[i] = write_row(order_number, "Candies Kiev Evening", order_date, address)
          i += 1
        if random.random() < 0.04:
          df.loc[i] = write_row(order_number, "Candies Chocolate Cherry", order_date, address)
          i += 1
        if random.random() < 0.07:
          df.loc[i] = write_row(order_number, "Assortment Elegant", order_date, address)
          i += 1 

      if random.random() <= 0.02:
        product_choice = random.choices(product_list, weights)[0]
        df.loc[i] = write_row(order_number, product_choice, order_date, address)
        i += 1

      if random.random() <= 0.002:
        df.loc[i] = columns
        i += 1

      if random.random() <= 0.003:
        df.loc[i] = ["","","","","",""]
        i += 1

      order_number += 1
      orders_amount -= 1

    month_name = calendar.month_name[month]
    df.to_csv(f"Sales_{month_name}_2019.csv", index=False)
    print(f"{month_name} Complete")
import zipfile
import os
import sys

#Create folder for files
!mkdir ./SalesData
!mv ./*.csv ./SalesData

#zip folder
zipname = 'synthetic_sales_data'
def zipfolder(foldername, target_dir):            
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])

zipfolder(zipname, '/kaggle/working')
!ls -lh ./SalesData
import pandas as pd
path = "./SalesData"
files = [file for file in os.listdir(path) if not file.startswith('.')] # Ignore hidden files

all_months_data = pd.DataFrame()

for file in files:
    current_data = pd.read_csv(path+"/"+file)
    all_months_data = pd.concat([all_months_data, current_data])
    
all_months_data.to_csv("all_data.csv", index=False)
df_raw = pd.read_csv('all_data.csv')
df_raw
df_raw['Order ID'].unique()
df_raw['Product'].unique()
df_raw['Quantity Ordered'].unique()
# Show problems data
df_raw[df_raw.isna().any(axis=1)]
data=df_raw.dropna()
data.shape
df = data[data['Price Each']!='Price Each']
df
df['Quantity Ordered'].unique()
df['Quantity Ordered'] = df['Quantity Ordered'].astype('int')
df['Quantity Ordered'].unique()
df
# FACEST
!pip install facets-overview

# set the sprite_size based on the number of records in dataset,
# larger datasets can crash the browser if the size is too large (>50000)
#sprite_size = 32 if len(df.index)>50000 else 64

jsonstr = df.to_json(orient='records')
# Display the Dive visualization for the training data.
from IPython.core.display import display, HTML

jsonstr = df.to_json(orient='records')
HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="600"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#elem").data = data;
        </script>"""
html = HTML_TEMPLATE.format(jsonstr=jsonstr)
display(HTML(html))