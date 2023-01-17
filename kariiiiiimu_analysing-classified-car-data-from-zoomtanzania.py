import pandas as pd
import numpy as np

autos = pd.read_csv(r"../input/ZoomTanzania Used Cars.csv", index_col=0)
autos.isnull().sum()
autos = autos[autos.price_currency == 'TSh']
autos.head()
autos.price_value = [ v if type(v) is not str else float(v.replace(',', '')) for v in autos.price_value]
autos.year = [ int(v) for v in autos.year]
print(autos["price_currency"].unique())
print(autos["posted_weekday"].unique())
print(autos["four_wheel_drive"].unique())
print(autos["delivery_offered"].unique())
print(autos["price_negotiable"].unique())
print(autos["transmission"].unique())
print(autos["location"].unique())

autos.drop(["id", "url", "ad_id", "fetched", "description", "price_currency", 
            "mileage", "import_duty_paid", "current_location"], axis=1, inplace=True)
autos.info()
autos.describe()
autos.price_value = autos.price_value.astype(np.uint32)
autos.year = autos.year.astype(np.uint8)
autos.posted_day = autos.posted_day.astype(np.uint8)
autos.page = autos.page.astype(np.uint8)
autos.info()
print(autos.price_value.value_counts().sort_index().head())
print(len(autos.price_value.unique()))
c_prices = autos.price_value.copy()
s_prices = c_prices.sort_values(ascending=False)
s_prices.index = autos.index
print(s_prices.head())

