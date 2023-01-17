import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import os



datapath="../input/taaiv2dip"

temp_data = pd.read_csv(datapath+ '/daily-minimum-temperatures-in-me.csv', header=0)
temp_data.head()
type(list(temp_data["Date"].values)[0])
temp_data["Date"] = pd.to_datetime(temp_data["Date"], format="%Y-%m-%d")
type(list(temp_data["Date"].values)[0])
temp_data = temp_data.sort_values(by=["Date"])
import matplotlib.pylab as plt

%matplotlib inline
plt.plot(temp_data["Date"], temp_data["Daily minimum temperatures in Melbourne, Australia, 1981-1990"])

plt.title("Dagelijkse minimum temperatuur in Melbourne, Australia, 1981-1990")

plt.xlabel("Datum")

plt.ylabel("Temperatuur")

plt.show()
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(temp_data["Daily minimum temperatures in Melbourne, Australia, 1981-1990"])

plt.show()
plot_acf(temp_data["Daily minimum temperatures in Melbourne, Australia, 1981-1990"], lags=500)

plt.show()
plot_acf(temp_data["Daily minimum temperatures in Melbourne, Australia, 1981-1990"], lags=500, alpha=.01)

plt.show()
airline_data = pd.read_csv(datapath+'/international-airline-passengers.csv', header=0)

airline_data.head()
airline_data["Month"] = pd.to_datetime(airline_data["Month"], format="%Y-%m")



plt.plot(airline_data["Month"], airline_data["International airline passengers: monthly totals in thousands"])

plt.title("International Airline Passengers Monthly Totals in Thousands")

plt.xlabel("Date")

plt.ylabel("Monthly Totals (Thousands)")

plt.show()
plot_acf(airline_data["International airline passengers: monthly totals in thousands"], lags=50)

plt.show()
gift_data = pd.read_csv(datapath+'/all_data_gift_certificates.csv', header=0)

gift_data.head()
gift_data["BeginTime"] = pd.to_datetime(gift_data["BeginTime"], format="%Y-%m-%d %H:%M:%S")

plt.plot(gift_data["BeginTime"], gift_data["Count"])

plt.title("Aantal keer dat een 'Cadeaukaart' werd geselecteerd. ")

plt.xlabel("Datum")

plt.ylabel("Aantal")

plt.show()
plot_acf(gift_data["Count"], lags=400)

plt.show()