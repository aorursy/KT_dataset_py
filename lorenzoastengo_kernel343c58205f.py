%pylab inline

import pandas as pd

pd.set_option('display.max_columns', 5000)

pd.set_option('display.max_rows', 5000)

import cx_Oracle
!pwd
target1=pd.read_pickle('../shared_data/target/target_number_of_plants.pkl')
target1.tail()
result1=pd.DataFrame(columns=target1.columns,index=[20190308,20190309,20190310,20190311,20190312,20190313,20190314])

result1.index.name='date'
result1
target2=pd.read_pickle('../shared_data/target/target_amount_of_energy.pkl')
target2.tail()
result2=pd.DataFrame(columns=target2.columns,index=[20190308,20190309,20190310,20190311,20190312,20190313,20190314])

result2.index.name='date'
result2
target3=pd.read_pickle('../shared_data/target/target_avg_price.pkl')
target3.tail()
result3=pd.DataFrame(columns=target3.columns,index=[20190308,20190309,20190310,20190311,20190312,20190313,20190314])

result3.index.name='date'
result3
real_weather=pd.read_csv('../shared_data/weather/real/weather.csv',sep=';').sort_values('observation_date')
real_weather.head()
forecast_temperature_df=pd.read_csv('../shared_data/weather/forecast/temperature.csv',sep=';')
forecast_temperature_df.tail()
renewable_df=pd.read_csv('../shared_data/renewable_energy/renewable_energy.csv',sep=';')
renewable_df.tail()
availability_df=pd.read_csv('../shared_data/UP_availability/availability.csv',sep=';')
availability_df.head()
energy_costs_df=pd.read_csv('../shared_data/production_cost/production_cost.csv',sep=';')
energy_costs_df.tail()
acc_amount_energy_df=pd.read_csv('../shared_data/volume/volume.csv',sep=';')
acc_amount_energy_df.tail()
price_df=pd.read_csv('../shared_data/price/price.csv',sep=';')
price_df.head()
offers_df=pd.read_csv('../shared_data/offers/offers.csv',sep=';')
offers_df.head()