import pandas as pd
PATH = '../input/prozorro-public-procurement-dataset/Competitive_procurements.csv'
import warnings
warnings.filterwarnings('ignore')
procs = pd.read_csv(PATH)
procs.head()
procs['lot_announce_month'] = procs.apply(lambda row: row.lot_announce_date[:7], axis = 1) 
lots_per_month = procs.groupby(['lot_announce_month', 'lot_id']).size().groupby('lot_announce_month').count().reset_index()
lots_per_month.columns = ['lot_announce_month', 'number_of_lots']
import matplotlib.pyplot as plt
import numpy as np
%matplotlib notebook

plt.figure(figsize=(12,5))
plt.plot(lots_per_month.lot_announce_month, lots_per_month.number_of_lots)
plt.xticks(['2015-02','2015-12','2016-12',
            '2017-12','2018-12', '2019-12'], 
           ['2015-02','15-12','16-12',
            '17-12','18-12','2019-12'])
plt.yticks([30000, 25000, 20000, 15000, 10000, 5000, 0],
           ['30 000', '25 000', '20 000', '15 000', '10 000', '5 000', 0])
plt.xlabel('month')
plt.ylabel('number of lots')
plt.title('Number of lots')
plt.grid(True)
plt.show()
supps_per_lot = procs.groupby(['lot_announce_month', 'lot_id']).size().groupby('lot_announce_month').agg('mean').reset_index()
supps_per_lot.columns = ['lot_announce_month', 'supps_per_lot']
plt.figure(figsize=(12,5))
plt.plot(supps_per_lot.lot_announce_month, supps_per_lot.supps_per_lot, color='#4f5c8a')
plt.xticks(['2015-02', '2015-12', '2016-12', '2017-12', '2018-12', '2019-12'], 
           ['2015-02', '15-12', '16-12', '17-12', '18-12', '2019-12'])
plt.xlabel('month')
plt.ylabel('supps per lot')
plt.title('Suppliers per lot')
plt.grid(True)
plt.show()
sales = procs[procs.supplier_dummy == 1].groupby('lot_announce_month').agg({'lot_final_value':'sum'}).reset_index()
sales.columns = ['lot_announce_month', 'sales']
procs['savings'] = procs.apply(lambda row: row.lot_initial_value - row.lot_final_value, axis = 1)
savings = procs[procs.supplier_dummy == 1].groupby('lot_announce_month').agg({'savings':'sum'}).reset_index()
plt.figure(figsize=(12,5))
plt.plot(sales.lot_announce_month, sales.sales, color='#395838')
plt.plot(savings.lot_announce_month, savings.savings, color='#4f8a56')
plt.xticks(['2015-02', '2015-12', '2017-02', '2017-12', '2018-11', '2019-12'], 
           ['2015-02', '15-12', '17-02', '17-12', '18-11', '2019-12'])
plt.yticks([50*10**9, 40*10**9, 30*10**9, 20*10**9, 10*10**9, 0],
           ['50B', '40B', '30B', '20B', '10B', '0'])
plt.xlabel('month')
plt.ylabel('UAH')
plt.title('Sales and Savings')
plt.legend(['sales', 'savings'])
plt.grid(True)
plt.show()
plt.figure(figsize=(12,5))
plt.plot(savings.lot_announce_month, savings.savings, color='#4f8a56')
plt.xticks(['2015-02', '2015-12', '2017-02', '2017-12', '2018-11', '2019-12'], 
           ['2015-02', '15-12', '17-02', '17-12', '18-11', '2019-12'])
plt.yticks([5*10**9, 4*10**9, 3*10**9, 2*10**9, 10**9, 0],['5B', '4B', '3B', '2B', '1B', '0'])
plt.xlabel('month')
plt.ylabel('savings (UAH)')
plt.title('Savings')
plt.grid(True)
plt.show()
times_participated = procs.groupby(['participant_name']).filter(lambda row: row['supplier_dummy'].sum() > 0).groupby(['participant_name']).size().reset_index()
times_participated.columns = ['participant_name', 'times_participated']
times_won = procs[procs.supplier_dummy == 1].groupby(['participant_name']).size().reset_index()
times_won.columns = ['participant_name', 'times_won']
total_sales = procs[procs.supplier_dummy == 1].groupby(['participant_name']).agg({'lot_final_value':'sum'}).reset_index()
total_sales.columns = ['participant_name', 'total_sales']
pd.options.display.float_format = '{:,.0f}'.format
pd.options.display.max_colwidth = 80
top_suppliers = pd.DataFrame({'participant_name':times_participated.participant_name,
                                 'times_participated':times_participated.times_participated,
                                 'times_won':times_won.times_won,
                                 'total_sales':total_sales.total_sales})

top_suppliers.sort_values('total_sales', ascending=False).head(20)
procs['market'] = procs.apply(lambda row: row.lot_cpv_2_digs[11:], axis = 1)
top_suppliers_by_market = procs[procs.supplier_dummy == 1].groupby(['market','participant_name']).agg({'lot_final_value':'sum'}).reset_index()
top_suppliers_by_market.columns = ['market','participant_name','total_sales']
market_total_sales = top_suppliers_by_market.groupby(['market']).agg({'total_sales':'sum'}).reset_index()

top_suppliers_by_market = top_suppliers_by_market.groupby(['market']).apply(lambda row: row.nlargest(5,['total_sales'])).reset_index(drop=True)
top_suppliers_by_market['market_share'] = ''

for i in range(len(market_total_sales)):
    for j in range(5):
        top_suppliers_by_market.market_share[5*i+j] = top_suppliers_by_market.total_sales[5*i+j]/market_total_sales.total_sales[i]

def insert_row_(row_number, df, row_value): 
    df1 = df[0:row_number] 
    df2 = df[row_number:] 
    df1.loc[row_number]=row_value 
    df_result = pd.concat([df1, df2]) 
    df_result.index = [*range(df_result.shape[0])] 
    return df_result 

        
for i in range(len(top_suppliers_by_market)+46):
    if (i%6 == 0):
        top_suppliers_by_market = insert_row_(i,  top_suppliers_by_market, ['','','',''])
pd.set_option('display.max_rows', 500)
pd.options.display.max_colwidth = 100
pd.options.display.float_format = '{:,.2f}'.format
top_suppliers_by_market
market_total_sales.sort_values('total_sales', ascending=False)
construction_market = procs[procs.market.isin(['Construction work',
                                               'Construction structures and materials; auxiliary products to construction (except electric apparatus)',
                                               'Electrical machinery, apparatus, equipment and consumables; lighting',
                                               'Architectural, construction, engineering and inspection services'])]

fuel_market = procs[procs.market.isin(['Petroleum products, fuel, electricity and other sources of energy',
                                       'Services related to the oil and gas industry'])]

agricultural_market = procs[procs.market.isin(['Agricultural, farming, fishing, forestry and related products',
                                               'Agricultural, forestry, horticultural, aquacultural and apicultural services',
                                               'Agricultural machinery'])]
construction_market_sales = construction_market[construction_market.supplier_dummy == 1].groupby('lot_announce_month').agg({'lot_final_value':'sum'}).reset_index()
construction_market_sales.columns = ['lot_announce_month', 'construction']
fuel_market_sales = fuel_market[fuel_market.supplier_dummy == 1].groupby('lot_announce_month').agg({'lot_final_value':'sum'}).reset_index()
fuel_market_sales.columns = ['lot_announce_month', 'fuel']
agricultural_market_sales = agricultural_market[agricultural_market.supplier_dummy == 1].groupby('lot_announce_month').agg({'lot_final_value':'sum'}).reset_index()
agricultural_market_sales.columns = ['lot_announce_month', 'agricultural']
plt.figure(figsize=(12,5))
plt.plot(construction_market_sales.lot_announce_month, construction_market_sales.construction, color='#ac0000')
plt.plot(fuel_market_sales.lot_announce_month, fuel_market_sales.fuel, color='#007dac')
plt.plot(agricultural_market_sales.lot_announce_month, agricultural_market_sales.agricultural, color='#007810')
plt.xticks(['2015-02', '2015-12', '2016-12', '2017-12', '2018-12', '2019-12'], 
           ['2015-02', '15-12', '16-12', '17-12', '18-12', '2019-12'])
plt.yticks([35*10**9, 30*10**9, 25*10**9, 20*10**9, 15*10**9, 10*10**9, 5*10**9, 0],
           ['35B', '30B', '25B', '20B', '15B', '10B', '5B', '0'])
plt.xlabel('month')
plt.ylabel('UAH')
plt.title('Sales of market')
plt.legend(['construction', 'fuel', 'agricultural'])
plt.grid(True)
plt.show()
construction_market_savings = construction_market[construction_market.supplier_dummy == 1].groupby('lot_announce_month').agg({'savings':'sum'}).reset_index()
construction_market_savings.columns = ['lot_announce_month', 'construction']
fuel_market_savings = fuel_market[fuel_market.supplier_dummy == 1].groupby('lot_announce_month').agg({'savings':'sum'}).reset_index()
fuel_market_savings.columns = ['lot_announce_month', 'fuel']
agricultural_market_savings = agricultural_market[agricultural_market.supplier_dummy == 1].groupby('lot_announce_month').agg({'savings':'sum'}).reset_index()
agricultural_market_savings.columns = ['lot_announce_month', 'agricultural']
plt.figure(figsize=(12,5))
plt.plot(construction_market_sales.lot_announce_month, construction_market_sales.construction, color='#ac0000')
plt.plot(construction_market_savings.lot_announce_month, construction_market_savings.construction, color='#ab4d4d')
plt.xticks(['2015-02', '2015-12', '2016-12', '2017-12', '2018-12', '2019-12'], 
           ['2015-02', '15-12', '16-12', '17-12', '18-12', '2019-12'])
plt.yticks([35*10**9, 30*10**9, 25*10**9, 20*10**9, 15*10**9, 10*10**9, 5*10**9, 0],
           ['35B', '30B', '25B', '20B', '15B', '10B', '5B', '0'])
plt.xlabel('month')
plt.ylabel('UAH')
plt.title('Sales and savings - Construction')
plt.legend(['sales', 'savings'])
plt.grid(True)
plt.show()
plt.figure(figsize=(12,5))
plt.plot(fuel_market_sales.lot_announce_month, fuel_market_sales.fuel, color='#007dac')
plt.plot(fuel_market_savings.lot_announce_month, fuel_market_savings.fuel, color='#4d86ab')
plt.xticks(['2015-12', '2016-12', '2017-12', '2018-12', '2019-12'], 
           ['15-12', '16-12', '17-12', '18-12', '2019-12'])
plt.yticks([ 20*10**9, 15*10**9, 10*10**9, 5*10**9, 0],
           ['20B','15B', '10B', '5B', '0'])
plt.xlabel('month')
plt.ylabel('UAH')
plt.title('Sales and savings - Fuel')
plt.legend(['sales', 'savings'])
plt.grid(True)
plt.show()
plt.figure(figsize=(12,5))
plt.plot(agricultural_market_sales.lot_announce_month, agricultural_market_sales.agricultural, color='#007810')
plt.plot(agricultural_market_savings.lot_announce_month, agricultural_market_savings.agricultural, color='#3f7847')
plt.xticks(['2015-12', '2016-12', '2017-12', '2018-12', '2019-12'], 
           ['15-12', '16-12', '17-12', '18-12', '2019-12'])
plt.yticks([700*10**6, 550*10**6, 400*10**6, 250*10**6, 100*10**6, 0],
           ['700M', '550M', '400M','250M', '100M', '0'])
plt.xlabel('month')
plt.ylabel('UAH')
plt.title('Sales and savings - Agricultural')
plt.legend(['sales', 'savings'])
plt.grid(True)
plt.show()
plt.figure(figsize=(12,5))
plt.plot(construction_market_sales.lot_announce_month, 
         construction_market_savings.construction/construction_market_sales.construction, color='#ac0000')
plt.plot(fuel_market_sales.lot_announce_month, 
         fuel_market_savings.fuel/fuel_market_sales.fuel, color='#007dac')
plt.plot(agricultural_market_sales.lot_announce_month, 
         agricultural_market_savings.agricultural/agricultural_market_sales.agricultural, color='#007810')
plt.xticks(['2015-02', '2015-12', '2016-12', '2017-12', '2018-12', '2019-12'], 
           ['2015-02', '15-12', '16-12', '17-12', '18-12', '2019-12'])
plt.yticks([0.5, 0.4, 0.3, 0.2, 0.1, 0],
           ['50%', '40%', '30%', '20%', '10%', 0])
plt.xlabel('month')
plt.ylabel('percent')
plt.title('% of savings')
plt.legend(['construction', 'fuel', 'agricultural'])
plt.grid(True)
plt.show()