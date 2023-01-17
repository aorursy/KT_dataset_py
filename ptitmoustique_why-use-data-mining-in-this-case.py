import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import seaborn as sns
import math

from matplotlib import pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


def remove_car(string):
    if string.find('(') != -1:
        return int(string[string.find('(') + 1: string.find(')')])
    else:
        return int(string)
    

def get_week(df_data):
    #     print(dt.date(df_data['Year'], df_data['Month'], df_data['Day']))
    #     print(dt.date(df_data['Year'], df_data['Month'], df_data['Day']).isocalendar()[1])
    return dt.date(df_data['Year'], df_data['Month'], df_data['Day']).isocalendar()[1]
    
    
df_data = pd.read_csv('../input/Historical Product Demand.csv')

df_data = df_data.dropna()
df_data['Date'] = df_data['Date'].apply(lambda x: str(x))
df_data['Year'] = df_data['Date'].apply(lambda x: int(x[0:x.find('/')]))
df_data['Date'] = df_data['Date'].apply(lambda x: x[x.find('/') + 1: len(x)])
df_data['Month'] = df_data['Date'].apply(lambda x: int(x[0:x.find('/')]))
df_data['Day'] = df_data['Date'].apply(lambda x: int(x[x.find('/') + 1: len(x)]))
df_data['Week'] = df_data.apply(get_week, axis=1)
df_data = df_data.drop(['Date'], axis=1)

df_data['Month_evol'] = df_data.apply(lambda x: (x['Year'] - 2011) * 12 + x['Month'], axis=1)
df_data['Week_evol'] = df_data.apply(lambda x: (x['Year'] - 2011) * 52 + x['Week'], axis=1)
df_data = df_data.sort_values(['Year', 'Month', 'Day'])
df_data['Order_Demand'] = df_data['Order_Demand'].apply(remove_car)

df_gb = pd.DataFrame(
    df_data.groupby(['Product_Code', 'Month_evol'], as_index=False)['Order_Demand'].agg(['sum']).reset_index())

product_list = list(set(df_gb['Product_Code'].as_matrix()))


tot_nb_month = 0
#total unsatisfied demand in terms of volume
tot_demand_not_satisfy = 0
#total satisfied demand in terms of volume
tot_demand_satisfy = 0
#number of months with a deadstock
tot_dead_stock = 0
#total volume in the stock
tot_inventory = 0

#for each product
for p in product_list:
    #we study only that product
    serie = df_gb[df_gb['Product_Code'] == p]['sum'].as_matrix()

    #current unsatisfied volume
    demand_not_satisfy = 0
    #current number of dead-stock
    out_of_stock = 0
    
    
    i = 10

    #we have to set some initial parameters, there is probably better methods to do it
    reorder_level = [serie[0]] * 10
    upto_level = [serie[0]] * 10
    inventory_level = [serie[0]] * 10
    reorder_quantity = [serie[0]] * 10
    past_i = 0
    time_window = 12
    while i < len(serie):
        past_i = i - time_window
        if past_i < 0:
            past_i = 0
        #at the beginning of each month, we will recompute the mean and the variance of the previous time_window values
        mean_serie = np.mean(serie[past_i:i])
        var_serie = np.std(serie[past_i:i])
        #we have to set a limit under which we have to order a volume
        reorder_level.append(mean_serie + 1.0 * var_serie)
        #we reorder a volume to reach upto_level limit.
        upto_level.append(mean_serie + 1.5 * var_serie)
        tot_nb_month += 1
        
        #to begin the new month , we have the inventory of the previous month + what we have order two months ago (1 month of delay) 
        #at the end of the month we have to remove the demand of the current month  
        if inventory_level[len(inventory_level) - 1] + reorder_quantity[len(reorder_quantity) - 2] - serie[i] < 0:
            #we have an unsatisfied demand which corresponds to what is asked for this month - what we had 
            demand_not_satisfy += serie[i] - inventory_level[len(inventory_level) - 1] - reorder_quantity[
                len(reorder_quantity) - 2]
            #the satisfied demand corresponds to what we had in stock.
            tot_demand_satisfy += inventory_level[len(inventory_level) - 1] + reorder_quantity[len(reorder_quantity) - 2]
            #if the stock is negative we have a dead stock
            out_of_stock = out_of_stock + 1
            #the inventory_level falls down 0
            inventory_level.append(0)

        else:
            #if it's possitive, we satisfied
            tot_demand_satisfy += serie[i]
            #we remove the demand from what we had in stock at the beginning of the month.
            inventory_level.append(
                inventory_level[len(inventory_level) - 1] + reorder_quantity[len(reorder_quantity) - 2] - serie[i])

        #at the end of the month, we have to decide if we reorder a volume of not.
        #if it's under of the computed limit reorder_level, we do either we don't
        #we reorder a quantity in order to reach upto_level limit
        if inventory_level[len(inventory_level) - 1] < reorder_level[len(reorder_level) - 1]:
#             if upto_level[len(upto_level) - 1] - inventory_level[len(inventory_level) - 1] - reorder_quantity[len(reorder_quantity) - 1] > 0:
            reorder_quantity.append(upto_level[len(upto_level) - 1] - inventory_level[len(inventory_level) - 1])
#             else :
#                 reorder_quantity.append(0)
                
        else:
            reorder_quantity.append(0)

        i = i + 1
    tot_demand_not_satisfy += demand_not_satisfy
    tot_dead_stock += out_of_stock
    tot_inventory += np.sum(inventory_level)
    
#     plt.figure()
#     plt.plot(serie, label='serie')
#     plt.plot(inventory_level, label='inventory')
#     plt.plot(reorder_level, label='reorder_level')
#     plt.plot(upto_level, label='upto')
#     plt.legend()
#     plt.show()

print('number of total months')
print(tot_nb_month)
print('number of dead stocks (all products)')
print(tot_dead_stock)
print('satisfied demand (all products)')
print(tot_demand_satisfy)
print('unsatisfied demand (all products)')
print(tot_demand_not_satisfy)
print('total volume in the inventory (all products)')
print(tot_inventory)
print("ratio of satisfied demands")
print(tot_demand_satisfy/(tot_demand_satisfy + tot_demand_not_satisfy))
print('ratio of inventory')
print(tot_inventory/(tot_demand_satisfy + tot_demand_not_satisfy))
