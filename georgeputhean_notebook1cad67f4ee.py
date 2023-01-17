import pandas as pd
import numpy as np
from geopy.distance import geodesic 
df=pd.read_csv("../input/foodorders/cs_data_2.csv",index_col='food_order_id',engine="python")
df.head()
df=df.dropna()
df=df.drop_duplicates()
len(df)
df=df[(df.city_name=="Bengaluru") & (df.order_status==6) & (df.driver_type==15)]
len(df)
grouped=df.groupby('food_order_id')
Unique_orders=grouped.first()
Unique_orders.sort_values('created_at',inplace=True)
Unique_orders.head()
len(Unique_orders)
def caldistance(startlat,startlon,endlat,endlon):
        start=(startlat,startlon)
        end=(endlat,endlon)
        return (geodesic(start, end).km)
#smaple trial
for label, row in df.iterrows():
        print(caldistance(row['pickup_latitude'],row['pickup_longitude'],row['drop_latitude'],row['drop_longitude']))
        break
# I am calculating no batch delivery time that is time to deliver first order by itself time - Actual time for first order

Unique_orders['no_batch_delivery_time']=Unique_orders['pickup_to_drop_location_km']*4
Unique_orders['no_batch_delivery_time']=Unique_orders['no_batch_delivery_time']+Unique_orders['predicted_food_prep_time']
Unique_orders.head()
def Batch_orders(breach_val):
    i=0
    Unique_orders['potential_second_order']=""
    for index_1, row_1 in Unique_orders.iterrows():
        i+=1
        if(i%5000==0):
            print(i," out of ",len(Unique_orders)," Completed")#Just to keep track of progress
        if(row_1['potential_second_order']!="Batched"):#pick only nonbatched orders
            same_restaurant_id = Unique_orders[(Unique_orders['food_restaurant_id'] == row_1['food_restaurant_id']) & (Unique_orders.index != index_1)]
            for index_2, row_2 in same_restaurant_id.iterrows(): 
                time = caldistance(row_1['drop_latitude'],row_1['drop_longitude'],row_2['drop_latitude'],row_2['drop_longitude'])*4 #60 divided by 15km/hr
                breach = (row_1['no_batch_delivery_time'] + time) - row_2['expected_delivery_time'] #breach on second order
                if (breach <= breach_val) and (len(same_restaurant_id) > 0):
                    Unique_orders.loc[index_1,'potential_second_order'] = "Batched"
                    Unique_orders.loc[index_2,'potential_second_order'] = "Batched"    
                    break
    calculate_batch_percent(breach_val)
#this function was used in the above expression just inserted later on for single

def calculate_batch_percent(breach_val):    
    batched_count=len(Unique_orders[Unique_orders.potential_second_order=="Batched"])
    Totalcount=len(Unique_orders)
    batch_percent=(batched_count/Totalcount)*100
    print(batch_percent,"%","for breach value=",breach_val,"mins")
%%time
Batch_orders(30)
#30 here is breach Value you can try chaning it and running
#this is my final function which batch orders and give you the batch perccent
#This Function Takes Approx. 27 min run completely. To make it more effective i could have used list comprehension
#or I could have converted the dataframe to to_numpy. But this is the most simple approach to problem
%%time
Unique_orders['potential_second_order'].value_counts(dropna=False)


