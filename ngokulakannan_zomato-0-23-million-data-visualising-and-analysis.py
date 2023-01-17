import pandas as pd

import numpy as np

import matplotlib 

from matplotlib import pyplot as plt

import seaborn as sns

import json

import re

from collections import Counter,OrderedDict

from ipywidgets import interact, interactive, fixed, interact_manual,Layout

import ipywidgets as widgets

'''

import gmaps

import gmaps.datasets

gmaps.configure(api_key=" ") # Your Google API key



AS kaggle wont render gmaps im commenting this section also I have used images for output whenever gmaps is used

'''
df=pd.read_csv('../input/zomato-india-restaurants2-lakh-restaurants-data/data/india_all_restaurants_details.csv')
df.set_index('zomato_url',inplace=True)

df.drop('sno',axis=1,inplace=True)

df.drop('Unnamed: 0',axis=1,inplace=True)

df.columns
def clean_data(row):

    '''

    impose none

    '''

    for cell in ['rating','rating_count','telephone','cusine','cost_for_two']:

        try:

            if float(row[cell]) == 0:

                row[cell] =None

        except:

            pass 

    if str(row['rating']) == "NEW" or str(row['rating']) == "Nové":

        row['rating']=None

    elif row['rating'] is not None :

        

        row['rating'] =float(row['rating'] ) 

        

    '''

    change quotes in timing from other language

    (this is because of unknown causes during scraping of data)

    to english

    '''

    if type(row['timings'])==str:

            row['timings']=row['timings'].replace("'",'"')

            

            row['timings']=row['timings'].replace("Pon",'Mon')

            row['timings']=row['timings'].replace("Uto",'Tue')

            row['timings']=row['timings'].replace("Str",'Wed')

            row['timings']=row['timings'].replace("Štv",'Thu')

            row['timings']=row['timings'].replace("Pia",'Fri')

            row['timings']=row['timings'].replace("Sob",'Sat')

            row['timings']=row['timings'].replace("Ned",'Sun')

            

            row['timings']=row['timings'].replace("Lun",'Mon')

            row['timings']=row['timings'].replace("Mar",'Tue')

            row['timings']=row['timings'].replace("Mer",'Wed')

            row['timings']=row['timings'].replace("Gio",'Thu')

            row['timings']=row['timings'].replace("Ven",'Fri')

            row['timings']=row['timings'].replace("Sab",'Sat')

            row['timings']=row['timings'].replace("Dom",'Sun')

            

            row['timings']=row['timings'].replace("Po",'Mon')

            row['timings']=row['timings'].replace("Út",'Tue')

            row['timings']=row['timings'].replace("St",'Wed')

            row['timings']=row['timings'].replace("Čt",'Thu')

            row['timings']=row['timings'].replace("Pá",'Fri')

            row['timings']=row['timings'].replace("So",'Sat')

            row['timings']=row['timings'].replace("Ne",'Sun')

            

                        

            row['timings']=row['timings'].replace("Pts",'Mon')

            row['timings']=row['timings'].replace("Sa",'Tue')

            row['timings']=row['timings'].replace("Çrş",'Wed')

            row['timings']=row['timings'].replace("Prş",'Thu')

            row['timings']=row['timings'].replace("Cum",'Fri')

            row['timings']=row['timings'].replace("Cts",'Sat')

            row['timings']=row['timings'].replace("Paz",'Sun')

            

            row['timings']=row['timings'].replace("Wt",'Tue')

            row['timings']=row['timings'].replace("Śr",'Wed')

            row['timings']=row['timings'].replace("Czw",'Thu')

            row['timings']=row['timings'].replace("Pt",'Fri')

            row['timings']=row['timings'].replace("Sb",'Sat')

            row['timings']=row['timings'].replace("Nd",'Sun')

            

            row['timings']=row['timings'].replace("Tuet",'Sat')

            row['timings']=row['timings'].replace("Zamknięte",'Closed')

            row['timings']=row['timings'].replace("Zatvorené",'Closed')

            

            '''

            change timings from format 10am - 8pm to

            [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0]

            '''    

            timings=json.loads(row['timings'])

            new_timings=dict()

            for key,value in timings.items():

                if (value.find('24')>=0):

                    new_timings[key]= [1 for i in range(24)]

                    continue

                if (value.find('Closed')>=0  ):

                    new_timings[key]= [0 for i in range(24)]

                    continue



                value=value.replace(" do ","–")

                value=value.replace(" a ","–")

                value=value.replace("-","–")

                value=value.replace("midnight","am")

                value=value.replace("noon","pm")

                value=value.replace("PM","pm")

                value=value.replace("AM","pm")

                value=value.replace("12 am","0 am")

                value=value.replace("12am","0am")



                if (not value.find('–')>=0  ):

                    new_timings[key]= [0 for i in range(24)]

                    fff.append(value)

                    continue



                _temp=[0 for i in range(24)]

                _time=value.split(',')

                for t in _time:

                    hour=[0,0]

                    mins=[0,0]

                    j=0

                    for i in t.split("–"):

                        try:

                            ___==int(re.findall("\d+", i)[0])

                        except:

                            print(key,value,t,i)

                        if "12" in i and "am" in i:

                            i=i.replace("12","0")

                        #hour[j]=int(re.findall("\d+", i)[0])

                        tym=i.replace("pm","").replace("am","").split(":")

                        hour[j]=int(tym[0])

                        if(len(tym)>1):

                            if(int(tym[1])<=15):

                                mins[j]= float( 0.25 )

                            elif(int(tym[1])<=30):

                                mins[j]=float( 0.50)

                            elif(int(tym[1])<=45):

                                mins[j]=float( 0.75)



                        if(i.find('pm')>=0):

                            if not i.find('12')>=0:

                                hour[j]=hour[j]+12 

                        j=j+1



                    for i in range(hour[0],hour[0]+23):

                        if i%24 ==hour[1]:

                            if mins[1]>0:

                                _temp[i%24]=mins[1]

                            break;

                        _temp[i%24]=1

                        if i==hour[0]:

                            if mins[0]>0:

                                _temp[i%24]=mins[0]

                new_timings[key]=_temp

            row['timings']=new_timings

    if type(row['cost_for_two'])==str:

            row['cost_for_two']=int(row['cost_for_two'].replace(',',''))



    '''

    Get latitude and longitide coordinates seperately and impose none if

    the values are not in range.

    '''

    lat,long=row['coordinates'].split(',')

    long=float(long)

    lat=float(lat)

    if long > 180  or long < -180 or lat > 90  or lat < -90:

        lat=None

        long=None

    row['longitude']=long

    row['latitude']=lat

    del row['coordinates']

    

    

    return row





 
df=df.apply(clean_data,axis=1)


df.to_csv("../input/zomato-india-restaurants2-lakh-restaurants-data/data/indian_restaurants_details_cleaned_data.csv")
df=pd.read_csv('../input/zomato-india-restaurants2-lakh-restaurants-data/data/indian_restaurants_details_cleaned_data.csv')

df.head()
def find_famous_cusines(data_frame,title,min_no_of_cusines):    

    cusines=data_frame['cusine'].dropna()

    all_cus=list()

    for cusine in cusines:

            temp=cusine.split(',')

            for t in temp:

                all_cus.append(t.strip().lower())

    cusines_counter=dict(Counter(all_cus))

    cusine_dict= OrderedDict()

    cusine_list=[]

    counter_list=[]

    

    for key, value in sorted(cusines_counter.items(), key=lambda item: item[1],reverse=True ):

        cusine_dict[key]=value

        if value > min_no_of_cusines:

            cusine_list.append(key)

            counter_list.append(value)

    sns.set(style="whitegrid")

    plt.figure(figsize=(15,10))  

    sns.barplot(x= counter_list,y=cusine_list)

    plt.xlabel("Number of restaurants",size=18)

    plt.title(title,size=24)

    for tick in plt.gca().get_yticklabels():

        tick.set_fontsize(12)

    plt.show()

#Famous cusines - india

find_famous_cusines(df,"Famous cusines around the country",1000)



#Famous cusines - chennai

df_chennai=df[df.city=="Chennai"]

find_famous_cusines(df_chennai,"Famous cusines around Chennai city",200)
@interact

def show_famous_cusines(min_no_of_cusines=widgets.IntSlider(description='Cusines with restaurants count more than',min=100,max=25000,step=500,value=1000, continuous_update=False,style = {'description_width': 'initial'},layout=Layout(width='50%', height='80px'))

                 ,city=widgets.Dropdown(

    options=df.city.unique(),

    value='Chennai',

    description='City:',

    disabled=False,

)          ):

    try:

        df_city=df[df.city==city]

        cusines=df_city['cusine'].dropna()

        all_cus=list()

        for cusine in cusines:

                temp=cusine.split(',')

                for t in temp:

                    all_cus.append(t.strip().lower())

        cusines_counter=dict(Counter(all_cus))

        cusine_dict= OrderedDict()

        cusine_list=[]

        counter_list=[]





        for key, value in sorted(cusines_counter.items(), key=lambda item: item[1],reverse=True ):

            cusine_dict[key]=value

            if value > min_no_of_cusines:

                cusine_list.append(key)

                counter_list.append(value)

        sns.set(style="whitegrid")

        plt.figure(figsize=(15,10))  

        sns.barplot(x= counter_list,y=cusine_list)

        #sns.barh(cusine_list, counter_list)

        plt.xlabel("Number of restaurants",size=18)

        plt.title("Find the famous cusines around the country",size=24)

        for tick in plt.gca().get_yticklabels():

            tick.set_fontsize(14)

        plt.show()

    except:

        print("please reduce the count...")

    





   


india_map_df = df[['latitude', 'longitude']]

india_map_df=india_map_df[(india_map_df.latitude > 0) & (india_map_df.longitude >0) ]

india_map_df.dropna(inplace=True)

heatmap = gmaps.heatmap_layer(india_map_df)

heatmap.max_intensity = 5000

heatmap.point_radius = 10

centre_coord=(np.mean( india_map_df.latitude),np.mean( india_map_df.longitude))

fig = gmaps.figure(center=centre_coord,zoom_level=4)

fig.add_layer(heatmap)

fig

df_chennai=df[df.city=="Chennai"]

chennai_map_df = df_chennai[['latitude', 'longitude']]

chennai_map_df=chennai_map_df[(chennai_map_df.latitude > 0) & (chennai_map_df.longitude >0) ]

chennai_map_df.dropna(inplace=True)

heatmap_chennai = gmaps.heatmap_layer(chennai_map_df)

heatmap_chennai.max_intensity = 20

heatmap_chennai.point_radius = 5



centre_coord_chennai=(np.mean( chennai_map_df.latitude),np.mean( chennai_map_df.longitude))

fig_chennai = gmaps.figure(center=centre_coord_chennai,zoom_level=10)

fig_chennai.add_layer(heatmap_chennai)

fig_chennai

@interact

def show_heatmap_of_restaurants(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):   

    df_city=df[df.city==city]

    city_map_df = df_city[['latitude', 'longitude']]

    city_map_df=city_map_df[(city_map_df.latitude > 0) & (city_map_df.longitude >0) ]

    city_map_df.dropna(inplace=True)

    heatmap_city = gmaps.heatmap_layer(city_map_df)

    heatmap_city.max_intensity = 20

    heatmap_city.point_radius = 5



    centre_coord_city=(np.mean( city_map_df.latitude),np.mean( city_map_df.longitude))

    fig_city = gmaps.figure(center=centre_coord_city,zoom_level=10)

    fig_city.add_layer(heatmap_city)

    display(fig_city)
restaurant_chains=df.city.value_counts()[:10]



fig1, ax1 = plt.subplots(figsize=(20,10))

patches, texts, autotexts=ax1.pie(restaurant_chains.values,  labels=restaurant_chains.index, autopct='%1.1f%%',

        shadow=True, startangle=0)

ax1.axis('equal')

plt.title("Restaurant distribution of top Cities in India",size=24)

for text in texts:

    text.set_fontsize(15)

for text in autotexts:

    text.set_fontsize(15)

plt.show()
def find_famous_restaurant_chains(title,df):

    restaurant_chains=df.name.value_counts()[:25]

    sns.set(style="whitegrid")

    plt.figure(figsize=(15,10))  

    sns.barplot(x= restaurant_chains.values,y=restaurant_chains.index)

    plt.xlabel("Numer of outlets",size=18)

    plt.title(title,size=24)

    for tick in plt.gca().get_yticklabels():

        tick.set_fontsize(14)

    

    plt.show()

find_famous_restaurant_chains("Famous restaurant chains in India",df)

find_famous_restaurant_chains("Famous restaurant chains in Chennai",df[df.city=="Chennai"])
@interact

def show_famous_restaurant_chains(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):

    restaurant_chains=df[df.city==city].name.value_counts()[:25]

    sns.set(style="whitegrid")

    plt.figure(figsize=(15,10))  

    sns.barplot(x= restaurant_chains.values,y=restaurant_chains.index)

    plt.xlabel("Numer of outlets",size=18)

    plt.title("Famous restaurant chains in "+city,size=24)

    for tick in plt.gca().get_yticklabels():

        tick.set_fontsize(14)

    plt.show()

plt.figure(figsize=(15,10)) 

sns.distplot(df.rating.dropna(),bins=30,color='b',kde_kws={"color": "g"})

plt.ylabel("Probability")

plt.xlabel("Restaurant Rating")

plt.title("Rating distribution ",size=24)

plt.show()
fig, ax = plt.subplots(figsize =(15,10)) 

plt.subplot(1,2,1)

sns.boxplot(y=df.cost_for_two.dropna())

plt.ylabel("Cost for two")

plt.title("Cost distribution box plot ",size=20)

plt.subplot(1,2,2)

sns.boxplot( y = df.cost_for_two.dropna()[(df.cost_for_two<1000)])

plt.ylabel("Cost for two")

plt.title("Cost distribution - zoomed in ",size=20)

plt.suptitle("Cost distribution",size=24)

plt.show()
def plot_famous_food_static(df_food,title):

    famous_food_df=df_food.famous_food.dropna()

    famous_food_list=[]

    for food in famous_food_df:

            temp=food.split(',')

            for t in temp:

                # this commented code was to split the food name and take the second part of name

                # for counting.for example, chicken biriyani -> biriyani. 

                # If we do so biriyani count will increase

                ''''t=t.split()

                if len(t) >1:

                    t=t[1]

                else:

                    t=t[0]'''

                famous_food_list.append(t.strip().lower())



    famous_food_df= pd.Series( famous_food_list)

    famous_food_list=famous_food_df.value_counts()[:25]



    plt.figure(figsize=( 15,10))

    sns.barplot(x=famous_food_list.values,y=famous_food_list.index)

    plt.title(title,size=24)

    for tick in plt.gca().get_yticklabels():

        tick.set_fontsize(14)

    plt.xlabel("Number of restaurants",size=18)

    plt.show()

plot_famous_food_static(df,"Famous food in India")



plot_famous_food_static(df[df.city=="Chennai"],"Famous food in Chennai")
@interact

def plot_famous_food_dynamic(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):

    famous_food_df=df[df.city==city].famous_food.dropna()

    famous_food_list=[]

    for food in famous_food_df:

            temp=food.split(',')

            for t in temp:

                # this commented code was to split the food name and take the second part of name

                # for counting.for example, chicken biriyani -> biriyani. 

                # If we do so biriyani count will increase

                '''t=t.split()

                if len(t) >1:

                    t=t[1]

                else:

                    t=t[0]'''

                famous_food_list.append(t.strip().lower())



    famous_food_df= pd.Series( famous_food_list)

    famous_food_list=famous_food_df.value_counts()[:25]



    plt.figure(figsize=( 15,10))

    sns.barplot(x=famous_food_list.values,y=famous_food_list.index)

    plt.title("Famous food in "+str(city),size=24)

    for tick in plt.gca().get_yticklabels():

        tick.set_fontsize(14)

    plt.xlabel("Number of restaurants",size=18)

    plt.show()    

    
def plot_online_orders_static(df_oo,title):

    online_orders=df_oo.online_order.value_counts()



    idx_list=online_orders.index.tolist()

    

    if idx_list[0]==True:

        idx_list[0]="Yes"

        idx_list[1]="No"

    else:

        idx_list[1]="Yes"

        idx_list[0]="No"

    online_orders.index=idx_list 

    

    plt.figure(figsize=(15,10))

    plt.pie(x=online_orders.values,labels=online_orders.index, autopct='%1.1f%%',

            shadow=True, startangle=90,textprops={'size':20})

    plt.title(title,size=24)

    plt.show()



#plot_online_orders_static(df,"Restaurants accepting online orders - India")

plot_online_orders_static(df[df.city=="Chennai"],"Restaurants accepting online orders - Chennai")   

    

    
@interact

def plot_famous_food_dynamic(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):

    online_orders=df[df.city==city].online_order.value_counts()

    

    idx_list=online_orders.index.tolist()

    if idx_list[0]==True:

        idx_list[0]="yes"

        idx_list[1]="No"

    else:

        idx_list[1]="yes"

        idx_list[0]="No"

    online_orders.index=idx_list 

    

    plt.figure(figsize=(15,10))

    plt.pie(x=online_orders.values,labels=online_orders.index, autopct='%1.1f%%',

            shadow=True, startangle=90,textprops={'size':20},explode=(0,0.05))

    plt.title("Restaurants accepting online orders - "+str(city),size=24)

    plt.show()
def plot_table_reserve_static(df_tr,title):

    table_reserve=df_tr.table_reservation.value_counts()



    idx_list=table_reserve.index.tolist()

    if idx_list[0]==True:

        idx_list[0]="yes"

        idx_list[1]="No"

    else:

        idx_list[1]="yes"

        idx_list[0]="No"

    table_reserve.index=idx_list 

    

    plt.figure(figsize=(15,10))

    plt.pie(x=table_reserve.values,labels=table_reserve.index, autopct='%1.1f%%',

            shadow=True, startangle=0,textprops={'size':20})

    plt.title(title,size=24)

    plt.gca().axis('equal')

    plt.show()



plot_table_reserve_static(df,"Restaurants accepting table reservation - India")

plot_table_reserve_static(df[df.city=="Chennai"],"Restaurants accepting table reservation - Chennai")   

    

    
@interact

def plot_famous_food_dynamic(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):

    table_reserve=df[df.city==city].table_reservation.value_counts()



    idx_list=table_reserve.index.tolist()

    if idx_list[0]==True:

        idx_list[0]="yes"

        idx_list[1]="No"

    else:

        idx_list[1]="yes"

        idx_list[0]="No"

    table_reserve.index=idx_list 



    plt.figure(figsize=(15,10))

    plt.pie(x=table_reserve.values,labels=table_reserve.index, autopct='%1.1f%%',

            shadow=True, startangle=90,textprops={'size':20},explode=(0,0.05))

    plt.title("Restaurants accepting table reservation - "+str(city),size=24)

    plt.show()
def plot_delivery_only_static(df_tr,title):

    delivery_only=df_tr.delivery_only.value_counts()



    idx_list=delivery_only.index.tolist()

    if idx_list[0]==True:

        idx_list[0]="yes"

        idx_list[1]="No"

    else:

        idx_list[1]="yes"

        idx_list[0]="No"

    delivery_only.index=idx_list 

    

    plt.figure(figsize=(15,10))

    plt.pie(x=delivery_only.values,labels=delivery_only.index, autopct='%1.1f%%',

            shadow=True, startangle=0,textprops={'size':20})

    plt.title(title,size=24)

    plt.gca().axis('equal')

    plt.show()



plot_delivery_only_static(df,"Delivery only restaurants - India")

plot_delivery_only_static(df[df.city=="Chennai"],"Delivery only restaurants - Chennai")   

    

    
@interact

def plot_delivery_only_dynamic(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):

    delivery_only=df[df.city==city].delivery_only.value_counts()



    idx_list=delivery_only.index.tolist()

    if idx_list[0]==True:

        idx_list[0]="yes"

        idx_list[1]="No"

    else:

        idx_list[1]="yes"

        idx_list[0]="No"

    delivery_only.index=idx_list 



    plt.figure(figsize=(15,10))

    plt.pie(x=delivery_only.values,labels=delivery_only.index, autopct='%1.1f%%',

            shadow=True, startangle=90,textprops={'size':20})

    plt.title("Delivery only restaurants - "+str(city),size=24)

    plt.show()
g_days={}

def get_timings(row):

    global g_days

    if type(row) is str:

        try:

            days = json.loads(row.replace("'",'"'))

            for day in days:

                for hour,state in enumerate( days[day]):

                    if state > 0:

                        g_days[day][hour] =g_days[day][hour]+ state

        except:

            print(row)

    else:

        pass

    return row
def plot_restaurants_open_static(df_tr,title):

    global g_days

    g_days={'Mon': np.zeros(24), 'Tue': np.zeros(24), 'Wed': np.zeros(24), 'Thu': np.zeros(24), 'Fri': np.zeros(24), 'Sat': np.zeros(24), 'Sun': np.zeros(24)}

    _=df_tr['timings'].apply(get_timings)

    

    plt.figure(figsize=(12,7))

    plt.plot(np.arange(0,24), g_days['Mon'])

    plt.plot(np.arange(0,24), g_days['Tue'])

    plt.plot(np.arange(0,24), g_days['Wed'])

    plt.plot(np.arange(0,24), g_days['Thu'])

    plt.plot(np.arange(0,24), g_days['Fri'])

    plt.plot(np.arange(0,24), g_days['Sat'])

    plt.plot(np.arange(0,24), g_days['Sun'])

    plt.legend(('Mon', 'Tue', 'Wed','Thu','Fri','Sat','Sun'),

               loc='upper right')

    plt.title(title,size=20)

    plt.ylabel("Number of open restaurants")

    plt.xlabel("Hours of the day (0-23)")

    plt.show()



plot_restaurants_open_static(df,"At what time most of the restaurants remains open - India?")

plot_restaurants_open_static(df[df.city=="Chennai"],"At what time most of the restaurants remains open - Chennai?")
@interact

def plot_restaurants_open_static(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):

    global g_days

    g_days={'Mon': np.zeros(24), 'Tue': np.zeros(24), 'Wed': np.zeros(24), 'Thu': np.zeros(24), 'Fri': np.zeros(24), 'Sat': np.zeros(24), 'Sun': np.zeros(24)}

    df_tr=df[df.city==city]

    _=df_tr['timings'].apply(get_timings)

    

    plt.figure(figsize=(12,7))

    plt.plot(np.arange(0,24), g_days['Mon'])

    plt.plot(np.arange(0,24), g_days['Tue'])

    plt.plot(np.arange(0,24), g_days['Wed'])

    plt.plot(np.arange(0,24), g_days['Thu'])

    plt.plot(np.arange(0,24), g_days['Fri'])

    plt.plot(np.arange(0,24), g_days['Sat'])

    plt.plot(np.arange(0,24), g_days['Sun'])

    plt.legend(('Mon', 'Tue', 'Wed','Thu','Fri','Sat','Sun'),

               loc='upper right')

    plt.title("At what time most of the restaurants remains open - "+str(city),size=20)

    plt.ylabel("Number of open restaurants")

    plt.xlabel("Hours of the day (0-23)")

    plt.show()
def get_night_restaurants(row): 

    timings=row['timings']

    if type(timings) is str:

        days = json.loads(timings.replace("'",'"'))

        day="Mon"

        for hour,state in enumerate( days[day]):

                if state > 0 and (hour<5 ):

                    return row
chennai_map_df=df[df.city=="Chennai"]

chennai_map_df = chennai_map_df.apply(get_night_restaurants,axis=1,result_type='broadcast')

chennai_map_df.dropna(inplace=True)



chennai_map_df = chennai_map_df[['latitude', 'longitude']]

chennai_map_df=chennai_map_df[(chennai_map_df.latitude > 0) & (chennai_map_df.longitude >0) ]





heatmap_chennai = gmaps.heatmap_layer(chennai_map_df)

heatmap_chennai.max_intensity = 10

heatmap_chennai.point_radius = 5



centre_coord_chennai=(np.mean( chennai_map_df.latitude),np.mean( chennai_map_df.longitude))

fig_chennai = gmaps.figure(center=centre_coord_chennai,zoom_level=10)

fig_chennai.add_layer(heatmap_chennai)

fig_chennai
@interact

def plot_restaurants_open_dynamic(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):

    city_map_df=df[df.city==city]

    city_map_df = city_map_df.apply(get_night_restaurants,axis=1,result_type='broadcast')

    city_map_df.dropna(inplace=True)



    city_map_df = city_map_df[['latitude', 'longitude']]

    city_map_df=city_map_df[(city_map_df.latitude > 0) & (city_map_df.longitude >0) ]

    if city_map_df.count().latitude <1:

        return "No restaurants available"



    heatmap_city = gmaps.heatmap_layer(city_map_df)

    heatmap_city.max_intensity = 10

    heatmap_city.point_radius = 5



    centre_coord_city=(np.mean( city_map_df.latitude),np.mean( city_map_df.longitude))

    fig_city = gmaps.figure(center=centre_coord_city,zoom_level=10)

    fig_city.add_layer(heatmap_city)

    display(fig_city)
def get_midnight_restaurants(row): 

    timings=row['timings']

    if type(timings) is str:

        days = json.loads(timings.replace("'",'"'))

        day="Mon"

        for hour,state in enumerate( days[day]):

                if state > 0 and (hour<5 ):

                    return row
def find_famous_cusines_at_midnight(data_frame,title,min_no_of_cusines): 

    data_frame = data_frame.apply(get_midnight_restaurants,axis=1,result_type='broadcast')

    data_frame.dropna(inplace=True)

    cusines=data_frame['cusine'].dropna()

    all_cus=list()

    for cusine in cusines:

            temp=cusine.split(',')

            for t in temp:

                all_cus.append(t.strip().lower())

    cusines_counter=dict(Counter(all_cus))



    cusine_dict= OrderedDict()

    cusine_list=[]

    counter_list=[]

    

    for key, value in sorted(cusines_counter.items(), key=lambda item: item[1],reverse=True ):

        cusine_dict[key]=value

        if value > min_no_of_cusines:

            cusine_list.append(key)

            counter_list.append(value)

    sns.set(style="whitegrid")

    plt.figure(figsize=(15,10)) 



    sns.barplot(x= counter_list,y=cusine_list)

    plt.xlabel("Number of restaurants",size=18)

    plt.title(title,size=24)

    for tick in plt.gca().get_yticklabels():

        tick.set_fontsize(12)

    plt.show()

#Famous cusines - india

find_famous_cusines_at_midnight(df,"Famous midnight cusines around the country",1000)



#Famous cusines - chennai

df_chennai=df[df.city=="Chennai"]

find_famous_cusines_at_midnight(df_chennai,"Famous midnight cusines around Chennai city",20)
@interact

def find_famous_cusines_at_midnight_dynamic(city=widgets.Dropdown(options=df.city.unique(),value='Chennai', description='City:',disabled=False)):

    min_no_of_cusines = 20

    data_frame=df[df.city==city]

    data_frame = data_frame.apply(get_midnight_restaurants,axis=1,result_type='broadcast')

    data_frame.dropna(inplace=True)

    cusines=data_frame['cusine'].dropna()

    all_cus=list()

    for cusine in cusines:

            temp=cusine.split(',')

            for t in temp:

                all_cus.append(t.strip().lower())

    cusines_counter=dict(Counter(all_cus))



    cusine_dict= OrderedDict()

    cusine_list=[]

    counter_list=[]

    

    for key, value in sorted(cusines_counter.items(), key=lambda item: item[1],reverse=True ):

        cusine_dict[key]=value

        if value > min_no_of_cusines:

            cusine_list.append(key)

            counter_list.append(value)

            

    if len(cusine_list) <1:

        return "No data available"

    

    sns.set(style="whitegrid")

    plt.figure(figsize=(15,10)) 



    sns.barplot(x= counter_list,y=cusine_list)

    plt.xlabel("Number of restaurants",size=18)

    plt.title("Famous midnight cusines around "+city+" city",size=24)

    for tick in plt.gca().get_yticklabels():

        tick.set_fontsize(12)

    plt.show()
