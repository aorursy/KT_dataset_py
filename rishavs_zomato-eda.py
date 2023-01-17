#import important stuff

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import re

%matplotlib inline

plt.style.use("seaborn")
df = pd.read_csv("../input/zomato.csv")

df.head(3)
df.info()
#let's modify few column names for our ease. This is subjective, you may/may not want to do this  

df = df.rename({'approx_cost(for two people)': 'approx_cost', 'listed_in(type)': 'type', 'listed_in(city)':'city'}, axis='columns')

df.head(1)
###################################################################################

#      This section is for data cleanup and handling missing values and stuff     #

###################################################################################

from numpy import median



# This methods strips off anything after '?' in the url.

def clean_url():

    url_regex = '^.+www.zomato.com\/.+\?'     

    for index,row in df.iterrows():

        new_url= re.findall(url_regex,row['url'])

        if new_url != []:

            new_url = new_url[0][:-1]

            df.at[index,'url']=new_url #strip the ?

    print("-Done URL-")



# This methods reformats the data in column 'rate', from (eg) '4.1/5' to 4.1

def clean_rate():

    all_rates =[]

    for index,row in df.iterrows():

        if row['rate']=='NEW' or row['rate']=='-':

            df.at[index,'rate']= float(3.7)   #I already calculated the median(3.7) of the rest of the values, so using that(3.7) directly here.

            continue

        elif type(row['rate']) != str:

            continue

        rate = row['rate']

        rate = rate.split("/")

        rate= rate[0]

        df.at[index,'rate']=float(rate)

        all_rates.append(float(rate))

    value_for_nan = median(all_rates)

    #print(value_for_nan)

    df['rate'] = df['rate'].fillna(value_for_nan)

    print("-Done rate-")



# This method changes the column 'phone', to 1 or 0, depending on whether we have the phone no. or not

def clean_phone():

    for index,row in df.iterrows():

        if type(row['phone']) != str:

            df.at[index,'phone']=0

        else:

            df.at[index,'phone']=1

    print("-Done phone-")



            

# The following method 'one-hot' encodes the restaurant type(s), effectively creating a whole bunch of new columns, corresponding to the respective restaurant's type 

def clean_rest_type():

    all_types_list=[]

    for index,row in df.iterrows():

        rest_type= row['rest_type']

        if type(rest_type) != str:

            continue

        rest_type_list = rest_type.split(',')

        for val in rest_type_list:

            val = val.strip()

            if val not in all_types_list:

                all_types_list.append(val)

    print("These are the type categories for restaurants we have-\n %s"%all_types_list)

    

    for index,row in df.iterrows():

        rest_type= row['rest_type']

        if type(rest_type) != str:

            present = None

            for types in all_types_list:

                df.at[index,types]=0

            continue

        for types in all_types_list:

            if types in rest_type:

                df.at[index,types]=1

            else:

                df.at[index,types]=0

    print("-Done rest_type-")



# This method modifies the type of approx_cost, from str to float. It also fills the nan values with the median

def clean_approx_cost():

    all_cost= []

    for index,row in df.iterrows():

        if type(row['approx_cost'] ) != str:

            continue

        else:

            cost_str = row['approx_cost']

            cost_str= cost_str.replace(',','')

            df.at[index,'approx_cost'] = float(cost_str)

            all_cost.append(float(cost_str))  

#     median(all_cost) #Comes as 400

    df['approx_cost']=df['approx_cost'].fillna(400.0)

    print("-Done approx_cost -")

    



# This method creates 5 new columns for customer rates. For each entry, these column contain information on how many 1,2,3,4, or 5 star(s) were given by customers in review_list    

def clean_reviews_list():

    rate_list=[]

    #add column with default values

    df['cust_rate_1']=0

    df['cust_rate_2']=0

    df['cust_rate_3']=0

    df['cust_rate_4']=0

    df['cust_rate_5']=0

    for index,row in df.iterrows():

        reviews = row['reviews_list']

        regex_rate = 'Rated\s[0-9].[0-9]'  #Regex to fetch just the phrase 'Rated X.X'

        rates = re.findall(regex_rate,reviews)

        for cust_rate in rates:

            given_rate = float(cust_rate.split(' ')[1])

            if given_rate == 1.0:

                    current=df.loc[index,'cust_rate_1']

                    current = current + 1

                    df.at[index,'cust_rate_1'] = float(current)

            elif given_rate == 2.0:

                    current=df.loc[index,'cust_rate_2']

                    current = current + 1

                    df.at[index,'cust_rate_2'] = float(current)

            elif given_rate == 3.0:

                    current=df.loc[index,'cust_rate_3']

                    current = current + 1

                    df.at[index,'cust_rate_3'] = float(current)

            elif given_rate == 4.0:

                    current=df.loc[index,'cust_rate_4']

                    current = current + 1

                    df.at[index,'cust_rate_4'] = float(current)

            elif given_rate == 5.0:

                    current=df.loc[index,'cust_rate_5']

                    current = current + 1

                    df.at[index,'cust_rate_5'] = float(current)

                    

    print('-Done reviews_list -')





# This method creates a whole new bunch of columns based on different cuisines. These column have 1 or 0, based on whether the restaurant serves that particular cuisine

def clean_cuisines():

    all_cuisines=[]

    df['cuisines'].count()

    for index,row in df.iterrows():

        cuisine = row['cuisines']

        if type(cuisine) != str:

            continue

        cuisine = cuisine.split(',')

        for items in cuisine:

            item= items.strip()

            if item not in all_cuisines:

                all_cuisines.append(item)

    print("Following are all of the cuisines:\n%s"%all_cuisines) 

    #create columns for these cuisines with default value as 0(s). 

    for cuisine in all_cuisines:

        df[cuisine]=0

    for index,row in df.iterrows():

        cuisine = row['cuisines']

        if type(cuisine) != str:

            continue

        cuisine = cuisine.split(',')

        for items in cuisine:

            item= items.strip()

            df.at[index,item]=1

    print("-Done cuisines -")

    



# ------------Call----------------------------------------#



df.drop(['url'],inplace = True,axis=1)

df.drop(['address'],inplace = True,axis=1)

clean_rate()

clean_phone()

df['location']=df['location'].fillna('Unknown')

clean_rest_type()

df.drop(['rest_type'],inplace = True,axis=1)

clean_approx_cost()

clean_reviews_list()

df.drop(['reviews_list'],inplace=True,axis=1)

clean_cuisines()

df.drop(['cuisines'],inplace= True, axis=1)

df.drop(['dish_liked'],inplace= True, axis=1) #Any suggestions on how to handle a large number of missing data is welcomed. 51717-23639 = 28,078 missing values.

df.drop(['menu_item'],inplace=True,axis=1)

df.head(2)
# df.info(verbose =True,null_counts=True)
#UNIVARIATE ANALYSIS

##########################################

plt.xkcd(True) 

fig = plt.figure(figsize=(15,5))

fig.subplots_adjust(wspace=2)

fig.subplots_adjust(hspace=0.2)

ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,3)

ax4 = fig.add_subplot(2,2,4)



##########################################

 

colors = ['#66b3ff','#ff9999']

#column 'online_order' has categorical values. We could use a bar chart

x,y=[],[]

online_orders=df['online_order'].value_counts().to_frame()

for index,row in online_orders.iterrows():

    x.append(index)

    y.append(row.values[0])

ax1.set_ylim(top=50000)

ax1.bar(x,y,color = colors)

ax1.set_xlabel("Online Order available?")

ax1.set_ylabel("No. of restaurants")

ax1.set_title("Do restaurants accept online orders?")



#column 'book_table' has categorical values. We could use a bar chart

x2,y2=[],[]

tables_booked=df['book_table'].value_counts().to_frame()

for index,row in tables_booked.iterrows():

    x2.append(index)

    y2.append(row.values[0])



x2 = x2[::-1] #Reversing it to say Yes first, then No

y2 = y2[::-1] #Reversing it to say Yes first, then No

ax2.set_ylim(top=50000)

ax2.bar(x2,y2,color=colors)

ax2.set_xlabel("Table booking available?")

ax2.set_ylabel("No. of restaurants")

ax2.set_title("Do restaurants allow table booking?")







#Do restaurants which allow online booking have a better rating?

online_order_yes_avg_rate= df.loc[(df['online_order'] == "Yes")]['rate'].mean()    

online_order_no_avg_rate= df.loc[(df['online_order'] == "No")]['rate'].mean()

print("Average Rating for restaurants which allow online booking: %s\nAverage Rating for restaurants which don't allow online booking: %s\n"%(online_order_yes_avg_rate,online_order_no_avg_rate))



x = ["Yes","No"]

y= [online_order_yes_avg_rate,online_order_no_avg_rate]

ax3.set_ylim(top=5)

ax3.bar(x,y,color=colors)

ax3.set_xlabel("Online Booking available")

ax3.set_ylabel("Average Rate")

ax3.set_title("What is the average rate for rest. which allow online ordering?")





#Do restaurants which allow table booking have a better rating?

book_table_yes_avg_rate= df.loc[(df['book_table'] == "Yes")]['rate'].mean()    

book_table_no_avg_rate= df.loc[(df['book_table'] == "No")]['rate'].mean()

print("Average Rating for restaurants which allow table booking: %s\nAverage Rating for restaurants which don't allow table booking: %s\n"%(book_table_yes_avg_rate,book_table_no_avg_rate))



x = ["Yes","No"]

y= [online_order_yes_avg_rate,online_order_no_avg_rate]

ax4.set_ylim(top=5)

ax4.bar(x,y,color=colors,align='center')

ax4.set_xlabel("Table Booking available")

ax4.set_ylabel("Average Rate")

ax4.set_title("What is the average rate for rest. which allow table booking?")

plt.tight_layout()

plt.show()
plt.xkcd(False)

dummy_2 = df.copy()

#create a new column 'booking_ordering', which basically tells whether a restaurant allows or doesn't allow, booking or online ordering.

dummy_2['booking_ordering']='None'  



for index,row in dummy_2.iterrows():

    if row['online_order'] == "Yes" and row['book_table']=='Yes':

        dummy_2.at[index,'booking_ordering'] = 'booking Available, online Available'      

    elif row['online_order'] == "Yes" and row['book_table']=='No':

        dummy_2.at[index,'booking_ordering'] = 'booking Unavailable, online Available'

    elif row['online_order'] == "No" and row['book_table']=='Yes':

        dummy_2.at[index,'booking_ordering'] = 'booking Available, online Unavailable'

    elif row['online_order'] == "No" and row['book_table']=='No':

        dummy_2.at[index,'booking_ordering'] = 'booking Unavailable, online Unavailable'



dummy_3 = dummy_2[['booking_ordering','rate']]



#Plot

fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(1,1,1)

sns.countplot(data=dummy_3,x='rate',hue='booking_ordering',ax =ax1,palette='tab10')

plt.legend(loc='upper left')

plt.setp(ax1.get_legend().get_texts(), fontsize='22') # for legend text

plt.setp(ax1.get_legend().get_title(), fontsize='32') # for legend title

plt.tight_layout()

plt.show()
#column 'rate' has quantitative values. We could use a histogram

plt.xkcd(False)  

plt.hist(df['rate'],bins=15,color = "#ADADEB", lw="0")

plt.xlabel("overall rest. rating")

plt.ylabel("frequency")

plt.tight_layout()

plt.grid(True)

fig = plt.gcf()

fig.set_size_inches(6,4)

plt.show()

print("Mean-",df['rate'].mean())

print("Standard Dev-",df['rate'].std())





#Uncomment below please

# sns.boxplot(df['rate'])  #Can someone please suggest to me what we can do with the outliers? or why should we even do something about it?
# column 'location' has categorical values. We could use a bar chart or maybe piechart?



#########[ generate colors ]#################################

# import pylab                                              #

# NUM_COLORS = 94   #94 colors for 94 locations             #

# cm = pylab.get_cmap('gist_rainbow')                       #

# cgen = (cm(2.*i/NUM_COLORS) for i in range(NUM_COLORS))   #

#############################################################



plt.xkcd(False)

size=[]

location=[]

for index,row in (df['location'].value_counts().to_frame()).iterrows():

    location.append(index)

    size.append(row.values[0])



#Keep only top 20 location names, replace all others with '.' to avoid text overlapping.

#THIS IS A HACK, DON'T USE IT (Suggestions welcomed)

for i in range(20,len(location)):

    location[i]=''

    

colors = ['#ff9999','#ff99cf','#ff99b4','#ff99a7','#ffcf99','#ffdd99','#66b3ff','#66dcff','#668aff','#7f66ff','#99ff99','#ddff99','#a7ff99','#99ffb4','#99ffdd','#ffcc99','#eeff99']

explode = [0.05]*len(location)

# plt.barh(location[::-1],size[::-1])   #If you prefer bar chart 

plt.pie(size, labels=location,colors=colors,explode =explode, autopct='%1.f%%', pctdistance=0.85,textprops={'fontsize': 14},shadow=True)

plt.title('"% Of Total Restaurants In An Area"', y=1.08,fontsize=20)

plt.axis("equal")



#draw circle

centre_circle = plt.Circle((0,0),0.80,fc='whitesmoke')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

fig.set_facecolor('whitesmoke')

fig.set_size_inches(18,9)

plt.show()
plt.xkcd(False)

fig = plt.figure(figsize=(15,7))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)





#BIGGEST CHAIN RESTAURANTS (top 10)

names,count = [],[]

top_ten_by_numbers = df['name'].value_counts().to_frame()[:10]

for index,row in top_ten_by_numbers.iterrows():

    names.append(index)

    count.append(row.values[0])



# ax1.barh(names,count)   #This was boring. :( 

sns.barplot(count, names, palette="Blues_d",ax=ax1)   #Seaborn ! :D

ax1.set_xlabel("No. of restaurants")

ax1.set_title("BIGGEST CHAIN RESTAURANTS. (TOP 10)")



#TOP 10 BEST CHAIN RESTAURANT (top 10)  (Rated 4.8 and above)

name_by_rate,count_by_rate=[],[]

top_ten_by_rate_and_number = df[df['rate']>=4.8]['name'].value_counts().to_frame()[:10]

for index,row in top_ten_by_rate_and_number.iterrows():

    name_by_rate.append(index)

    count_by_rate.append(row.values[0])



# ax2.barh(name_by_rate,count_by_rate)    #This was boring. :( 

ax2 = sns.barplot(count_by_rate,name_by_rate,palette="Reds_d",ax=ax2) #Seaborn,what is dead may never d...wait what?

ax2.set_xlabel("No. of restaurants")

ax2.set_title("BEST RATED CHAIN RESTAURANTS. (TOP 10)")

plt.tight_layout()

plt.show()
#Which restuarant has the best rating, and which restaurant has most number of 4 stars or above? Are they same?

best_restaurants= df[df['rate']==df['rate'].max()]  #ie, restaurants which have an overall rating of 4.9(the max in out dataset)



print("All the restaurants mentioned below had an overall rating of 4.9! :\n")

print("Restaurant name\t\t\t\t\t\t\t\t\tNo of outlets\n")

print("---------------------------------------------------------------------------------------------")

print(best_restaurants['name'].value_counts())
cust_rate_df= best_restaurants.groupby(['name'])['cust_rate_4','cust_rate_5'].sum() #Create the dataframe grouped according to names, and having the sum of total 4 & 5 stars.

# print(cust_rate_df.head()) #Uncomment to check it out.

plt.xkcd(False)

x_axis_names = []

x_axis_4=[]

x_axis_5=[]

y_axis_4=[]

y_axis_5=[]

width = 0.4

i=1

for group_names in cust_rate_df.index:

    x_axis_names.append(group_names)

    i=i+2

    x_axis_4.append(i)

    x_axis_5.append(i+width)

    y_axis_4.append(cust_rate_df.loc[group_names].values[0])

    y_axis_5.append(cust_rate_df.loc[group_names].values[1])



x_axis_names[-1]='Santa Spa Cuisine'

x_axis_4=x_axis_4[::-1]  

x_axis_5=x_axis_5[::-1]

#########

#  Plot #

#########

fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(1,1,1)

ax1.barh(x_axis_4,y_axis_4,color="#89bedc",label='4 Stars')

ax1.barh(x_axis_5,y_axis_5,color="#0b559f",label='5 Stars')

ax1.set_yticks(x_axis_4,[])

ax1.set_yticklabels(x_axis_names,[])



plt.title("How many 4 & 5 stars did the best rated(4.9 overall rate) restaurants get?", y=1.08,fontsize=20)



ax1.legend(prop={'size': 22,'weight':'bold'})

ax1.set_xlabel("No of stars")

plt.grid(True)

plt.tight_layout()

plt.show()
rest_types = df.columns[10:35] #All types of restaurants

#How many of each type of restaurant

print("(*) Restaurants are categorized as following, on the basis of the type of meal.\nMeal Type\t\tNo. of restaurants in this meal type\n--------------------------------------------\n%s"%df.groupby('type')['name'].count())

print("\n(*) Restaurants are categorized as following, on the basis of restaurant type\n ")

print("Rest. Type\t\t\t\t\tNo. of restaurants of this type")

print("-------------------------------------------------------------------------------")

for rest_type in rest_types:

    total = df[rest_type].sum().astype(int)

    print("%s\t\t\t\t\t\t%s"%(rest_type,total))
#TODO do few MULTIVARIATE ANALYSES

#TODO Organise the notebook with a proper navigation section and descriptions