import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns
Data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
Data.shape #Checking the number of rows and columns
Data.head(5) #Quick overview of the table and display first 5 rows of the dataset
Data.tail(5)
Data.drop(['id','name','host_name','last_review'], axis=1, inplace=True)
Data.isnull().sum()
Data.fillna({'reviews_per_month':0}, inplace=True)

Data.reviews_per_month.isnull().sum() #Checking the changes made

Data.isnull().sum() #There are no null values left

Data.neighbourhood_group.unique()
Data.room_type.unique()
top_borough=Data.neighbourhood_group.value_counts()

top_borough1=pd.concat([Data.neighbourhood_group.value_counts() , 

                        Data.neighbourhood_group.value_counts(normalize=True).mul(100)],axis=1, keys=('listings','percentage'))

print (top_borough1)
sns.set(rc={'figure.figsize':(12,8)})                     #Setting figure size for future visualizations

sns.set_palette("pastel")                                 #Set the palette to the "pastel" default palette

V1 = sns.countplot(x='neighbourhood_group', data=Data)    #Using Seaborn to create a countplot directly 

V1.set_xlabel('Borough')                                  #Changing Labels

V1.set_ylabel('Listings')

V1.set_xticklabels(V1.get_xticklabels(), rotation=45)     #Rotating Labels slightly
V2 = sns.countplot(y='neighbourhood',                                            #Create a Horizontal Plot

                   data=Data,                                                    

                   order=Data.neighbourhood.value_counts().iloc[:10].index,      #We want to view the top 10 Neighbourhoods

                   edgecolor=(0,0,0),                                            #This cutomization gives us black borders around our plot bars

                   linewidth=2)

V2.set_title('Listings by Top NYC Neighbourhood')                                #Set Title

V2.set_xlabel('Neighbourhood')                                  

V2.set_ylabel('Listings')
Data = Data.rename(columns = {"number_of_reviews" : "Visits"})   #Renaming the column to Visits

Data.head(2)                                                     #Checking the change that was made
Listings_by_borough = pd.DataFrame(Data.neighbourhood_group.value_counts().reset_index().values, columns=['Borough', 'Listings']) #Creating a new table directly into a dataframe

Listings_by_borough = Listings_by_borough.sort_index(axis=0, ascending=True)                                                      #sorting the data

Listings_by_borough ['% Listings']=  (Listings_by_borough['Listings'] / Listings_by_borough['Listings'].sum())*100                #Adding a % Listings column

Listings_by_borough                                                                                                               #Printing the table
V10 = sns.barplot(x='Borough', y = '% Listings',                                           

                   data=Listings_by_borough,                                                         

                   edgecolor=(0,0,0),                                            

                   linewidth=2)

V10.set_title('% Listings by Borough')

V10.set_xlabel('Borough')                                  

V10.set_ylabel('% Listings')
visits_by_borough = Data.groupby(['neighbourhood_group'])['Visits'].agg(np.sum).reset_index()            #Using Groupby to get 'by Borough' and numpy sum function to get 'Total Vists'

visits_by_borough.columns = ['Borough', 'Visits']                                                        #Renaming the columns

visits_by_borough = visits_by_borough.sort_values('Visits', ascending=False)                             #Sorting Visit Values in descending order

visits_by_borough ['% Visits']=  (visits_by_borough['Visits'] / visits_by_borough['Visits'].sum())*100   #Creating a new column called % Visits



visits_by_borough                                                                                        #Printing the table
V3 = sns.barplot(x='Borough', y = '% Visits',                                           

                   data=visits_by_borough,                                                         

                   edgecolor=(0,0,0),                                            

                   linewidth=2)

V3.set_title('% Visits by Borough')

V3.set_xlabel('Borough')                                  

V3.set_ylabel('% Visits')
V4 = sns.barplot(

    x='neighbourhood_group', y='Visits', 

    estimator=np.sum,                          # "sum" function from numpy as estimator , you can also use lambda x: sum(x==0)*100.0/len(x) for a percentage function

    data=Data,                                 # Raw dataset fed directly to Seaborn

    edgecolor=(0,0,0), 

    linewidth=2,

    ci=None)                                   #Removes error bars



V4.set_title('Visits by Borough')

V4.set_xlabel('Borough')                                  

V4.set_ylabel('Visits')
V9=sns.barplot(x='neighbourhood',

               y='Visits',

estimator=np.sum,

data=Data,

ci=None,           

order=Data.neighbourhood.value_counts().iloc[:10].index)



V9.set_title(' Total Visits by Neighbourhood')

V9.set_xlabel('Neighbourhood')                                  

V9.set_ylabel('Visits')

V9.set_xticklabels(V9.get_xticklabels(), rotation=45);
sns.set(style="whitegrid")                                     #Setting a new style

V6 = sns.barplot(

    x='neighbourhood', y='Visits', 

    estimator=np.mean,                                         # "mean" function from numpy as estimator

    data=Data,                                                 # Raw dataset fed directly to Seaborn

    ci=None,                               

    order=Data.neighbourhood.value_counts().iloc[:10].index)   #Top 10 Neighbourhoods only #Another Order function to get specific values order=Data['neighbourhood'].value_counts().index.tolist()[0:10]



V6.set_title('Avg. Visits by Neighbourhood')

V6.set_xlabel('Neighbourhood')                                  

V6.set_ylabel('Visits')

V6.set_xticklabels(V6.get_xticklabels(), rotation=45)
V7 = sns.barplot(x='room_type',

                 y='Visits',

                 estimator=np.sum,                                         

                 data=Data,

                 ci=None,

                 order=Data.room_type.value_counts().index)   



V7.set_title('Visits by Roomtype')                                

V7.set_xlabel('Room Type')

V7.set_ylabel('Visits')
rt = Data.groupby(['room_type'])               #Generate a table to look at the numbers, grouped by room_type

vrt = rt['Visits'].agg(np.sum).reset_index()   #aggregating the data with numpy sum function

vrt
price_bin=Data.price.value_counts(bins=[0,25,50,100,150,200,250,300,350,400,450,500,1000,2000,5000,10000])  #Using binning function to see listings fall in what price range

price_bin
V8=price_bin.plot(kind ='bar')

V8.set_title('Listings by Price Range')

V8.set_ylabel('Listings')

V8.set_xlabel('Price Range')

V8.set_xticklabels(V8.get_xticklabels(), rotation=45)
Price_by_NG =Data.groupby(                                          #Groupby Borough

   ['neighbourhood_group'], as_index=False                                

).agg(

    {

         'Visits':sum,

         'price':'mean'

    }

)



Price_by_NG = np.round(Price_by_NG, decimals=2)                     #Function to generate avg_price with only upto two decimals

Price_by_NG = Price_by_NG.rename(columns = {"price" : "Avg_Price"}) #Switching the column name to avg_price

Price_by_NG = Price_by_NG.sort_values('Visits',ascending=False)     #Sorting values by descending for Visits

Price_by_NG
sns.catplot(x='Avg_Price' , y='Visits', hue='neighbourhood_group', data=Price_by_NG, height=6, aspect=2);
Price_by_NG1 =Data.groupby(                                          #Groupby Borough

   ['neighbourhood_group', 'room_type'], as_index=False                                

).agg(

    {

         'Visits':sum,

         'price':'mean'

    }

)



Price_by_NG1 = np.round(Price_by_NG1, decimals=2)                     #Function to generate avg_price with only upto two decimals

Price_by_NG1 = Price_by_NG1.rename(columns = {"price" : "Avg_Price"}) #Switching the column name to avg_price

Price_by_NG1 = Price_by_NG1.sort_values('Visits',ascending=False)     #Sorting values by descending for Visits

Price_by_NG1
sns.relplot(x='Avg_Price' , y='Visits', hue='neighbourhood_group',col='room_type', data=Price_by_NG1);
Price_by_NG2 =Data.groupby(                                          

   ['neighbourhood','room_type'], as_index=False                                

).agg(

    {

         'Visits':sum,

         'price':'mean'

    }

)



Price_by_NG2 = np.round(Price_by_NG2, decimals=2)                     

Price_by_NG2 = Price_by_NG2.rename(columns = {"price" : "Avg_Price"}) 

Price_by_NG2 = Price_by_NG2.sort_values('Visits',ascending=False)

Price_by_NG2 = Price_by_NG2.head(10)

sns.catplot(x='Avg_Price' , y='Visits', hue='neighbourhood', col='room_type',aspect=2, data=Price_by_NG2 );

Price_by_NG2
Price_by_N =Data.groupby(

   ['neighbourhood'], as_index=False

).agg(

    {

         'Visits':sum,

         'price':'mean'

    }

)



Price_by_N = np.round(Price_by_N, decimals=2)

Price_by_N = Price_by_N.sort_values('Visits',ascending=False)



Price_by_N = Price_by_N.head(10)



Price_by_N
fig,ax = plt.subplots()                                                             # create figure and axis objects with subplots()

ax.plot(Price_by_N.neighbourhood, Price_by_N.Visits, color="green", marker="o")     # make a plot

ax.set_xlabel("Neighbourhood",fontsize=14)                                          # set x-axis label

ax.set_ylabel("Visits",color="green",fontsize=14)                                   # set y-axis label

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)



ax2=ax.twinx()                                                                      # twin object for two different y-axis on the sample plot

ax2.plot(Price_by_N.neighbourhood, Price_by_N.price,color="blue",marker="s")        # make a plot with different y-axis using second axis object

ax2.set_ylabel("Avg_Price",color="blue",fontsize=14)

plt.show()

# save the plot as a file

#fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',

            #format='jpeg',

            #dpi=100,

            #bbox_inches='tight')'''
fig,ax = plt.subplots()

ax.plot(Price_by_NG.neighbourhood_group, Price_by_NG.Visits, color="green", marker="o")

ax.set_xlabel("Borough",fontsize=14)

ax.set_ylabel("Visits",color="green",fontsize=14)



ax2=ax.twinx()

ax2.plot(Price_by_NG.neighbourhood_group, Price_by_NG.Avg_Price,color="blue",marker="s")

ax2.set_ylabel("Avg_Price",color="blue",fontsize=14)

plt.show()