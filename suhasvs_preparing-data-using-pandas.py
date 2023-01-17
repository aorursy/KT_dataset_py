# importing the libraires

import pandas as pd
# read the (sales-jan-2015) dataset.



jan=pd.read_csv("../input/sales-ds/sales-jan-2015.csv - sales-jan-2015.csv.csv",index_col=0)

jan.head()
# read the datasets sepatetly (sales-jan-2015 and sales-feb-2015)



### START CODE HERE : 

jan=pd.read_csv("../input/sales-ds/sales-jan-2015.csv - sales-jan-2015.csv.csv",index_col=0)

feb=pd.read_csv("../input/sales-ds/sales-feb-2015.csv - sales-feb-2015.csv.csv",index_col=0)



### END CODE
jan.head()
feb.head()
# read the datasets ("sales-feb-2015","sales-jan-2015") using a loop



### START CODE HERE : 



filename=["../input/sales-ds/sales-feb-2015.csv - sales-feb-2015.csv.csv","../input/sales-ds/sales-jan-2015.csv - sales-jan-2015.csv.csv"]

dataframes=[]



for f in filename:

    

    dataframes.append(pd.read_csv(f,index_col=0))

                      

print(dataframes)



### END CODE
# read the datasets ("sales-jan-2015" ) using a glob



### START CODE HERE : 



from glob import glob



filename=glob("sales*.csv")



dataframes=[ (pd.read_csv(f,index_col=0))    for f in filename]



dataframes



### END CODE
# Create a dataframe

data = {'county': ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'], 

        'year': [2012, 2012, 2013, 2014, 2014], 

        'reports': [4, 24, 31, 2, 3]}

df = pd.DataFrame(data)

df
df.index=["Co","Pi","Sc","Ma","Yu"]
#sort the table(data) with columns



### START CODE HERE : 



df.sort_values("county")



### END CODE
df.sort_values("year")
df.sort_values("reports")
# sorting table with index



### START CODE HERE : 

df.sort_index()

### END CODE
# Change the order (the index) of the rows



### START CODE HERE : 

df.sort_index(ascending=False)  # by sorting the index with ascending =False I am changing the order of observations/rows.



### END CODE
# Creating a new dataframe to have the index to dataframe "df"



data2 = {'county': ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'], 

        'year': [2012, 2012, 2013, 2014, 2014], 

        'reports': [4, 24, 31, 2, 3]}

df2=pd.DataFrame(data2)

df2.index=["Pi","Co","Ma","Sc","Yu"]

df2
# reindexing from the dataframe index



### START CODE HERE : 

df.reindex(df2.index)

### END CODE
# load the date sets separately(sales-feb-2015.csv & sales-jan-2015.csv)



### START CODE HERE : 

jan=pd.read_csv("../input/sales-ds/sales-jan-2015.csv - sales-jan-2015.csv.csv",index_col=0)

feb=pd.read_csv("../input/sales-ds/sales-feb-2015.csv - sales-feb-2015.csv.csv",index_col=0)



### END CODE
#display the first five rows of data



### START CODE HERE : 

jan.head(5)

### END CODE
feb.head(5)
#mean value of both the data set



### START CODE HERE : 



jan['Units'].mean()



### END CODE
feb['Units'].mean()
# percentage change to 100%



### START CODE HERE : 

jan['Units'].pct_change()*100

### END CODE
feb['Units'].pct_change()*100
# Adding sales1 and sales2 by using the add function



### START CODE HERE : 

add_of_sales_JanFeb=jan['Units'].sum()+jan['Units'].sum()

add_of_sales_JanFeb

### END CODE