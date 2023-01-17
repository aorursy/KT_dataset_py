import pandas as pd



#Now this is the default way on how we import libararies in python. 

#You may choose any meaningful name in place of 'pd'
#Now let's bring in our dataset



realEstTrans = pd.read_csv('../input/Sacramentorealestatetransactions.csv')

#To view the dataset in your workspace, we can simply call the name of the variable or use the print function



realEstTrans
#Now let's explore our dataset

#To view the headers or features we use the .columns method

realEstTrans.columns #or print(realEstTrans.columns) will work the same



#To view a specific column we use the column name we want to see



#realEstTrans['street'] #This will load all the data in the 'street column'

realEstTrans['street'][0:5] #We can also give a range of rows we need from the column we need



#realEstTrans.street even this method works when the header is a single word
#We can also view multiple columns by simple specifying the names

realEstTrans[['street','type','price']].head()

#To read specific row we can use the .iloc[] method

realEstTrans.iloc[2]

#realEstTrans.iloc[2:4] will give the the 3rd and 4th rows 

#Index begins from 0
#We can also use .iloc[] to read a specific data at a given location



realEstTrans.iloc[1,6] #This will give the data in the 2nd row, 7th column
realEstTrans.loc[realEstTrans['type'] == 'Condo']

#We can also use multiple conditions, try it our yourself
realEstTrans.describe()
#Let's sort our dataset

realEstTrans.sort_values('price').head() 
realEstTrans.sort_values('price', ascending=False).head()

#Asigning False or 0 to ascending will reverse sort the values
realEstTrans.sort_values(['price','beds'], ascending=[0,1]).head()

#We can also use multiple parameters to sort our data

#ascending=[0,1] means 'price' will be in descending and beds will be in ascending orders

#Try out with different parameters to get a good understanding