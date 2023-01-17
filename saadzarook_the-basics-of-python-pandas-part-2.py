#We will continue the series with our previous dataset



import pandas as pd



realEstTrans = pd.read_csv('../input/Sacramentorealestatetransactions.csv')
realEstTrans.head()
#Now let's look at making some changes to our dataset

#Let's try to  add a column

#We will



realEstTrans['PerSqFt'] = realEstTrans['price'] / realEstTrans['sq__ft']

realEstTrans.head()

#As you can see we created a new column which has the price per square feet.

#So this is really useful when you have to shape you data efficiently

#We can also drop certain columns that we will not be needing for operations

#We will use the .drop(columns='') method for this



realEstTrans = realEstTrans.drop(columns=['PerSqFt'])

realEstTrans.head()

#As you can see we have dropped the column we created a while back

#Notice the assignment i have made in 'realEstTrans = realEstTrans.drop(columns=['PerSqFt'])'

#This is because if we don't assign, the  original dataset will not change.Try it out

#Lets create the previous column again

realEstTrans['PerSqFt'] = realEstTrans['price'] / realEstTrans['sq__ft']

realEstTrans.head()
#As you can see any columns that we create go far to the end. 

#We need to make outr dataset well organized, else it's really stressful !!



#Lets decide how our colums should be arranged



cols = list(realEstTrans.columns.values)

realEstTrans = realEstTrans[cols[0:10] + [cols[-1]] + cols[10:12]]

realEstTrans.head()

#As you can see we have moved the last column just next to 'price'

#Try to understand the above lines

#We have used column indexes to achieve this
#Let's try to export our modified dataset into csv,excel and txt formats



realEstTrans.to_csv('New_realEstTrans.csv', index=False)

#This will create the file 'New_realEstTrans.csv' in your root folder



#realEstTrans.to_excel('New_realEstTrans.xlxs', index=False)

#realEstTrans.to_excel('New_realEstTrans.xlxs', index=False, sep='\t')



#Try out the other codes top export the dataset in different formats.

#OK, Lets end for this part and continue the series with more operations on the datasets