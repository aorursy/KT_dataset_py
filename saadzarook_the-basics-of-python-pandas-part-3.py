#We will continue from the previous dataset



import pandas as pd



realEstTrans = pd.read_csv('../input/Sacramentorealestatetransactions.csv')

realEstTrans.head(50)

#Let's try to do some filtering

#Let's say you want to get all the data where the 'street' has the word 'WAY'in it



realEstTrans.loc[realEstTrans['street'].str.contains('WAY')]

#Now will say all the results will have streets that contains the word 'WAY'

#realEstTrans.loc[~realEstTrans['street'].str.contains('WAY')]

#The above line with the character '~' is similar to a NOT operator. Try it out
#Lets's go a step further

#We can also use regex for more complex operations

#Say we need to know the rows where 'street' has either the words 'ST' or 'AVE'

import re

realEstTrans.loc[realEstTrans['street'].str.contains('st|ave', flags=re.I, regex=True)]



#Here we initialy import the regex library, then we use it in our method

#As you can see i have typed in lower case 'st|ave' even though the data is in UPPER

#But this can be a issue if the data set is very big where we do not know if there were

#any lower case characters. So we use the flags=re.I where we basically say to ignore the case

#Let's say you want to find all the records where the 'street' number begins from a 4

realEstTrans.loc[realEstTrans['street'].str.contains('^4[0-9]*', flags=re.I, regex=True)]



#Here we use the basic rules of regex as in python. Play around to know more
#Let's do some changes to the data

#Suppose we need to replace the 'type' Residential to Normal, we do the following

realEstTrans.loc[realEstTrans['type'] == 'Residential', 'type'] = 'Normal'

realEstTrans.head(10)

#There you go !! We can see 'Residential has been replaced with the word 'Normal'
#We can also do changes to other columns based on conditions of another column

#Let's say you want to change the 'type' to 'Luxury' if the price greater than 90000



realEstTrans.loc[realEstTrans['price'] > 90000, 'type'] = 'Luxury'

realEstTrans.head(10)



#So now you will see the type of house that has a price greater than 90K has been changed as 'Luxury'

#We also can have multiple condition as well as multiple parameters. Do some self exploration on that.
#Let's try to do some aggregate functions

#We can find the average price of different types of residences by using the .mean() method

realEstTrans.groupby(['type']).mean()

#We need only price but here you see, we get all the columns additional columns. 

#To solve that problem, we do the following



realEstTrans.groupby(['type']).mean()['price']

#We also can play with various way in this. Please refer the documentation to find out how !!
#That's it for this beginners series on the basics of Python Pandas. 

#I hope you would have got a clear idea on how the basics work.

#Leave an upvote if you liked this notenbook