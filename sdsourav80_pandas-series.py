my_list=['kodu','jodu',3,4]
import pandas as pd
pd.Series(my_list)
lottery=[4,6,7,4,8]

pd.Series(lottery)
#this ones is for the datatype
registrations=[True,False,True,False]
pd.Series(registrations)
#index is generated from 0
#index can be of any data type
webster={'aadvark':'an animal',
          'Banana':'A delicious fruit',
          'Cyan':'A colour'}
pd.Series(webster)
#dtype only refer to the values in both cases the list or the dictionary
#INdex can be of any type
# attributes gives the information or the details about the object or the summary
#attributes does not modify view, review,observe etc are done by the attributes
#opposite to methods
me=['charming', 'joss','handsome','brilliant' ]
s=pd.Series(me) #store it in a variable
s
#now comes the attributes
#moves out of the pandas library
s.values #return as an array

s.index #gives the information about the index
s.dtype #o stands for the obect
#we look at the values piece by piece by index,values,and the dtype
#pd.Series is also a method because it calls the pandas library and do some manipulation
prices=[1.2,2.3,4.5]
s=pd.Series(prices)
s
s.sum() #method reqires parenthesis
s.product()
s.mean()
s.median()
s.mode()
#think of something like a video game
# dificulty setting - Easy, Medium, Hard
# volume - 1 to 10
# subtitle- True or false
# the option itself is the parameter like the difficulty its a setting something we can provide too
# option or choices are the argument like easy medium etc

fruits=['apple','orange','plum','grape','blueberry']
weekdays=['monday','tuesday','wednesday','thursday','friday']

pd.Series(fruits,weekdays)
pd.Series(data=fruits,index=weekdays) # allows you to skip sequential arguments
pd.Series(fruits,index=weekdays)
fruits=['apple','orange','plum','grape','blueberry','watermelon']
weekdays=['monday','tuesday','wednesday','thursday','friday','monday']
pd.Series(data=fruits,index=weekdays)
pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
pd.read_csv("../input/google-play-store-apps/googleplaystore.csv",usecols=['App'])
app=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv",usecols=['App'],squeeze=True)
app
rating=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv",usecols=['Rating'],squeeze=True)
rating
app.head() #giving us a five value series 

rating.tail()
len(app)
len(rating)
type(app)
type(rating)
dir(app)
sorted(rating)
sorted(app)
list(app)
list(rating)
dict(app)
max(app)
min(app)
max(rating)
min(rating)
app.dtype
rating.dtype
app.is_unique
app.index
app.name= 'apps'
app.size
app.ndim
app.shape
app.values
app.sort_values().head()
app.sort_values(ascending=False)
app.sort_values(ascending=False).tail()
rating.sort_values(ascending=False).head()
rating
rating.sort_values(ascending=False,inplace=True)
rating
rating=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv",usecols=['Rating'],squeeze=True)
app=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv",usecols=['App'],squeeze=True)
rating.sort_values(ascending=False,inplace=True)
rating.head()
rating.sort_index(ascending=True,inplace=True)
rating.head()
100 in rating
200 in rating
100 in app
d=pd.read_csv("../input/los-angeles-metro-bike-share-trip-data/metro-bike-share-trip-data.csv",usecols=['Duration'],squeeze=True)
d.sort_values()
124124124 in rating
