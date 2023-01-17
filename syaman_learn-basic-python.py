# with this kernel we will learn basic python command like matplotlib, pandas, numpy, dictionary...
#first we will extract command as you see below
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

mydata = pd.read_csv("../input/tmdb_5000_movies.csv")
mydata.info()
mydata.corr()
#in correlation Map, we can see correlation between variable by color
f,ax = plt.subplots(figsize = (12,12))
sns.heatmap(mydata.corr(),annot=True,linewidths=1,fmt=".1f",ax=ax)
#we will see first seven data with this command
mydata.head(7)
#data's columns names
mydata.columns
#we can see correlation between popularity and vote average with these plotting command
mydata.popularity.plot(kind = "line",color = "g",label = "popularity",linewidth=1,alpha=0.5,grid=True,linestyle="-")
mydata.vote_average.plot(kind = "line",color = "r",label = "vote_average",linewidth=1,alpha=0.5,grid=True,linestyle=":")
plt.legend(loc="upper right") #to choose label place
plt.xlabel("x axis")
plt.ylabel("y axix")
plt.title("line plot") 



#scatter plot
mydata.plot(kind = "scatter",x="popularity",y="budget",alpha=0.5,color="red")
plt.xlabel("popularity")
plt.ylabel("budget")
plt.title("ploting of popularity and budget")
#plotting histogram
mydata.budget.plot(kind="hist",bins=50,figsize=(8,8))

# clf() = cleans it up again you can start a fresh
mydata.budget.plot(kind = "hist",bins = 50)
plt.clf()
#dictionary
#create dictionary and look its keys and values
dictionary = {"Poland" : "wroclaw","spain" : "barcelona","italy":"rome"}
print(dictionary.keys())
print(dictionary.values())
#we can update values
dictionary["italy"]="venice"
print(dictionary.values())

#we can add new keys and values, these new values and keys will be added end of the dictionary
dictionary["germany"]="berlin"
print(dictionary)

#we can delete part of dictionary with "del" command
del dictionary["germany"]
print(dictionary)
#we can check part of dictionary someting include or no inside of the dictionary
print('france' in dictionary) # Answer will be "False" because there is not this value in dictionary
#we can delete all variables of dictionary with "clear" command
dictionary.clear()
print(dictionary)
#before we extracted csv file by using command pandas "mydata=pd.read_csv("")"
# in pandas generally it uses series and data frame
#in here we will see how we can use both series and data frame
series=mydata["budget"]
print(type(series))

dataFrame = mydata[["vote_average"]]
print(type(dataFrame))

#we will make a break litle bit to explain pandas
#we will use some basic comparison and boolian operators
print(1 > 2)
print(5!=2)
# Boolean operators
print(False and True )
print(True or False)
#Filtering Pandas data frame
x = mydata["budget"]>270000000 # we will see 3 movies budget bigger than 270000000
mydata[x]
#Filtering pandas with logical_and
mydata[np.logical_and(mydata["budget"]>270000000, mydata["id"]>300 )]
# or we can apply this filter like below
mydata[(mydata["budget"]>270000000) & (mydata["id"]>300 )]
i = 0
while i != 7 : # or we can use this while (i<8)
    print('i is: ',i)
    i +=1 
print(i,' is equal to 7')
i = 0
while (i<8):
    print('i is: ',i)
    i +=1 
print(i,' is equal to 7')
list = [1,2,3,4,5]
for i in list: # we will see what is include inside of the list
    print('i is: ',i)
print('')
#or we can see index and values of the list
for index, value in enumerate(list): # enumarate will provide us to see the index of the values
    print(index," : ",value)
print('')
# we can use the same things as a mentioned upside for dictionary
dictionary = {'spain':'madrid','france':'paris',"poland":"wroclaw"} # we can use '' or ""
for key,value in dictionary.items(): 
    print(key," : ",value)
print('')


# For pandas we can achieve index and value, we will see 5th and 6th column's budget of the mydata 
for index,value in mydata[['budget']][5:7].iterrows(): 
    print(index," : ",value)