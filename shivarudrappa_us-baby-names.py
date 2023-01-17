# Required libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
# Read the dataset
usbabyname_df = pd.read_csv("../input/NationalNames.csv")
# Exploratory Data Analysis
# Top 10 data entries
usbabyname_df.head(10)
# Last 10 data entries
usbabyname_df.tail(10)
# Summary on given data set
usbabyname_df.describe()
# filter the dataset as top 10 records of given year = 2014 and Gender = Male
babyboynames_2014 = usbabyname_df.loc[(usbabyname_df['Year']==2014)&(usbabyname_df['Gender']=="M")]
topboynames_2014 = babyboynames_2014[0:10]
boyname_2014 = topboynames_2014.loc[:,['Name','Count']]
boyname_2014.index = range(10) # rename index
boyname = boyname_2014['Name'].values.T.tolist() ## extract the names and convert dataframe into list

## take a values in the dic format
boyname_2014_dic = boyname_2014.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
    
print (boyname_2014, "\n\n", boyname_2014_dic)

# filter the dataset as top 10 records of given year = 2014 and Gender = Female
babygirlnames_2014 = usbabyname_df.loc[(usbabyname_df['Year']==2014)&(usbabyname_df['Gender']=="F")]
topgirlnames_2014 = babygirlnames_2014[0:10]
girlname_2014 = topgirlnames_2014.loc[:,['Name','Count']]
girlname_2014.index = range(10) # rename index
girlname = girlname_2014['Name'].values.T.tolist() ## extract the names and convert dataframe into list
girlname_2014

## take a values in the dic format
girlname_2014_dic = girlname_2014.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
    
print (girlname_2014, "\n\n", girlname_2014_dic)

# check the popularity of the latest top 10 boy names over past 10 years
past_10year = np.unique(usbabyname_df['Year'])[-11:-1] ## get the last 10 years values past 10th year
#print (past_10year)

cols = ('Id','Name','Year','Gender','Count')
boyname_df= pd.DataFrame(columns = cols)## create empty dataframe
 # create empty dictionary
boyname_2004_dic, boyname_2005_dic, boyname_2006_dic, boyname_2007_dic, boyname_2008_dic = {},{},{},{},{}
boyname_2009_dic, boyname_2010_dic, boyname_2011_dic, boyname_2012_dic, boyname_2013_dic= {},{},{},{},{}
   
for year in past_10year:
    for name in boyname:
        boyname_df = usbabyname_df[(usbabyname_df['Year'] == year) 
                                 & (usbabyname_df['Name']  == name )
                                & (usbabyname_df['Gender']=='M')]
        if (year == 2004):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2004_dic[name] = boyname_df[name]
        elif (year == 2005):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2005_dic[name] = boyname_df[name]
        elif (year == 2006):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2006_dic[name] = boyname_df[name]
        elif (year == 2007):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2007_dic[name] = boyname_df[name]
        elif (year == 2008):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2008_dic[name] = boyname_df[name]
        elif (year == 2009):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2009_dic[name] = boyname_df[name]
        elif (year == 2010):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2010_dic[name] = boyname_df[name]
        elif (year == 2011):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2011_dic[name] = boyname_df[name]
        elif (year == 2012):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2012_dic[name] = boyname_df[name]
        elif (year == 2013):
            boyname_df = boyname_df.loc[:,('Name','Count')].set_index('Name').T.to_dict('list')
            boyname_2013_dic[name] = boyname_df[name]


print ("2004:\n", boyname_2004_dic,"\n2005\n", boyname_2005_dic,"\n2006\n", boyname_2006_dic,
      "\n2007\n", boyname_2007_dic, "\n2008\n", boyname_2008_dic, "\n2009\n", boyname_2009_dic,
      "\n2010\n", boyname_2010_dic,"\n2011\n", boyname_2011_dic,"\n2012\n", boyname_2012_dic,
      "\n2013\n", boyname_2013_dic)


x = ["Noah", "Liam", "Mason", "Jacob", "William", "Ethan", "Michael", "Alexander", "James", "Daniel"]
y_2014, y_2013, y_2012,y_2011,y_2010,y_2009 = [],[],[],[],[],[]
for name in x:
    a = boyname_2014_dic[name]
    a = int(str(a).strip('[]'))
    y_2014.append(a)
for name in x:
    a = boyname_2013_dic[name]
    a = int(str(a).strip('[]'))
    y_2013.append(a)
for name in x:
    a = boyname_2012_dic[name]
    a = int(str(a).strip('[]'))
    y_2012.append(a)
for name in x:
    a = boyname_2011_dic[name]
    a = int(str(a).strip('[]'))
    y_2011.append(a)
for name in x:
    a = boyname_2010_dic[name]
    a = int(str(a).strip('[]'))
    y_2010.append(a)
for name in x:
    a = boyname_2009_dic[name]
    a = int(str(a).strip('[]'))
    y_2009.append(a)


## compare top 10 records count of 2014 with the records count of 2013 
ind = np.arange(len(y_2014))
width = 0.5
   
fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(ind+width+0.35, y_2014, 0.45, color='#228b22')
ax2 = ax.twinx()
ax2.bar(ind+width, y_2013, width, color='#9acd32')

#ax.set_xticks(ind+width+(width/2))
ax.set_xticklabels(x,rotation='vertical')
ax.tick_params(axis='x', pad=8)
ax.set_title("Compare Top 10 Boy's name of 2014 with 2013" )
ax.set_ylabel("Count")
ax.set_xlabel ("Top 10 Boy's Names of 2014")

plt.tight_layout()
plt.show()


## compare top 10 records count of 2014 with the records count of 2012
ind = np.arange(len(y_2014))
width = 0.5
   
fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(ind+width+0.35, y_2014, 0.45, color='#a52a2a')
ax2 = ax.twinx()
ax2.bar(ind+width, y_2012, width, color='#f4a460')

#ax.set_xticks(ind+width+(width/2))
ax.set_xticklabels(x,rotation='vertical')
ax.tick_params(axis='x', pad=8)
ax.set_title("Compare Top 10 Boy's name of 2014 with 2012" )
ax.set_ylabel("Count")
ax.set_xlabel ("Top 10 Boy's Names of 2014")

plt.tight_layout()
plt.show()


## compare top 10 records count of 2014 with the records count of 2011
ind = np.arange(len(y_2014))
width = 0.5
   
fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(ind+width+0.35, y_2014, 0.45, color='#4682b4')
ax2 = ax.twinx()
ax2.bar(ind+width, y_2011, width, color='#afeeee')

#ax.set_xticks(ind+width+(width/2))
ax.set_xticklabels(x,rotation='vertical')
ax.tick_params(axis='x', pad=8)
ax.set_title("Compare Top 10 Boy's name of 2014 with 2011" )
ax.set_ylabel("Count")
ax.set_xlabel ("Top 10 Boy's Names of 2014")

plt.tight_layout()
plt.show()


## compare top 10 records count of 2014 with the records count of 2010
ind = np.arange(len(y_2014))
width = 0.5
   
fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(ind+width+0.35, y_2014, 0.45, color='#b03060')
ax2 = ax.twinx()
ax2.bar(ind+width, y_2010, width, color='#ffb6c1')

#ax.set_xticks(ind+width+(width/2))
ax.set_xticklabels(x,rotation='vertical')
ax.tick_params(axis='x', pad=8)
ax.set_title("Compare Top 10 Boy's name of 2014 with 2010" )
ax.set_ylabel("Count")
ax.set_xlabel ("Top 10 Boy's Names of 2014")

plt.tight_layout()
plt.show()


## compare top 10 records count of 2014 with the records count of 2009
ind = np.arange(len(y_2014))
width = 0.5
   
fig = plt.figure()
ax = fig.add_subplot(111)

ax.bar(ind+width+0.35, y_2014, 0.45, color='#8b8989')
ax2 = ax.twinx()
ax2.bar(ind+width, y_2009, width, color='#eee9e9')

#ax.set_xticks(ind+width+(width/2))
ax.set_xticklabels(x,rotation='vertical')
ax.tick_params(axis='x', pad=8)
ax.set_title("Compare Top 10 Boy's name of 2014 with 2009" )
ax.set_ylabel("Count")
ax.set_xlabel ("Top 10 Boy's Names of 2014")

plt.tight_layout()
plt.show()


## Same code canbe applied to 2014 top 10 girl dataset
