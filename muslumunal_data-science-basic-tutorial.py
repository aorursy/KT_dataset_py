# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/googleplaystore.csv")
data.info()
data.head() #view the top five data
series = data['Category'] #type series

print(type(series))

data_frame = data[['Category']] #type data frame

print(type(data_frame))
#Filter Pandas data frame

control_rating = data['Rating']>4.0  #list with rating greater than 4

data[control_rating]
len(data[control_rating]) #total number of ratings with a rating greater than 4
#Filtering Pandas with logical_and

data[np.logical_and(data['Rating']>4.0, data['Content Rating']=='Teen')]
#we can also use '&' for filtering.

data[(data['Rating']>4.0) & (data['Content Rating'] == 'Teen')]
data['Rating'].mean() #average of ratings
data[data['Rating'] == data['Rating'].max()] #application with the highest rating
data[data['Rating'] == data['Rating'].min()] #applications with the lowest rating
data["Category"].nunique() #total number of categories
data.groupby("Category").mean() #average number of ratings in categories
data['Category'].value_counts() #Number of applications in categories
data.drop(['Android Ver'],axis = 1,inplace = True) #drop 'Android Ver' column

data.info()
import warnings

warnings.filterwarnings("ignore")

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

data2 = data2.set_index("date")

data2
#resample with year

data2.resample("A").mean()
#resample with month

data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
x = np.arange(1,6)

y = np.arange(2,11,2)

plt.plot(x,y,"red")

plt.show()
plt.subplot(2,2,1)

plt.plot(x,y,"blue")



plt.subplot(2,2,2)

plt.plot(y,x,"red")



plt.subplot(2,2,3)

plt.plot(x**2,y,"black")





plt.subplot(2,2,4)

plt.plot(x,y**2,"green")

plt.show()
fig = plt.figure()

axes = fig.add_axes([0.1,0.2,0.4,0.6])

axes.plot(x,y)

axes.set_xlabel("x axis")

axes.set_ylabel("y axis")

axes.set_title("Plot")



plt.show()
int_reviews = []



for i in data['Reviews']:

    try:

        int_reviews.append(int(i))

    except ValueError:

        int_reviews.append(0)



data['IntReviews'] = int_reviews #add new column

data.info()
x = data['Rating']

y = data['IntReviews']

fig = plt.figure()

axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(x,y,"red",linewidth=2,linestyle="--",marker="o",markersize=5,markerfacecolor="black",markeredgecolor="yellow",markeredgewidth=2)

plt.show()
data.Rating.plot(kind="line",color="y",label="Rating",linewidth=1,alpha=0.5,grid=True,linestyle=":")

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Rating Plot')

plt.show()
data.IntReviews.plot(kind="line",color="b",label="Rating",linewidth=5,alpha=1,grid=True,linestyle="-",figsize=(12,12))

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Rating Plot')

plt.show()
data.plot(kind='scatter',x='IntReviews', y='Rating', alpha = 0.5, color='green')

plt.xlabel('Reviews')

plt.ylabel('Rating')

plt.title('Scatter Plot')

plt.show()
data.Rating.plot(kind='hist',bins=100,figsize=(12,12))

plt.show()
data["rating_status"] = ["Popular" if i > 4.0 else "Not Popular" for i in data.Rating]



print(data.loc[:10,["rating_status","Rating"]])
numbers1 = np.arange(1,6)

print([i+10 for i in numbers1])
numbers2 = np.arange(1,25)



print(["Even Number" if i%2==0 else "Odd Number" for i in numbers2])
# classic function

def square_area_classic(x):

    return x**2



#lambda function

square_area_lambda = lambda x: x**2



#use functions

print(square_area_classic(10))

print(square_area_lambda(10))

#use multiple veriables

rectangle_area = lambda x,y: x*y

print(rectangle_area(3,5))
#exercise lambda function

reverse_str = lambda s : s[::-1]

print(reverse_str("Data Science"))
def calculate(v_r,v_pi = 3.14):

    result = v_pi * v_r**2

    return result

print(calculate(3)) # v_r = 3 , v_pir = 3.14(default) 

print(calculate(4,3)) # change default arguments
def flexible(*args):

    for i in args:

        print(i**2)

flexible(1,2,3)
data.head() #show first 5 rows

data.tail() #show last 5 rows
data.columns #gives column names of features
data.shape #gives number of rows and columns in a tuble
data.info() #gives data type like dataframe, number of sample or row, number of feature or column
data.describe()
print(data['Rating'].value_counts(dropna=False))
first_five_data = data.head()

melted = pd.melt(frame=first_five_data,id_vars='App',value_vars=['Rating','Type'])

print(melted)
data.boxplot(column="Rating",by ="rating_status")
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
data.dtypes
#convert object(str) to categorical

data['Type'] = data['Type'].astype('category')

data.dtypes
#look at does data have nan value

data.info()
#check Rating

data["Rating"].value_counts(dropna=False)
#drop nan values

data1 = data

data1["Rating"].dropna(inplace=True)
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
assert  data['Rating'].notnull().all() # returns nothing because we drop nan values
data["Rating"].fillna('empty',inplace = True)
assert  data['Rating'].notnull().all() # returns nothing because we do not have nan values