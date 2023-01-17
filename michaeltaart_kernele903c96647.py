# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from pandas import read_csv, notnull, DataFrame

class Yelp():
    """class for Yelp database"""
    def __init__(self, path = "../input/yelp_business.csv"):
        self.filename = path
        #read file in csv format into dataframe
        self.df = read_csv(self.filename)
        #print first few rows in dataframe
        print(self.df.head())
        
        
class Charts(Yelp):
    """Helper class for creating charts"""
    def __init__(self):
        """default constructor"""
        pass
        #self.df = df
    def group_top_buisness_cities_chart(self):
        #top 10 buisness cities
        #sf = df[["stars"]]
        sf = DataFrame({"size": self.df.groupby(["city"]).size()})
        return sf.sort_values(by="size", ascending=False)[:10]
        #top_10_buisness_cities.plot.bar()       
    def top_buisness_states(self):     
        """top 10 buisness states"""
        sf = DataFrame({"size": self.df.groupby(["state"]).size()})
        return sf.sort_values(by="size", ascending=False)[:10]           
    def review_counts_by_city(self):
         #top 10 cities id with review
        sf = DataFrame({"size": self.df.groupby(["city","review_count"]).size()})
        return  sf.sort_values(by="size", ascending=False)[:10]
    def top_cities_with_starts(self):
        #Top 10 cities with highest stars
        sf = DataFrame({"size": self.df.groupby(["city", "stars"]).sum()})        
        return sf.sort_values(by="size", ascending=False)[:10]        
class Compute(Charts):
    """child class inherts charts and used for accessing methods in parent class"""
    def __init__(self, path = "../input/yelp_business.csv"):
        self.filename = path
        objYelp = Yelp(self.filename)
        self.df = objYelp.df
        self.chart = Charts()        
    def describe_data(self): #mean average sum on multiple columns
        return self.df.describe()
        
    
objCompute = Compute()

%matplotlib inline
top_10_buisness_cities = objCompute.group_top_buisness_cities_chart()
top_10_buisness_cities.plot.bar()

top_10_cities_review_count = objCompute.review_counts_by_city()
top_10_cities_review_count.plot.bar()

top_buisness_states= objCompute.top_buisness_states()
top_buisness_states.plot.bar()




objCompute.df.hist()


print(objCompute.describe_data())