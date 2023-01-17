import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from sklearn.cross_validation import ShuffleSplit
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (18.0, 9.0)
data = pd.read_csv('../input/startup_funding.csv')
def getCityWise():
    print("Information on data, Citywise")
    cityNames = data['CityLocation']
    cityNames=cityNames.dropna()
    citygroups=data[['CityLocation','AmountInUSD']].groupby('CityLocation').agg(['count'])
    fig_ops,((ax1))=plt.subplots(1,1,sharex=True,facecolor='w')
    cities = citygroups['AmountInUSD','count'].sort_values(ascending=False)
    cities.head(15).plot(kind='bar',title='Number of startups funded City in India',ax=ax1,grid=True,rot=90)
def getIndustryVerticalWise():
    print("Information on data, IndustryVertical")
    cityNames = data['IndustryVertical']
    cityNames=cityNames.dropna()
    citygroups=data[['IndustryVertical','AmountInUSD']].groupby('IndustryVertical').agg(['count'])
    fig_ops,((ax1))=plt.subplots(1,1,sharex=True,facecolor='w')
    cities = citygroups['AmountInUSD','count'].sort_values(ascending=False)
    cities.head(15).plot(kind='line',title='Rate of growth to fund IndustryVertical',ax=ax1,grid=True,rot=90)
def getInvestmentTypewise():
    print("Information on data, InvestmentType")
    cityNames = data['InvestmentType']
    cityNames=cityNames.dropna()
    citygroups=data[['InvestmentType','AmountInUSD']].groupby('InvestmentType').agg(['count'])
    fig_ops,((ax1))=plt.subplots(1,1,sharex=True,facecolor='w')
    cities = citygroups['AmountInUSD','count'].sort_values(ascending=False)
    cities.head(15).plot(kind='pie',title='Amount distribution for investors',ax=ax1,rot=90)

    
getIndustryVerticalWise()
getCityWise()
getInvestmentTypewise()
