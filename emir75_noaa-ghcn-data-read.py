import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tarfile

import gzip

import statsmodels as st

import os

print(os.listdir("../input"))
#open year files and select station data in TMAX,TMIN,TAVG,PRCP,SNWD

def select_year(year = "2018",station_name = "TUM00017204",types = "TUM00017204"):

    #read year files

    year_files = "../input/ghcnd_all_years/"

    with gzip.open(year_files+year+".csv.gz",'rb') as year_file:

        i = 0

        for line in year_file:

                #print(line)

            i+=1

            if i > 4: break

        data = pd.read_csv(year_file, sep=r'\s+',index_col=None,delimiter=",")

#SELECT STATİON

    data_n = data.iloc[:,:4].values

    df = pd.DataFrame(data_n,columns = ["Station","Year","Type","Values"])

    #df = (df.set_index(['Station',"Year","Type", df.groupby(['Station',"Year","Type"]).cumcount()]).unstack().sort_index(axis=1, level=1))

    #df.columns = ['{}_{}'.format(i, j) for i, j in df.columns]

    #df = df.reset_index()

    df.set_index("Station",inplace = True)

    station_list = df.loc[station_name]

    station_list.set_index("Type",inplace = True)

    select_type = station_list.loc[types]

#DATE TİME 

    s = np.array(select_type)

    type_df = pd.DataFrame(s,columns = ["Years","Values"])

    end_num = len(type_df)

    new_year = []

    for i in range(0,end_num):

        a = type_df["Years"]

        b = str(a[i])

        new_year.append(b[:4]+"."+b[4:6]+"."+b[6:8])

    new_year_df = pd.DataFrame(new_year,columns=["NewYear"])

    new_df = pd.concat([new_year_df,type_df],axis = 1)

    #new_df = df.drop(["Years"],axis = 1)

    new_df['NewYear']=pd.to_datetime(new_df['NewYear'], format='%Y%m%d', errors='ignore')

    new_df['Year']=pd.DatetimeIndex(new_df['NewYear']).year

    new_df['Month']=pd.DatetimeIndex(new_df['NewYear']).month

    new_df['Day']=pd.DatetimeIndex(new_df['NewYear']).day

    new_df.to_csv(year+types+".csv",index = False)



    print("END")

#SELECT ALL TMAX,TMIN,TAVG,PRCP,SNWD DATA

def all_type(year = "2018"):

    

    types = ["TAVG","TMIN","TMAX","PRCP","SNWD"]

    for i in types:

        select_year(year = "2018",station_name="TUM00017096",types=i)

    tmax = pd.read_csv(year+"TMAX.csv")

    tmın = pd.read_csv(year+"TMIN.csv")

    tavg = pd.read_csv(year+"TAVG.csv")

    prcp = pd.read_csv(year+"PRCP.csv")

    snwd = pd.read_csv(year+"SNWD.csv")

    print("Save file list:\n",os.listdir().sort())

#STATİON SEARCH BY YEAR

def station_searc(year = "2018",station_name = "TUM00017204"):

    year_files = "../input/ghcnd_all_years/"

    with gzip.open(year_files+year+".csv.gz",'rb') as year_file:

        i = 0

        for line in year_file:

            i+=1

            if i > 4: break

        data = pd.read_csv(year_file, sep=r'\s+',index_col=None,delimiter=",")

    data_n = data.iloc[:,:4].values

    df = pd.DataFrame(data_n,columns = ["Station","Year","Type","Values"])

    df.set_index("Station",inplace = True)

    try:

        station_list = df.loc[station_name]   

        print("Stataion Found..")

    except:

        print("Stataion Not Found..")

#def visulation():

    

    
#year station select

for i in range(2000,2019):

    station_searc(year = str(i),station_name="TUM00017069")

    print(i)
liste = ["2018","2017","2016","2015","2014","2013","2012","2011","2010","2009","2008","2007"]

for i in liste:

    select_year(year = i , station_name="TUM00017069" , types="TAVG")

    print(i)

    
os.listdir()
avg_2008 = pd.read_csv("2008TAVG.csv")

avg_2009 = pd.read_csv("2009TAVG.csv")

avg_2010 = pd.read_csv("2010TAVG.csv")

avg_2011 = pd.read_csv("2011TAVG.csv")

avg_2012 = pd.read_csv("2012TAVG.csv")

avg_2013 = pd.read_csv("2013TAVG.csv")

avg_2014 = pd.read_csv("2014TAVG.csv")

avg_2015 = pd.read_csv("2015TAVG.csv")

avg_2016 = pd.read_csv("2016TAVG.csv")

avg_2017 = pd.read_csv("2017TAVG.csv")

avg_2018 = pd.read_csv("2018TAVG.csv")
avg_2008["Tavg"] = avg_2008["Values"] / 10

avg_2009["Tavg"] = avg_2009["Values"] / 10

avg_2010["Tavg"] = avg_2010["Values"] / 10

avg_2011["Tavg"] = avg_2011["Values"] / 10

avg_2012["Tavg"] = avg_2012["Values"] / 10

avg_2013["Tavg"] = avg_2013["Values"] / 10

avg_2014["Tavg"] = avg_2014["Values"] / 10

avg_2015["Tavg"] = avg_2015["Values"] / 10

avg_2016["Tavg"] = avg_2016["Values"] / 10

avg_2017["Tavg"] = avg_2017["Values"] / 10

avg_2018["Tavg"] = avg_2018["Values"] / 10
d = [avg_2008,avg_2009,avg_2010,avg_2011,avg_2012,avg_2013,avg_2014,avg_2015,avg_2016,avg_2017,avg_2018]

n = ["2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018"]

for i in d:

    plt.figure(figsize = (20,3))

    sns.lineplot(i["Day"],i["Tavg"])

    plt.show()
plt.figure(figsize = (20,4))

sns.lineplot(avg_2008["Day"],avg_2008["Tavg"])

sns.lineplot(avg_2009["Day"],avg_2009["Tavg"])

sns.lineplot(avg_2010["Day"],avg_2010["Tavg"])

sns.lineplot(avg_2011["Day"],avg_2011["Tavg"])

sns.lineplot(avg_2012["Day"],avg_2012["Tavg"])

sns.lineplot(avg_2013["Day"],avg_2013["Tavg"])

sns.lineplot(avg_2014["Day"],avg_2014["Tavg"])

sns.lineplot(avg_2015["Day"],avg_2015["Tavg"])

sns.lineplot(avg_2016["Day"],avg_2016["Tavg"])

sns.lineplot(avg_2017["Day"],avg_2017["Tavg"])

sns.lineplot(avg_2018["Day"],avg_2018["Tavg"])
m = [avg_2008,avg_2009,avg_2010,avg_2011,avg_2012,avg_2013,avg_2014,avg_2015,avg_2016,avg_2017,avg_2018]



for i in m:

    plt.figure(figsize = (20,3))

    sns.lineplot(i["Month"],i["Tavg"])

    plt.show()
plt.figure(figsize = (20,4))

sns.lineplot(avg_2008["Month"],avg_2008["Tavg"])

sns.lineplot(avg_2009["Month"],avg_2009["Tavg"])

sns.lineplot(avg_2010["Month"],avg_2010["Tavg"])

sns.lineplot(avg_2011["Month"],avg_2011["Tavg"])

sns.lineplot(avg_2012["Month"],avg_2012["Tavg"])

sns.lineplot(avg_2013["Month"],avg_2013["Tavg"])

sns.lineplot(avg_2014["Month"],avg_2014["Tavg"])

sns.lineplot(avg_2015["Month"],avg_2015["Tavg"])

sns.lineplot(avg_2016["Month"],avg_2016["Tavg"])

sns.lineplot(avg_2017["Month"],avg_2017["Tavg"])

sns.lineplot(avg_2018["Month"],avg_2018["Tavg"])

plt.show()
data = [avg_2008,avg_2009,avg_2010,avg_2011,avg_2012,avg_2013,avg_2014,avg_2015,avg_2016,avg_2017,avg_2018]

os.listdir()
data = [avg_2008,avg_2009,avg_2010,avg_2011,avg_2012,avg_2013,avg_2014,avg_2015,avg_2016,avg_2017,avg_2018]

q = []

for i in data:

    i = i.mean()

    df = pd.DataFrame(i)

    q.append(df)

df = pd.concat(q,axis = 1)

df_year = df.loc["Year"]

df_tavg = df.loc["Tavg"]

new_df = pd.concat([df_year,df_tavg],axis = 1)

new_year = []

end_num = len(new_df)

for i in range(0,end_num):

    string = str(new_df.iloc[i,0])

    n_str = string[:4]

    new_year.append(n_str)

new_df = new_df.drop(["Year"],axis = 1)

n_year = pd.DataFrame(new_year)

n_df = pd.DataFrame(new_df)

n_year = np.array(n_year).reshape(-1,1)

n_df = np.array(n_df).reshape(-1,1)



df = pd.concat([pd.DataFrame(n_year,columns = ["Year"]),pd.DataFrame(n_df,columns = ["Tavg"])],axis = 1)

df.set_index("Year",inplace = True)

df.sort_index()



plt.figure(figsize = (20,4))

plt.title("Annual Templature")

plt.xlabel("Year")

plt.ylabel("Temp(C)")

sns.lineplot(df.index,df["Tavg"])

plt.show()



plt.figure(figsize = (20,4))

sns.distplot(df["Tavg"])

plt.title("Annual Temperature Distrubiton")

plt.xlabel("Temp(C)")

plt.show()