import numpy as np

import pandas as pd

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

import tarfile

import gzip

import re

import os

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))
def station_number(year = "2018",find_station = "010010-99999"):

    year_files = os.listdir("../input/gsod_all_years")

    year_files.sort()

    pathstem = "../input/gsod_all_years/"

    with tarfile.open(pathstem +"gsod_"+year+".tar") as tar:

        i = 0

        for tarinfo in tar:

                '''

                print(tarinfo.name, "is", tarinfo.size, "bytes in size and is", end="")

                if tarinfo.isreg():

                    print("a regular file.")

                elif tarinfo.isdir():

                    print("a directory.")

                else:

                    print("something else.")

                '''

        tar.extractall(path='./temp/')

    station_files = sorted(os.listdir("./temp/"))

    df_st = pd.DataFrame(station_files)

    df_st2 = np.array(df_st[0].str.strip(".op.gz"))

    df_st3 = pd.DataFrame(df_st2,columns = ["Station"])

    t = len(df_st3)

    df_st3 = np.array(df_st3)

    year_list = [()]

    y_n = []

    y_df = pd.DataFrame([])

    

    for p in range(0,t):

        k = df_st3[p] == find_station+"-"+year

        if k == True:

            print("Found :",year)

    e_n = len(os.listdir("./temp/"))

    n = os.listdir("./temp")

    for i in range(0,e_n):

        os.remove("./temp/"+n[i])

        

def function(name = "Kayıt" , year = "2018" , station = "320690-99999",read = True):

    year_files = os.listdir("../input/gsod_all_years")

    year_files.sort()

    pathstem = "../input/gsod_all_years/"

    with tarfile.open(pathstem +"gsod_"+year+".tar") as tar:

        i = 0

        for tarinfo in tar:

                '''

                print(tarinfo.name, "is", tarinfo.size, "bytes in size and is", end="")

                if tarinfo.isreg():

                    print("a regular file.")

                elif tarinfo.isdir():

                    print("a directory.")

                else:

                    print("something else.")

                '''

        tar.extractall(path='./temp/'+name)

    station_files = sorted(os.listdir("./temp/"+name))

    with gzip.open("./temp/"+name+"/"+station+"-"+year+".op.gz",'rb') as station_file:

        i = 0

        for line in station_file:

                #print(line)

            i+=1

            if i > 4: break

        station_df = pd.read_csv(station_file, sep=r'\s+',index_col=None)

        columns = ["STN","WBAN","YEAR","TEMP","COUNT","DEWP","COUNT","SLP","COUNT",

                   "STP","COUNT","VISIB","COUNT","WDSP","COUNT","MXSPD","GUST",

                   "MAX","MIN","PRCP","SNDP","FRSHTT"]

    st = station_df.values

    df = pd.DataFrame(st,columns=columns)

    stn_df = df["STN"]

    wban_df = df["WBAN"]

    year_df = df["YEAR"]

    temp_df = df["TEMP"] - 32 / 1.8

    dewp_df = df["DEWP"]

    slp_df = df["SLP"]

    stp_df = df["STP"]

    vısıb_df = df["VISIB"]

    max_df = df["MAX"].str.strip("*")

    mın_df = df["MIN"].str.strip("*")

    

    max_df1 = (pd.DataFrame(max_df.values.astype(float),columns = ["MAX"]) - 32 / 1.8).round()

    mın_df1 = (pd.DataFrame(mın_df.values.astype(float),columns = ["MIN"]) - 32 / 1.8).round()

    df = pd.concat([stn_df,wban_df,year_df,temp_df,dewp_df,slp_df,stp_df,vısıb_df,max_df1,mın_df1],axis = 1)

    new_year = []

    n = len(df["YEAR"])

    for i in range(0,n): 

        a = str(df["YEAR"].values[i])

        a1 = a[0:4]

        a2 = a[4:6]

        a3 = a[6:8]

        new_year.append(a1+"."+a2+"."+a3)

    new_year = pd.DataFrame(new_year,columns = ["YEARS"])

    df = pd.concat([new_year,df],axis = 1)

    df = df.drop(["YEAR"],axis = 1)

    df['YEARS']=pd.to_datetime(df['YEARS'], format='%Y%m%d', errors='ignore')

    df['YEAR']=pd.DatetimeIndex(df['YEARS']).year

    df['MONTH']=pd.DatetimeIndex(df['YEARS']).month

    df['DAY']=pd.DatetimeIndex(df['YEARS']).day

    #df.set_index("MONTH",inplace = True)

    #nm = station_files[station]  

    df.to_csv(station+"-"+year+".csv",index = False)    

    #print(" Year Values : ",len(year_files),"\n Max Year Values : 90","\n Station Values : ",len(station_files))

    #print(os.listdir())

    sta = pd.DataFrame(station_files,columns = ["Station"])

    return df.head(2)



def visulation(x = 2008 , y = 2019 , st = "170200-99999"):

    for i in range(x,y):

        function(name = "Saving/"+st , year = str(i) , station=st)

    end_num = len(sorted(os.listdir()))

    file = sorted(os.listdir())



    plt.figure(figsize = (20,7))

    plt.title("Month")

    for i in range(1,end_num-2):

        df = pd.read_csv(file[i])

        sns.lineplot(df["MONTH"],df["TEMP"],label = file[i])

    plt.legend()

    

    plt.figure(figsize = (20,7))

    plt.title("Day")

    for i in range(1,end_num-2):

        df = pd.read_csv(file[i])

        sns.lineplot(df["DAY"],df["TEMP"],label = file[i])

    plt.legend()

    

    plt.figure(figsize = (20,7))

    plt.title("Distrubiton")

    for i in range(1,end_num-2):

        df = pd.read_csv(file[i])

        sns.distplot(df["TEMP"],hist = None,label = file[i])

    plt.legend()

    

def all_year():

    import glob

    end_num = len(sorted(os.listdir()))

    file = sorted(os.listdir())

    path =r'./' # use your path

    allFiles = glob.glob(path + "/*.csv")

    list_ = []

    for file_ in allFiles:

        df = pd.read_csv(file_,index_col=None, header=0)

        list_.append(df)

    frame = pd.concat(list_, axis = 0, ignore_index = True)

    plt.figure(figsize = (20,7))

    plt.title("Annual Mean Temperature")

    sns.lineplot(frame["YEAR"],frame["TEMP"],markers=".",color = "red",label = "Avg.Temp")

    plt.legend()

    

    for i in range(1,end_num-2):

        os.remove(file[i])