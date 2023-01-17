!pip install crossrefapi
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import matplotlib.pyplot as plt



from datetime import date

from crossref.restful import Works



import json, time







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
cutoff=.05





j=pd.value_counts(df["journal"])

k=j[j>max(j)*cutoff]



jdf=pd.concat(

    [

        pd.Series(list(k.keys()),name="journal"),

        pd.Series(list(k),name="count")

    ],

    axis=1

)



fig=px.pie(jdf,names="journal",values="count",title="Top "+str(100*(1-cutoff))+"% of journals by quantity")

fig.update_layout(showlegend=False)

fig.show()
# 81 percent have abtracts

# they've been adding to the dataset

round(len(df["abstract"][df["abstract"].notna()])/len(df["abstract"]),2)
#lets get an idea of how many unique authors there are here



auth_dict={}





for auth in df["authors"].dropna():

    for name in auth.split(";"):

        n=name.strip()

        if n in auth_dict:

            auth_dict[n]+=1

        else:

            auth_dict[n]=1

        

authdf=pd.concat(

    [

        pd.Series(list(auth_dict.keys()),name="author"),

        pd.Series(list(auth_dict.values()),name="count")

    ],

    axis=1    

)
print(str(len(authdf))+" unique authors")

# probably slightly less due to differences in naming

# i.e. John Smith vs John A. Smith
authdf=authdf.sort_values(by="count",ascending=False)



cutoff=30



px.bar(authdf[:cutoff],x="author",y="count",title="Top "+str(cutoff)+" authors by publication coauthorship")
# CORD19 date regularization function

# Kevin O'Neill



# example usage:

# [RegularizeDate(d) for d in  df["publish_time"]]



def RegularizeDate(d, asString=False):

    

    months=["Jan","Feb","Mar","Apr","May","Jun",

            "Jul","Aug","Sep","Oct","Nov","Dec"]

    maxdays_month=[31,28,31,30,31,30,31,31,30,31,30,31]

    

    season={

        "Winter":1, # jan

        "Spring":4, # apr

        "Summer":7, # jul

        "Fall":10,  # oct

        "Autumn":10 # oct

    }

    

    dateout=None



    if type(d)!=str:

        return d

    dashc=d.count("-")

    if dashc==0:

        # split by space and get date

        dsplit=d.split()

        

        # form YYYY Mmm DD or form YYYY Mmm DD Season

        if len(dsplit)==3 or len(dsplit)==4:

            try:

                dateout=date(

                    int(dsplit[0]),

                    months.index(dsplit[1])+1,

                    int(dsplit[2])

                )

            except:

                # out-of-bounds date like feb 31

                dateout=date(

                    int(dsplit[0]),

                    months.index(dsplit[1])+1,

                    min(int(dsplit[2]),maxdays_month[months.index(dsplit[1])])

                )

            

        elif len(dsplit)==2:

            try:

                # form YYYY Mmm

                dateout=date(

                    int(dsplit[0]),

                    months.index(dsplit[1])+1,

                    1

                )

            except:

                # form YYYY Season

                dateout=date(

                    int(dsplit[0]),

                    season[dsplit[1]],

                    1

                )

        

        # form YYYY

        elif len(dsplit)==1:

            dateout=date(

                int(dsplit[0]),

                1,

                1

            )

        else:

            print(d)

        

    elif dashc==1:

        dsplit=d.split()

        if len(dsplit)==4:

            # form YYYY Mmm DD Mmm-Mmm

            # or YYYY Mmm DD Mmm

            # (in which case drop the final Mmm-Mmm)

            try:

                dateout=date(

                    int(dsplit[0]),

                    months.index(dsplit[1])+1,

                    int(dsplit[2])

                )

            except:

                # out-of-bounds date like feb 31

                dateout=date(

                    int(dsplit[0]),

                    months.index(dsplit[1])+1,

                    min(int(dsplit[2]),maxdays_month[months.index(dsplit[1])])

                )

        elif len(dsplit)==2:

            # form YYYY Mmm-Mmm

            month=dsplit[1].split("-")[0]

            dateout=date(

                int(dsplit[0]),

                months.index(month)+1,

                1

            )

        else:

            print(d)

        

    elif dashc==2:

        if d.count("[")==0:

            # form YYYY-MM-DD

            dsplit=[int(di) for di in d.split("-")]

            dateout=date(dsplit[0],dsplit[1],dsplit[2])

        else:

            d_=d.strip("[]").split(",")[0].strip("'")

            dsplit=[int(di) for di in d_.split("-")]

            dateout=date(dsplit[0],dsplit[1],dsplit[2])      

    elif dashc>2:

        d_=d.strip("[]").split(",")[0].strip("'")

        dsplit=[int(di) for di in d_.split("-")]

        dateout=date(dsplit[0],dsplit[1],dsplit[2])     

        # form "['YYYY-MM-DD', 'YYYY-MM']"

        # discard the second date, take the first

    

    if asString:

        return dateout.isoformat()

    else:

        return dateout
# confirm that the RegularizeDate function handles all non-null publishing dates



print(sum([RegularizeDate(d)!=None or type(d)!=str for d in df["publish_time"]]))

print(len(df["publish_time"]))
# doi lookup function

# date from doi



# this is a more complete version of RegularizeDate

# if you have an internet connection, you should probably use this instead



# usage:

# df["ISO_date"]=dates_from_doi(df)



print(pd.value_counts([type(x) for x in df["doi"]]))

# str indicates it has an actual doi

# float indicates a null value (technically a NaN float value)



def dates_from_doi(df):

    works = Works()

    datesl=[]

    for row in df.itertuples():

        if type(row.publish_time)==float:

            if type(row.doi)!=float:

                print("Index "+str(row.Index)+": lookup for doi: "+str(row.doi))

                dat=works.doi(row.doi)

                if dat!=None:

                    issued=dat["issued"]["date-parts"][0]

                    issued+=[1]*(3-len(issued)) # fill in with earliest date

                    datesl.append(date(issued[0],issued[1],issued[2]))

                else:

                    datesl.append(float('nan'))

            else:

                datesl.append(float('nan'))

        else:

            datesl.append(RegularizeDate(row.publish_time,False))



# if works.doi(doi) returns none, that doi has no entry in the doi database

# some (but not all) are preprints

# google will usually still find them (if you search manually);

# doi lookup will not
years=[RegularizeDate(d,True).split("-")[0] for d in df["publish_time"] if type(d)==str]

counts=pd.value_counts(years)

px.bar(x=list(counts.keys()),y=list(counts),title="Publications in dataset by year")



# spike in 2003-2004 is presumably the effect of the 2002 SARS outbreak

# https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome
cutoff= date(2003,1,1)



dates=[RegularizeDate(d,False) for d in df["publish_time"] if type(d)==str]

dates2=[d for d in dates if d>cutoff]
px.histogram(x=dates2,title="Publications in dataset by date, "+cutoff.isoformat()+" to present")
#set them all to the same year just to check

# (set to leap year (year 4) so date doesn't complain about feb 29)

counts=pd.value_counts([date(4,d.month,d.day) for d in dates])



print(str(round(100*max(counts)/sum(counts),2))+"% of all articles are published on Dec 31st")

print()

print(counts)
doi_lookup=False

# look up missing dates via doi (or not)



start = time.time()

if doi_lookup:

    df["ISO_date"]=dates_from_doi(df) # obvi this one is gonna take freakin forever   

else:

    df["ISO_date"]=[RegularizeDate(d,False) for d in df["publish_time"]] # run this one for "good enough" date regularization

    # does not look up missing dates via doi

    # 0.2 seconds elapsed

stop = time.time()



print(str(stop-start)+" seconds elapsed")