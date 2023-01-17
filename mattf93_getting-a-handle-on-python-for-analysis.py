# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/marathon_results_2016.csv")

df.head()
def time_to_mins(x):

    if x is not '-':

        time = x.split(":")

        hours = int(time[0])

        mins = int(time[1])

        secs = int(time[2])

        total_mins = hours*60 + mins + secs/60

        return total_mins

    
df["Total Time (mins)"] = df["Official Time"].apply(lambda x: time_to_mins(x))

total = df["Total Time (mins)"]

sns.distplot(total[df["M/F"] == "M"])

sns.distplot(total[df["M/F"] == "F"])

plt.legend(["Males","Females"])
distance = 5 #in kilometres

df["sp_5"] = (distance*60)/df["5K"].apply(lambda x: time_to_mins(x))

df["sp_10"] = (distance*60)/(df["10K"].apply(lambda x: time_to_mins(x))-df["5K"].apply(lambda x: time_to_mins(x)))

df["sp_15"] = (distance*60)/(df["15K"].apply(lambda x: time_to_mins(x))-df["10K"].apply(lambda x: time_to_mins(x)))

df["sp_20"] = (distance*60)/(df["20K"].apply(lambda x: time_to_mins(x))-df["15K"].apply(lambda x: time_to_mins(x)))

df["sp_25"] = (distance*60)/(df["25K"].apply(lambda x: time_to_mins(x))-df["20K"].apply(lambda x: time_to_mins(x)))

df["sp_30"] = (distance*60)/(df["30K"].apply(lambda x: time_to_mins(x))-df["25K"].apply(lambda x: time_to_mins(x)))

df["sp_35"] = (distance*60)/(df["35K"].apply(lambda x: time_to_mins(x))-df["30K"].apply(lambda x: time_to_mins(x)))

df["sp_40"] = (distance*60)/(df["40K"].apply(lambda x: time_to_mins(x))-df["35K"].apply(lambda x: time_to_mins(x)))



#the mean() object doesn't include NaN's by default but they need to be dropped when plotting

sns.distplot(df["sp_5"].dropna())

#sns.boxplot(y=df["sp_5"])

#sns.boxplot(df["sp_40"])
#avgs contains the average speeds between the markers

avgs = [df["sp_5"].mean(), df["sp_10"].mean(), df["sp_15"].mean(), df["sp_20"].mean(),

        df["sp_25"].mean(), df["sp_30"].mean(), df["sp_35"].mean(),df["sp_40"].mean()]



avgs_M = [df["sp_5"][df["M/F"] == "M"].mean(), df["sp_10"][df["M/F"] == "M"].mean(), 

          df["sp_15"][df["M/F"] == "M"].mean(), df["sp_20"][df["M/F"] == "M"].mean(),

        df["sp_25"][df["M/F"] == "M"].mean(), df["sp_30"][df["M/F"] == "M"].mean(), 

          df["sp_35"][df["M/F"] == "M"].mean(),df["sp_40"][df["M/F"] == "M"].mean()]



avgs_F = [df["sp_5"][df["M/F"] == "F"].mean(), df["sp_10"][df["M/F"] == "F"].mean(), 

          df["sp_15"][df["M/F"] == "F"].mean(), df["sp_20"][df["M/F"] == "F"].mean(),

        df["sp_25"][df["M/F"] == "F"].mean(), df["sp_30"][df["M/F"] == "F"].mean(), 

          df["sp_35"][df["M/F"] == "F"].mean(),df["sp_40"][df["M/F"] == "F"].mean()]



distances = [5,10,15,20,25,30,35,40]

plt.plot(distances,avgs)

plt.plot(distances,avgs_M)

plt.plot(distances,avgs_F)

plt.xlabel("Distance (km)")

plt.ylabel("Speed (km/h)")

plt.legend(["All","Male","Female"])