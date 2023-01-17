# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from pandas import Series,DataFrame
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
dframe = pd.read_csv("../input/appendix.csv")
dframe.head()
ax = dframe.groupby("Institution").size().to_frame().plot(kind='bar',figsize=(10,10))

ax.set_ylabel("Count")

ax.set_xlabel("Institution")

ax.set_title("Count of program by institution")
dframe["year"]=dframe["Launch Date"].str.split('/').str[-1]
Unique_Courses = set(dframe["Course Number"])

len(Unique_Courses)
len(dframe["Course Number"].unique())
ax = dframe.groupby(["year","Institution"]).size().to_frame().unstack()[0].plot(kind="bar",figsize=(10,10))

ax.set_xlabel("Year")

ax.set_ylabel("Count")

ax.set_title("Course count by institution by year")
participants = dframe.groupby(["Course Subject","Institution"])["Participants (Course Content Accessed)"].sum().to_frame()

participants
participants.unstack().plot(kind="bar",figsize=(10,10))
dframe["No Male"]=(dframe["% Male"]/100)*(dframe["Participants (Course Content Accessed)"])

dframe["No Female"]=(dframe["% Female"]/100)*(dframe["Participants (Course Content Accessed)"])
ax = dframe.groupby("Course Subject")[["No Male","No Female"]].sum().plot(kind="bar",figsize=(10,10))

ax.set_xlabel("Course Subject")

ax.set_ylabel("Count of Attendees")

ax.set_title("Course Subject Attendance by Sex")
dframe.head()
dframe["Instructors"].str.split(",")