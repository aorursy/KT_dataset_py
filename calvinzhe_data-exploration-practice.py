# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read the Dataset

df = pd.read_csv("../input/dataset.csv")
#Extract Column Values

labels = df["Letter"]

orbPeriod = df["Orbital Period (days)"].values

orbPeriodUpperUnc = df["Orbital Period Upper Unc. (days)"].values

orbPeriodLowerUnc = df["Orbital Period Lower Unc. (days)"].values



#Create error array

error = np.absolute([-orbPeriodLowerUnc, orbPeriodUpperUnc])



#Set up x-axis locations

xPoints = np.arange(0,len(labels))



#Print to make sure values look right

print(xPoints)

print(orbPeriod)

print(error)
#Plot values in a bar plot

fig = plt.bar(xPoints, orbPeriod, yerr = error)

plt.xticks(xPoints, labels)

plt.xlabel("Label")

plt.ylabel("Orbital Period (Days)")

plt.title("Trappist-1 Planet Orbital Periods")



def autolabel(rects):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        height = rect.get_height()

        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')

        

autolabel(fig)
df["Reference"][0]
df.columns