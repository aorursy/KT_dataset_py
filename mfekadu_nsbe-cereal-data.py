# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def make_dataframe(csv_filename):

    """

    given a CSV filename from the input folder

    output a pandas dataframe

    """

    return pd.read_csv(csv_filename)
CSV_FILENAME = "/kaggle/input/80-cereals/cereal.csv"

df = make_dataframe(CSV_FILENAME)
def get_cool_cereals():

    """

    return a dataframe with with just 10 cool cereals 

    """

    return pd.concat([df[df.name == "Cocoa Puffs"],

                    df[df.name == "Apple Jacks" ],

                    df[df.name == "Count Chocula"],

                    df[df.name == "Cheerios"],

                    df[df.name == "Froot Loops"],

                    df[df.name == "Frosted Flakes"],

                    df[df.name == "Lucky Charms"],

                    df[df.name == "Rice Krispies"],

                    df[df.name == "Honey Nut Cheerios"],

                    df[df.name == "Frosted Mini-Wheats"]])
cereals = get_cool_cereals()

cereals
import matplotlib as mpl

#this will be where the code goes for matplot lib



def makeScatterPlot(x, y):

    mpl.pyplot.plot(x, y)

def makeHistogram(x, bins, colors = "blue"):

    mpl.pyplot.hist(x, bins, color = colors)

def makeBar(x, y):

    mpl.pyplot.bar(x, y)

def makeBarh(x, y):

    mpl.pyplot.barh(x, y)

def makePie(x):

    mpl.pyplot.pie(x)

    
x = cereals["sugars"].sort_values()

y = cereals["rating"]

title = "Amount of Sugars Vs. Rating"



makeScatterPlot(x,y)
x = cereals["sugars"]

bins = 10

#colors = "red"

makeHistogram(x, bins)
x = cereals["name"]

y = cereals["sodium"]



makeBar(x, y)
makeBarh(x,y)
#this is where the code goes for Machine learning