# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns
file_path="../input/prestigious-awards-in-india-from-1954-to-2013/Bharat Ratna and Padma Awards.csv"

data=pd.read_csv(file_path,index_col="YEAR")
#Replacing the values

data.replace(to_replace ="Awards Not Announced", value ="0",inplace=True)
data.info()
#Converting the object datatype to integer datatype

data["BHARAT RATNA"]=data["BHARAT RATNA"].astype(int)

data["PADMA VIBHUSHAN"]=data["PADMA VIBHUSHAN"].astype(int)

data["PADMA BHUSHAN"]=data["PADMA BHUSHAN"].astype(int)

data["PADMA SHRI"]=data["PADMA SHRI"].astype(int)

data["TOTAL"]=data["TOTAL"].astype(int)
data.info()
plt.figure(figsize=(18,10))

plt.title("National Honours from 1954-2013")

sns.lineplot(data=data)
#counting total number of awardees for each award

x1=data['PADMA SHRI'].sum()

x2=data['PADMA BHUSHAN'].sum()

x3=data['PADMA VIBHUSHAN'].sum()

x4=data['BHARAT RATNA'].sum()
count={"PADMA SHRI":[x1],"PADMA BHUSHAN":[x2],"PADMA VIBHUSHAN":[x3],"BHARAT RATNA":[x4]}
count=pd.DataFrame(count)
# Set the width and height of the figure

plt.figure(figsize=(20,6))



# Add title

plt.title("Comparison of each category")



sns.barplot(data=count)



# Add label for vertical axis

plt.ylabel("Number of people awarded")