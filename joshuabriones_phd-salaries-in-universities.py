import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 
# Import csv file to variable "data"

data = pd.read_csv("../input/phd-stipends/csv")

data.head(3)
# I decided to drop Comments column because didn't find it useful.



data.drop("Comments", axis=1, inplace=True)

data.head()
# Use info method to evaluate dtype and non_null entries

data.info()
# First, we drop all rows without University name



data = data.loc[data["University"].notnull() == True]

data.reset_index(drop=True, inplace=True)



# Repeat the same for rows without Overall Pay



data = data.loc[data["Overall Pay"].notnull() == True]

data.reset_index(drop=True, inplace=True)



data.info()
#This is a bad row I found while working in the notebook.



data.loc[data["University"] == "Scumbag college"]
data = data.loc[data["University"] != "Scumbag college"]
# Now we define a function to change entries with a numerical value to floats and add a Total column



def str_to_float(x):

    try:

        n = x.replace("$","").replace(",","")

        return float(n)

    except: return 0



data["Overall Pay"] = data["Overall Pay"].apply(str_to_float)

data["12 M Gross Pay"] = data["12 M Gross Pay"].apply(str_to_float)

data["9 M Gross Pay"] = data["9 M Gross Pay"].apply(str_to_float)

data["3 M Gross Pay"] = data["3 M Gross Pay"].apply(str_to_float)

data["Fees"] = data["Fees"].apply(str_to_float)



data["Total"] = data["3 M Gross Pay"]+data["9 M Gross Pay"]+data["12 M Gross Pay"]



data.head(5)
# Finaly we make another two columns to divide University name and it's short version



def u_name(x):

    if x == "ETH": return x

    else:

        name = x.split("(")[0].strip()

        return name.title()



def u_short(x):

    try:

        return x.split("(")[1].strip(")")

    except:

        return None



data["Name"] = data["University"].apply(u_name)

data["Short Name"] = data["University"].apply(u_short)

data.head()
top_10 = data.groupby("Name")["Total"].describe().sort_values(by="mean", ascending=False).iloc[:10]



import textwrap

f = lambda x: textwrap.fill(x.get_text(), 13)



plt.figure(figsize=(15,8))

ax = sns.barplot(x=top_10.index, y="mean", data=top_10)



ax.set_title("Top 10 Highest Paying Universities", fontsize = 28)

ax.set_ylabel("Overall Pay in $\$$ (Mean)", fontsize = 15)

ax.set_xlabel("University Name", fontsize = 15)

ax.set_xticklabels(ax.set_xticklabels(map(f, ax.get_xticklabels())), rotation=0)



plt.show()

low_13 = data.groupby("Name")["Total"].describe().sort_values(by="mean", ascending=False).iloc[-13:]



plt.figure(figsize=(15,8))

ax = sns.barplot(x=low_13.index, y="mean", data=low_13)



ax.set_title("Top 10 Lowest Paying Universities", fontsize = 28)

ax.set_ylabel("Overall Pay in $\$$ (Mean)", fontsize = 15)

ax.set_xlabel("University Name", fontsize = 15)

ax.set_xticklabels(ax.set_xticklabels(map(f, ax.get_xticklabels())), rotation=0)



plt.show()
