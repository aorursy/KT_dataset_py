# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
df = pd.read_csv("../input/haberman.csv", header = None)
df.columns = ['age','year_of_opr','no_aux_nodes','survival_status']
text = "SIZE OF THE DATASET"
print("*"*25 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*25)
print(df.shape)#tells the dimensions of data set
text = "STATISTICAL DETAILS OF THE DATA"
print("*"*20 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*20)
print(df.describe())
text = "CONSISE SUMMARY OF DATAFRAME"
print("*"*20 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*20)
print(df.info())
text = "NUMBER OF PEOPLE SURVIVED IN THE RANGE OF AGE BETWEEN 30 TO 60"
print("*"*15 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*15)
print(df["age"][df["age"].between(30,60)][df["survival_status"] == 1].count()) # number of people survived in between age 30 to 60
text = "GROUPING DONE ON THE BASIS OF SURVIVAL STATUS SHOWING MAXIMUM AND MINIUM AGED PERSON SURVIVED AND DIED"
print("*"*15 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*15)
g = df.groupby("survival_status")
print(g.age.max())#tells maximum age of person died and survived
print(g.age.min())# tells minimum age of the person who survived and died
print(g.describe())#statistical info about each column
text = "TOTLA NUMBER OF PEOPLE SURVIVED"
print("*"*25 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*25)
print(df["survival_status"].value_counts()) #counts number of people survived as well as number of people not survived
#calculating percentage of people died and survived
text = "PERCENTAGE OF PEOPLE SURVIVED AND DIED"
print("*"*25 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*25)
df_1 = df.loc[df["survival_status"] == 1].count();
df_2 = df.loc[df["survival_status"] == 2].count();
x = df_1 + df_2
print(df_1/x *100)
print(df_2/x *100)

#print(df)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from colorama import Fore, Back, Style
df = pd.read_csv("../input/haberman.csv", header = None)
df.columns = ['age','year_of_opr','no_aux_nodes','survival_status']
x=list(df.columns[:-1])
index = 1

#Pair Plots
text = "PAIR PLOTS"
print("*"*25 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*25)
df["survival_status"] = df["survival_status"].apply(lambda y: "Survived" if y == 1 else "Died")
plt.close()
sns.set_style("whitegrid")
sns.pairplot(df, hue = "survival_status", size = 2)


#Histograms
sns.set_style("whitegrid")
for att in x:
    sns.FacetGrid(df, hue = "survival_status" , size = 4).map(sns.distplot, att).add_legend()
    plt.show()

#PDF's and CDF's   
df_1 = df.loc[df["survival_status"] == "Survived"];
df_2 = df.loc[df["survival_status"] == "Died"];
plt.figure(1)
sns.set_style("whitegrid")
for att in x:
    plt.subplot(1,3,index)
    index = index + 1
    counts, bin_edges = np.histogram(df_1[att], bins = 10, density = True)
    pdf = counts/ sum(counts)
    text = att + "_of_people_survived"
    print("*"*25 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*25)
    print(Fore.YELLOW + "Probability Density Function values are:" + Style.RESET_ALL,pdf)
    print(Fore.YELLOW + "Bin edges values are:" + Style.RESET_ALL,bin_edges)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(att + "_of_ppl_survived")
    counts, bin_edges = np.histogram(df_2[att], bins = 10, density = True)
    pdf = counts/ sum(counts)
    text = att + "_of_people_died"
    print("*"*25 + Fore.GREEN + Back.WHITE + text + Style.RESET_ALL + "*"*25)
    print(Fore.YELLOW + "Probability Density Function values are:" + Style.RESET_ALL,pdf)
    print(Fore.YELLOW + "Bin edges values are:" + Style.RESET_ALL,bin_edges)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(att + "_of_ppl_died")
#BoxPlots
index = 1
plt.figure(3)
sns.set_style("whitegrid")
for att in x:
    plt.subplot(1,3,index)
    index = index + 1
    sns.boxplot(x = "survival_status", y = att, data = df,)
plt.show()


#Violin plot
index = 1
plt.figure(3)
sns.set_style("whitegrid")
for att in x:
    plt.subplot(1,3,index)
    index = index + 1
    sns.violinplot(x = "survival_status", y = att, data = df)
plt.show()


# contour plot
sns.jointplot(x = "age", y = "year_of_opr", data = df, kind = 'kde')
plt.show()
    

