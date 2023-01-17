import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from colorama import Fore, Back, Style
df = pd.read_csv("../input/haberman.csv",header = None)
df.columns = ['age','year','#auxNodes','status']
df['status'] = df['status'].apply(lambda y: 'survived' if y == 1 else 'died')
print("*"*15 +"Basic Information Of DataSet","*"*15)
print(df.info()) # #attributes, #entries
print("*"*15 +"Description Of DataSet","*"*15)
print(df.describe()) #descritption of the dataset such as count,mean,std,min,max etc.
print("No. of people survived",df["age"][df["age"].between(0,100)][df["status"] == 'survived'].count()) #no. of people who survived more than 5 years
print(df.iloc[:,-1].value_counts(normalize = True))
sns.set()
style.use("ggplot")
sns.set_style("whitegrid")

#univariate analysis
#histograms
for idx,features in enumerate(df.columns[:-1]):
    sns.FacetGrid(df,hue = 'status',size = 5).map(sns.distplot,features).add_legend()
    plt.title("Histogram Plot using "+ features.upper())
    plt.show()




#Probability Density Function and Cummulative Density Functioon
plt.figure(figsize = (20,8))
for idx,features in enumerate(df.columns[:-1]):
    plt.subplot(1,3,idx+1)
    #Survived People Probability Distribution 
    counts_survived,bins_edges_survived = np.histogram(df[features][df['status'] == 'survived'],bins = 10, density = True)
    pdf_survived = counts_survived/sum(counts_survived)
    cdf_survived = np.cumsum(pdf_survived)
    #Died People Probability Distribution 
    counts_died, bins_edges_died = np.histogram(df[features][df['status'] == 'died'], bins = 10, density = True)
    pdf_died = counts_died/sum(counts_died)
    cdf_died = np.cumsum(pdf_died)
    
    print(Fore.GREEN + "*"*20 + Style.RESET_ALL + "  "+features.upper()+"  "+Fore.GREEN + "*"*20 + Style.RESET_ALL)
    print (Fore.RED +"Probability Density of People Survived"+Style.RESET_ALL, pdf_survived)
    print (Fore.RED + 'Probability Density of People Died '+ Style.RESET_ALL , pdf_died)
    print (Fore.RED + 'Cummulative Density of People Survived  '+ Style.RESET_ALL, cdf_survived)
    print (Fore.RED + 'Cummulative Density of People Died  '+ Style.RESET_ALL, cdf_died)
    
    #Graph Plotting.
    plt.title("PDF and CDF of "+features)
    plt.plot(bins_edges_survived[1:],pdf_survived, color = 'black',label = 'pdf of survived patient')
    plt.plot(bins_edges_survived[1:],cdf_survived, color = 'blue',label = 'cdf of survived patient')
    plt.plot(bins_edges_died[1:],pdf_died, color = 'red', label = 'pdf of dead patient')
    plt.plot(bins_edges_died[1:],cdf_died, color = 'green', label = 'cdf of dead patient')
    plt.xlabel(features)
    plt.legend()
plt.show()

#Box plot with wishkers
fig,axes = plt.subplots(1,3,figsize = (15,5))
for idx, features in enumerate(df.columns[:-1]):
    sns.boxplot(x = 'status', y = features, data = df, ax = axes[idx]).set_title("Box plot with "+features)
plt.show()

#Violin Plot
fig,axes = plt.subplots(1,3,figsize = (15,5))
for idx, features in enumerate(df.columns[:-1]):
    sns.violinplot(x = 'status', y = features, data = df, ax = axes[idx]).set_title("violin plot using " +features)

plt.show()


#Bivariate Analysis

#Pair Plot
sns.pairplot(df,hue = "status", size = 3)
plt.show()



'''#Graph representing the density estimate (Contour Graph)
sns.jointplot(x = "age", y = "year", data = df, kind = 'kde')
plt.show()
sns.jointplot(x = "age", y = "#auxNodes", data = df, kind = 'kde')
plt.show()
sns.jointplot(x= "year", y = "#auxNodes", data = df, kind = 'kde')
plt.show()'''


#Graph with scatter plot then applying density estimate (contour graph) 
#Seperate Contour for died and survived to better visulize both the status.
g = (sns.jointplot("age", "year",data=df[df['status'] == 'survived'], color="black").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
g = (sns.jointplot("age", "year",data=df[df['status'] == 'died'], color="red").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
plt.show()

g = (sns.jointplot("age", "#auxNodes", data = df[df['status']== 'survived'], color = "black").plot_joint (sns.kdeplot, zorder = 0, n_levels = 6))
g = (sns.jointplot("age", "#auxNodes", data = df[df['status']== 'died'], color = "red").plot_joint (sns.kdeplot, zorder = 0, n_levels = 6))
plt.show()

g = (sns.jointplot("year", "#auxNodes", data = df[df['status']== 'survived'], color = "black").plot_joint (sns.kdeplot, zorder = 0, n_levels = 6))
g = (sns.jointplot("year", "#auxNodes", data = df[df['status']== 'died'], color = "red").plot_joint (sns.kdeplot, zorder = 0, n_levels = 6))
plt.show()
