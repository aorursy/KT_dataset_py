import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

from matplotlib.colors import LogNorm

import warnings

warnings.simplefilter("ignore")

import os
print(os.listdir(".."))
df = pd.read_csv("../input/malicious-and-benign-websites/dataset.csv")
df.head().T
def plot_freq(df, column, n = None):  

    counts = df[column].value_counts()

    Indices = counts.index

    if n == None:

        n = len(Indices)

   

    plt.figure(figsize=(20,10)) 

    plt.title("Value Counts (Overall) for " + column) 

    plt.bar(range(n), counts[:n], color="g", align="center") 

    plt.xticks(range(n), Indices[:n], fontsize=11, rotation=90) 

    plt.xlim([-1, n]) 

    plt.show() 
plot_freq(df, "SERVER", n = 30)   
plot_freq(df, "WHOIS_COUNTRY", n = 30)   
plot_freq(df, "CHARSET")   
plot_freq(df, "NUMBER_SPECIAL_CHARACTERS", n = 30)   
def plot_freq_malicious(df, column, n = None):  

#     counts = df[column].value_counts()

    counts = df.loc[df["Type"] == 1, ][column].value_counts() ## df.Type == 1 ==> Malicious websites

    Indices = counts.index

    if n == None:

        n = len(Indices)

   

    plt.figure(figsize=(20,10)) 

    plt.title("Value Counts (Malicious) for " + column) 

    plt.bar(range(n), counts[:n], color="g", align="center") 

    plt.xticks(range(n), Indices[:n], fontsize=11, rotation=90) 

    plt.xlim([-1, n]) 

    plt.show()

plot_freq_malicious(df, "SERVER", n = None) 
plot_freq_malicious(df, "WHOIS_COUNTRY", n = None) ## Most malicious websites are from "Spain"("ES") or they did not declare
plot_freq_malicious(df, "WHOIS_STATEPRO", n = None) ## Most malicious websites are from Barcelona?
plot_freq_malicious(df, "CHARSET", n = None) 
plot_freq_malicious(df, "NUMBER_SPECIAL_CHARACTERS", n = None) 
plot_freq_malicious(df, "REMOTE_IPS", n = None) 
plot_freq_malicious(df, "URL_LENGTH", n = None) ## A pattern: URL length for malicious websites is 100?