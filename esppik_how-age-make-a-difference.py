import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/CAERS_ASCII_2004_2017Q2.csv")
data = data.dropna()

#%%

indus_Name = data["PRI_FDA Industry Name"]

age = data["CI_Age at Adverse Event"]

#%%

#top 10 indus

indus = list(indus_Name.value_counts().head(5).index)

data = data[data["PRI_FDA Industry Name"].isin(indus)]

#remove wrong age

data = data[data["CI_Age at Adverse Event"] < 150]

age = data["CI_Age at Adverse Event"]

#cut ages in bins

bins = bins = np.arange(0, 140, 10)

age_cut = pd.cut(age, bins)
##form the plot data

plot_ser = data["PRI_FDA Industry Name"].groupby([age_cut,data["PRI_FDA Industry Name"]]).agg("count")

df = plot_ser.unstack()
#define plot func

def bp(df):

    plt.style.use('ggplot')

    fig, ax = plt.subplots(1,1)

    for col in range(df.shape[1]):

        

        if col >= 1:

            ax.bar(np.arange(12),df.iloc[:, col-1], label = df.columns[col])

        else:

            ax.bar(np.arange(12),df.iloc[:,col], label = df.columns[col])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_xticks(np.arange(12))

    ax.set_xticklabels(df.index,rotation=45)

    ax.set_title("Top food accident")

    ax.set_xlabel("age group")

    ax.set_ylabel("count")

    plt.show()
bp(df)