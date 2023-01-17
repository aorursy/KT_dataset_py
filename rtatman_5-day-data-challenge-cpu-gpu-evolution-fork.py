# libraries

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# dataset

gpu_df = pd.read_csv("../input/All_GPUs.csv")

cpu_df = pd.read_csv("../input/Intel_CPUs.csv")
#Lets add columns and quarters columns

cpu_df.Launch_Date = cpu_df.Launch_Date.str.replace("\'", "-")

cpu_df[["Quarter", "Year"]] = cpu_df.Launch_Date.str.split("-", expand=True)

cpu_df.head()
# plotting library

import seaborn as sns



# plot lithography count by quarter 

sns.factorplot(x="Quarter", hue="Lithography",

               data=cpu_df, kind="count");



# looks like there's a little bit more data cleaning to be done!