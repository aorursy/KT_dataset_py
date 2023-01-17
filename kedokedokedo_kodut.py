import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline

pd.set_option('display.max_rows', 20)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/vgsaleeeeee/vgsales.csv")

df
df2 = pd.DataFrame({"Aasta" : df["Year"],

             "Globaalne_läbimüük" : df["Global_Sales"]})

df2
fig,ax = plt.subplots(figsize=(10,3))

df["Platform"].value_counts(sort=False).plot(kind="bar",ax=ax,rot=90)

plt.xlabel("Platform")

plt.ylabel("Number of sales")
df.plot.scatter("Global_Sales","EU_Sales",alpha = 0.1)

df.plot.scatter("Global_Sales","NA_Sales",alpha = 0.1)

df.plot.scatter("Global_Sales","JP_Sales",alpha = 0.1)
