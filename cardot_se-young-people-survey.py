import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/responses.csv")

df_columns = pd.read_csv("../input/columns.csv")

df
df_columns
df.Gender.value_counts()
print("Suurim vanus on", df["Age"].max())

print("Väikseim vanus on", df["Age"].min())
df.Age.value_counts()
df.Education.value_counts() #haridustase
df.Music.plot.hist(grid=True) #I enjoy listening to music 1-5
df.plot.scatter("Happiness in life", "Age") #Võrdleb rahulolu eluga ja vanust
df.groupby(["Age"])["Entertainment spending"].mean().sort_values(ascending = False)

#spending = "I spend a lot of money on partying and socializing"