import pandas as pd # For data processing, csv input

import matplotlib.pyplot as plt # For making the static visualizations

import seaborn as sns # For the attractive and informative statistical graphs

import os # For using system dependent functionality 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Readind the .csv file

df = pd.read_csv("/kaggle/input/most-popular-programming-languages-since-2004/Most Popular Programming Languages from 2004 to 2020.csv")



# Converting to the standard datatime format

df["Date"] = pd.to_datetime(df["Date"])

df.set_index("Date", inplace = True)



# To explore the format

df.head()
plt.figure(figsize = (20,10))



# making a heatmap for analysing the distribution using the mako colourmap

hmap_one = sns.heatmap(df, cmap="mako", yticklabels = False)
# Filtering out the top 10 languages

df = df.loc[:, (df.mean() > 2.5)]

plt.figure(figsize = (20,10))



# making a heatmap using the mako colourmap

hmap_two = sns.heatmap(df, cmap="mako", yticklabels = False)
# Creating a list for the languages 

# inspired by aditya2803's work

columns = df.columns.tolist()
plt.figure(figsize = (30,10))

# Plotting a line for each language on the plot

for lang in columns:

    ax = sns.lineplot(x = df.index, y = df[lang], label = lang, linewidth = 4)



# setting background colour to white

ax.set_facecolor("white")

# Plotting basic axes and title

plt.title("Line Chart")

plt.ylabel("Popularity in %")

plt.xlabel("Time in date-time format")

#. Customizing the legends 

legend = plt.legend(loc = 9, ncol = 10, fancybox = True, facecolor = 'black', edgecolor = 'blue', fontsize = 14)

plt.setp(legend.get_texts(), color='w')

plt.yticks()

plt.show()