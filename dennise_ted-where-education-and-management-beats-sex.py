import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
ted=pd.read_csv("../input/ted_main.csv")
transcripts=pd.read_csv("../input/transcripts.csv")
ted.info()
ted[ted["speaker_occupation"].isnull()]
ted.describe()
ted.head()
ted.corr()
ted.sort_values("views",ascending=False).head(5)
ted.sort_values("views",ascending=True).head(5)
ted.event.value_counts().head(10)
ted.event.value_counts().tail(10)
ted.speaker_occupation.value_counts().head(10)
import matplotlib.pyplot as plt
fig=plt.figure()

axes=fig.add_axes([0,0,1,1])

axes.set_xlabel("views")

axes.set_ylabel("comments")

axes.plot(ted["views"],ted["comments"],ls="",marker=".")
fig=plt.figure()

axes=fig.add_axes([0,0,1,1])

axes.set_xlabel("views")

axes.set_xlim(2200000)

axes.set_ylabel("comments")

axes.set_ylim(200)

axes.plot(ted["views"],ted["comments"],ls="",marker=".")
labels = ted.num_speaker.unique()

sizes = ted.num_speaker.value_counts()



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title("Number of presenters")



plt.show()
ted.num_speaker.unique()

ted.num_speaker.value_counts()
ted.columns
ted.ratings[0]
# ast.literal_eval evaluates the content of a string as python code - in a safe way (https://docs.python.org/3/library/ast.html#ast.literal_eval)

import ast

ted.ratings=ted.ratings.apply(lambda x:ast.literal_eval(x))

# Now ted.ratings is of type list
ted.ratings[0]
#Now I have a lists of dictionaries

ted.ratings[0][1]
ted.ratings[0][1]["name"]
interim=pd.DataFrame(ted.ratings[0])

interim
interim2=pd.DataFrame(ted.ratings.sum())

interim2.name.value_counts()
interim3=interim2.groupby("name").sum()["count"].sort_values(ascending=False)

interim3
Categories=["Inspiring","Informative","Fascinating","Persuasive","Beautiful","Courageous","Funny","Ingenious","Jaw-dropping","OK","Unconvincing","Longwinded","Obnoxious","Confusing"]

Categories
"""

This took me a very long time to somehow solve it. I am quite sure in a very, very ugly way.

If you happen to be able to point me in a direction how to do this much more elegant please do so!

"""

def normalized_count_of(i,x):

    # x is the full ratings dataframe

    # i is the Category name which is now the column to be filled

    for count in range(14):

        if pd.DataFrame(x)["name"][count]==Categories[i]:

            return pd.DataFrame(x).loc[count,"count"]



def total_count(x):

    total_votes=0

    for count in range(14):

        total_votes+=pd.DataFrame(x).loc[count,"count"]

    return total_votes

    

ted["Total_Votes"]=ted["ratings"].apply(lambda x:total_count(x))

        

for i in range(14):

    ted[Categories[i]]=ted["ratings"].apply(lambda x:normalized_count_of(i,x))/ted["Total_Votes"]

ted.head(2)
import seaborn as sns

sns.heatmap(ted.corr().iloc[6:33,6:33])
ted[ted["Inspiring"]==ted["Inspiring"].max()]
ted[ted["Funny"]==ted["Funny"].max()]