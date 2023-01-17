# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter #Counter class is used to count frequency of data 

import matplotlib.pyplot as plt #library to plot graphs



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
sales_data = pd.read_csv("../input/Video_Games_Sales_as_at_30_Nov_2016.csv")

sales_data.head(3)
#filtering data to get rows with "Rating" coulmn not null 

refined_data = sales_data[~sales_data["Rating"].isnull()]
#data grouped on "Rating"

grouped_data = refined_data.groupby("Rating")
figure_number = 0

for category in grouped_data.groups:

    #Counter class used to calculate frequency of Genres of games with different categories.

    count = Counter(grouped_data.get_group(category)["Genre"]) 

    print ("Frequency of genres of games rated with : ", category, " are ", dict(count), " \n\n")



    keys = []

    value = []

    for k, v in count.items():

        keys.append(k)

        value.append(v)



    plt.figure(figure_number)

    figure_number += 1

    

    #create pie chart

    patches, texts = plt.pie(value, explode=None, shadow=False, startangle=90)



    for ele in range(len(texts)):

        texts[ele].set_fontsize(10)



    value_arr = np.array(value)

    percent = 100.*value_arr/value_arr.sum()

    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(keys, percent)]

    sort_legend = True

    if sort_legend:

        patches, labels, dummy = zip(*sorted(zip(patches, labels, value_arr), key=lambda x: x[2],reverse=True))



    plt.legend(patches, labels, loc="lower right", title="Genre", prop={'size': 8})

    plt.axis('equal')

    plt.title("Games with rating : " + category)

plt.show()