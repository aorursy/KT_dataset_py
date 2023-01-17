# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing data set of movie review

with open ("/kaggle/input/sholay-movie-review/Sholay Movie_Sample Review_Data.txt") as fh:

    Reviews  = [line.strip() for line in fh if line.strip()]
print("Reviews:\n",Reviews[:10])
from textblob import TextBlob

print('{:80}:{:25}:{:25}'.format("Reviews","Polarity","Subjectivity"))
for review in Reviews:

    sentiment=TextBlob(review)

    print('{:80}: {:01.2f}: {:01.2f}'.format(review[:60],sentiment.polarity,sentiment.subjectivity))

    
#Categorize polarity into positive, neutral or negative

labels = ["Negative", "Neutral", "Positive"]

#Initialize count array

values =[0,0,0]



#Categorize each review

for review in Reviews:

    sentiment = TextBlob(review)

    

    #Custom formula to convert polarity 

    # 0 = (Negative) 1 = (Neutral) 2=(Positive)

    polarity = round(( sentiment.polarity + 1 ) * 3 ) % 3

    

    #add the summary array

    values[polarity] = values[polarity] + 1

    

print("Final summarized counts :", values)



import matplotlib.pyplot as plt

#Set colors by label

colors=["Green","Blue","Red"]



print("\n Pie Representation \n-------------------")

#Plot a pie chart

plt.pie(values, labels=labels, colors=colors, \

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()