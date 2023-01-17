# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from textblob import TextBlob



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

           



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
for dirname, _, filenames in os.walk('/kaggle/input'):

    

    dff = {"Movie":[],

           "Total_Dialogues":[],

           "Positive_Dialogues":[],

           "Neutral_Dialogues":[],

           "Negative_Dialogues":[],

           "Positive_Sentiment":[],

           "Neutral_Sentiment":[],

           "Negative_Sentiment":[]}

    df = pd.DataFrame(dff)

    

    for filename in filenames:

        print(filename)

        

        fname = os.path.join(dirname, filename)

        with open(fname,'r',encoding="windows-1252") as f:

            raw_data = f.read()

            data = TextBlob(raw_data)

            pos = []

            neu = []

            neg = []

            for sentence in data.sentences:

                pol = sentence.sentiment.polarity

                if pol > 0.25:

                    pos.append(pol)

                elif pol > -0.25 and pol <= 0.25:

                    neu.append(pol)

                else:

                    neg.append(pol)

                

            total = len(data.sentences)

                

            pos_t = len(pos)

            neu_t = len(neu)

            neg_t = len(neg)

                

            pos_avg = sum(pos)/pos_t

            neu_avg = sum(neu)/neu_t

            neg_avg = sum(neg)/neg_t

                

            df = df.append({"Movie":filename[:-4],

                            "Total_Dialogues":total,

                            "Positive_Dialogues":pos_t,

                            "Neutral_Dialogues":neu_t,

                            "Negative_Dialogues":neg_t,

                            "Positive_Sentiment":pos_avg,

                            "Neutral_Sentiment":neu_avg,

                            "Negative_Sentiment":neg_avg}, ignore_index=True)
df.head()
import seaborn as sns

import matplotlib.pyplot as plt
colors = ["#006D2C","#74C476"]

ax = df.loc[:,['Positive_Dialogues','Negative_Dialogues']].plot.bar(stacked=True, color=colors,figsize=(14, 10))

ax.set_xticklabels(df.Movie,rotation=80)

ax.set_xlabel("Movies")

ax.set_ylabel("Dialogue Count")

plt.show()