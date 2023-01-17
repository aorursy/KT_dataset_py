# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from wordcloud import WordCloud, STOPWORDS

import pandas as pd

import numpy as np

import  matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
round_one = pd.read_csv("../input/French_Presidential_Election_2017_First_Round.csv")

pd.set_option('display.max_columns', 500)
round_one.head()


ax=sns.factorplot(x="Voted",y="Surname",data=round_one,kind="bar")

plt.title("Vote share of each candidate over all first round")
# Let see the patteren of votes get by each candidate in first round

Votes_name=round_one.groupby("Surname")['Voted'].sum().reset_index().sort_values(by='Voted',ascending=False).reset_index(drop=True)

Votes_name
plt.figure(figsize=(10,10))

ax=sns.barplot(x="Surname",y="Voted",data=Votes_name)

plt.xticks(rotation=45)

plt.title("Each candidate mean vote")


wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=2200,

                          height=2000

                         ).generate(" ".join(round_one["Surname"]))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#total vote share of the candidate department wise

plt.figure(figsize=(40,40))

ax=sns.factorplot(x="Surname",y="Voted",hue="Department",data=round_one,kind="bar",size=12,aspect=3)



plt.title("Vote share of each candidate departement wise")
#No of votes given to  man or women

Votes_name_sex=round_one.groupby("Sex")['Voted'].sum().reset_index().sort_values(by='Voted',ascending=False).reset_index(drop=True)

Votes_name_sex
plt.figure(figsize=(10,10))

ax=sns.barplot(x="Sex",y="Voted",data=Votes_name_sex)



plt.title("mean vote for male and female")
second_round = pd.read_csv("../input/French_Presidential_Election_2017_Second_Round.csv")
second_round.head()
#Checking for the numerical data

second_round.describe()
# Let see the patteren of votes get by each candidate in second round

Votes_name_2=second_round.groupby("Surname")['Voted'].sum().reset_index().sort_values(by='Voted',ascending=False).reset_index(drop=True)

Votes_name_2
plt.figure(figsize=(10,10))

ax=sns.barplot(x="Surname",y="Voted",data=Votes_name_2)

plt.xticks(rotation=45)

plt.title("Each candidate mean vote in second round")
#total vote share of the candidate department wise for second round

ax=sns.factorplot(x="Surname",y="Voted",hue="Department",data=second_round,kind="bar",size=12,aspect=2)

plt.title("Vote share of each candidate departement wise")


ax=sns.factorplot(x="Voted",y="Surname",data=second_round,kind="bar")

plt.title("Vote share of each candidate over all second round")
wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=2200,

                          height=2000

                         ).generate(" ".join(second_round["Surname"]))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()