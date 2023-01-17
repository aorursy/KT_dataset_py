# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt  

import matplotlib

from wordcloud import WordCloud, STOPWORDS #wordcloud's generator and english STOPWORDS list

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



pd.options.mode.chained_assignment = None #Just disabling SettingWithCopyWarning

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/offenders.csv", encoding = "latin1")

races = df["Race"].unique()

print (races)
for i in range(0,len(df)):

	race = df["Race"].iloc[i]

	if race == "White ":

		 df["Race"].iloc[i] = "White"

	elif race == "Hispanic ":

		 df["Race"].iloc[i] = "Hispanic"

	else:

		pass

    

races = df["Race"].unique()

print (races)
sns.factorplot('Race',data=df,kind='count')

plt.title("Number of offenders by race")

plt.show()
white = df[df["Race"]=="White"]

hispanic = df[df["Race"]=="Hispanic"]

black = df[df["Race"]=="Black"]



#Wordcloud of white offender's last statements:

wordcloud = WordCloud(

                         stopwords=STOPWORDS,

                         background_color='white',

                         width=1200,

                         height=1000

                        ).generate(" ".join(white['Last Statement']))



plt.imshow(wordcloud)

plt.title("White offenders")

plt.axis('off')

plt.show()



#Wordcloud of hispanic offender's last statements:

wordcloud = WordCloud(

                         stopwords=STOPWORDS,

                         background_color='white',

                         width=1200,

                         height=1000

                        ).generate(" ".join(hispanic['Last Statement']))



plt.imshow(wordcloud)

plt.title("Hispanic offenders")

plt.axis('off')

plt.show()



#Wordcloud of black offender's last statements:

wordcloud = WordCloud(

                         stopwords=STOPWORDS,

                         background_color='white',

                         width=1200,

                         height=1000

                        ).generate(" ".join(black['Last Statement']))

plt.imshow(wordcloud)

plt.title("Black offenders")

plt.axis('off')

plt.show()
