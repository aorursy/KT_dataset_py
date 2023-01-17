import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', header=None)

print(df.head())

df_other = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv' , header=None)

print(df_other.head())

df = df.iloc[2:]

print(df.head())

df_other = df_other.iloc[2:]

print(df_other.head())
import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud
sns.set(style="darkgrid")



ax = sns.countplot(y=df[1], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

ax.set_title("Q1: What is your age (# years)?")

plt.show()
ax = sns.countplot(y=df[2], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

ax.set_title("Q2: What is your gender? - Selected Choice")



plt.show()
q2_other_texts =df_other[20].dropna()

q2_others_list = q2_other_texts.tolist()

q2_others_list.pop(0) #remove the text corresponding to first row which states 'What is your gender? - Prefer to self-describe - Text'

q2_others_string = ' '.join([str(elem) for elem in q2_others_list]) #list to string conversion as word cloud api requires data in string



# Create the wordcloud object

wordcloud = WordCloud(width=960, height=960, margin=0).generate(q2_others_string)

plt.figure( figsize=(20,10) )



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()
ax = sns.countplot(x=df[4], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

ax.set_xticklabels(

    ax.get_xticklabels(), 

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='10'



)

ax.set_title("Q3: In which country do you currently reside?")



plt.show()
country_count =df[4].value_counts()

country_count.drop(country_count.tail(1).index,inplace=True) # remove the count for 'In which country do you currently reside?'

print(country_count)

print("Total No. of Countries: ", len(country_count))
ax = sns.countplot(y=df[5], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

ax.set_title("Q4:  What is the highest level of formal education that you have attained or plan to attain within the next 2 years?")



plt.show()
ax = sns.countplot(y=df[6], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

ax.set_title("Q5: Select the title most similar to your current role (or most recent title if retired): - Selected Choice")



plt.show()
q5_other_texts =df_other[26].dropna()

q5_others_list = q5_other_texts.tolist()

q5_others_list.pop(0) #remove the text corresponding to first row which states 'Select the title most similar to your current role (or most recent title if retired): - Other - Text'

q5_others_string = ' '.join([str(elem) for elem in q5_others_list]) #list to string conversion as word cloud api requires data in string



# Create the wordcloud object

wordcloud = WordCloud(width=960, height=960, margin=0).generate(q5_others_string)

plt.figure( figsize=(20,10) )



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()
ax = sns.countplot(y=df[8], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

ax.set_title("Q6: What is the size of the company where you are employed?")



plt.show()
ax = sns.countplot(y=df[9], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

ax.set_title("Q7: Approximately how many individuals are responsible for data science workloads at your place of business?")



plt.show()
plt.figure(figsize=(10,5))



ax = sns.countplot(x=df[8], hue=df[9], data=df,   palette='Set1')



ax.set_xticklabels(

    ax.get_xticklabels(), 

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='x-large'



)

ax.set_title("Combining results for organizational size and individuals responsible for data science workload")



plt.show()
ax = sns.countplot(y=df[10], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                    edgecolor=sns.color_palette("dark", 3))

ax.set_title("Q8: Does your current employer incorporate machine learning methods into their business?")



plt.show()
plt.figure(figsize=(10,5))



ax = sns.countplot(y=df[10], hue=df[8], data=df,   palette='Set1')



ax.set_xticklabels(

    ax.get_xticklabels(), 

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='x-large'



)

ax.set_title("Combining results for organizational size and use of ML methods in the organization")



plt.show()
ax = sns.countplot(x=df[20], data=df,

                   facecolor=(0, 0, 0, 0),

                    linewidth=5,

                   palette='Set1',

                    edgecolor=sns.color_palette("dark", 3))

ax.set_xticklabels(

    ax.get_xticklabels(), 

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='10'



)

ax.set_title("Q10: What is your current yearly compensation (approximate $USD)?")



plt.show()
plt.figure(figsize=(10,5))



ax = sns.countplot(x=df[20], hue=df[8], data=df,   palette='Set1')



ax.set_xticklabels(

    ax.get_xticklabels(), 

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='10'



)

ax.set_title("Combining results for organizational size and yearly compensation")



plt.show()