# Read in data from pandas
import pandas as pd

# This is used for fast string concatination
from io import StringIO

# Use nltk for valid words
import nltk
import collections as co


import warnings # ignore warnings 
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Read the input
d = pd.read_csv("../input/loan.csv") 
# Density Distribution of Loan Amount
sns.set_style("whitegrid")
ax=sns.distplot(d.loan_amnt)
ax.set(xlabel='Loan Amount', 
       ylabel='% Distribution',title='Density Plot of Loan Amount')

plt.legend();
# For fun, we'll make it green
ax = sns.violinplot(d.loan_amnt,color="g");
ax.set(xlabel='Loan Amount', 
       ylabel='Distribution',title='Violin Plot of Loan Amount')

plt.legend();
# Density Distribution of Interest Rate
sns.set_style("whitegrid")
ax=sns.distplot(d.int_rate, color="r")
ax.set(xlabel='Interest Rate %', 
       ylabel='% Distribution',title='Density Plot of Interest Rate')

plt.legend();
# We want a very fast way to concat strings.
# Final value will be stored in s
si=StringIO()
d['title'].apply(lambda x: si.write(str(x)))
s=si.getvalue()
si.close()
# Note sure how meaningful this is
# but here's a look.
s[0:400]
from wordcloud import WordCloud

# Read the whole text.
text = s

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt


# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(background_color="white",max_words=len(s),max_font_size=40, relative_scaling=.5).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()