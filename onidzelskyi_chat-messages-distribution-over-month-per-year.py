import pandas as pd

from matplotlib import pyplot as plt





%matplotlib inline
# Read dataset

df = pd.read_csv('../input/messages.csv')
# Convert date column to datetime object

# and show first 5 rows of dataset

df.date = pd.to_datetime(df.date)

df.set_index('date', inplace=True)

df.head()
# Print out some dataset' statistics, such as

# total count of messages in dataset

# count of unique messages in dataset

# and count of non-unique (occuried more than once) messages in dataset.

print('''

    #messages in dataset:              {}

    #unique messages in dataset:       {}

    #messages occuried multiple times: {}'''.format(len(df), len(df.msg.unique()), sum(df.msg.value_counts()>1)))
# We see that almost half of messages in dataset are duplicates.



# Let's show up how are #messages distributed in dataset

plt.figure(figsize=(15, 5))

plt.hist(df.msg.value_counts(), bins=50)

plt.yscale('log', nonposy='clip')

plt.title('Log-Histogram of messages appearance counts')

plt.xlabel('Number of occurences of messages')

plt.ylabel('Number of messages')

print()
# From plot above we can make desicion how many duplicates are in the dataset.

# 

# Show up how are messages distributed by number of characters in message

plt.figure(figsize=(15, 10))

plt.hist(df.msg.apply(len), bins=200, range=[0, 200], normed=True, label='msg')

plt.title('Normalised histogram of character count in messages', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)

print()
# From plot above we can see that roughly speaking, the message contains from 30 to 80 characters.

# 

# Next, show up how are messages distributed by number of words

plt.figure(figsize=(15, 10))

plt.hist(df.msg.apply(lambda x: len(x.split(' '))), bins=50, range=[0, 50], normed=True, label='msg')

plt.title('Normalised histogram of word count in messages', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)

print()
# Roughly speaking, the message contains 10 words.

# 

# Finally, let's show up word' distribution in dataset

from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate(" ".join(df.msg))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
# At the end, let's show up how are messages distributed by month and year

# Select data for 2013, 2014, 2015, and 2016 years

ds = df['2012-01-01': '2016-12-31']



# Plot histogram of message's count per month per year

ds.groupby([ds.index.year, ds.index.month]).count().unstack(level=0).plot(kind='bar', subplots=False)

plt.legend(loc='upper right', fancybox=True, fontsize=8)

plt.xlabel('months')

plt.ylabel('#messages')

plt.tight_layout()
# From plot above we see, that most 'chatted' year was 2013, and after it overall number of messages was gone out.