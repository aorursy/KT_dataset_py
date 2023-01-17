import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/dataisbeautiful/r_dataisbeautiful_posts.csv"

my_data = pd.read_csv(my_filepath, parse_dates=True)
my_data.head()

my_data_author = my_data.groupby('author').count().sort_values(by='id',ascending=False).drop('[deleted]').head()



plt.title('Articles per Author')

sns.barplot(x=my_data_author.index, y=my_data_author['id']).set_xticklabels(my_data_author.index, rotation=45)

plt.xlabel('Author')

plt.ylabel('Number of Articles')
#plot the number of comments per post

my_data_awards = my_data[my_data['num_comments']>1000].sort_values('num_comments',ascending=False)

plt.figure(figsize=(12,4))

sns.distplot(a=my_data_awards['num_comments'], kde=False)

plt.title('Articles with more than 1000 comments distribution')

plt.xlabel('number of comments')

plt.ylabel('number of articles')

plt.xlim(left=1000)
# plot the highest scoring articles

my_data_score = my_data.sort_values('score', ascending=False).head(12)

my_data_score.title=my_data_score.title.map(lambda i: i.replace('[OC]', ''))

plt.figure(figsize=(14,7))

plt.title('Highest Score articles')

sns.barplot(y=my_data_score['title'],x=my_data_score['score'])

plt.xlabel('score',fontsize=14)

plt.ylabel('Title',fontsize=12)