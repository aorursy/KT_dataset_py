import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from wordcloud import WordCloud, STOPWORDS

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')

users['RegisterDate'] = pd.to_datetime(users['RegisterDate'])

users['Year'] = users['RegisterDate'].dt.year



kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')

kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'])

kernels['Year'] = kernels['CreationDate'].dt.year
messages = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv')



messages['Thank'] = messages.Message.str.lower().str.contains('thank|great work|good work').fillna(0).astype(int)

messages['Upvote'] = messages.Message.str.lower().str.contains('upvote').fillna(0).astype(int)

messages['Promotion'] = messages.Message.str.lower().str.contains('check my notebook|check my kernel|my other notebook|my other kernel|check out my notebook|check out my kernel').fillna(0).astype(int)

messages['Question'] = messages.Message.str.lower().str.contains('why|how|what|"\?"').fillna(0).astype(int)

messages['Medal'] = messages.Medal.fillna(0)

messages['PostDate'] = pd.to_datetime(messages['PostDate'])

messages['Year'] = messages.PostDate.dt.year

messages['Week'] = messages.PostDate.dt.week



messages.head()
fig = plt.figure(figsize=(12, 24), facecolor='#f7f7f7') 

fig.subplots_adjust(top=0.95)

fig.suptitle('Evolution of Comments on Kaggle', fontsize=18)



gs = GridSpec(5, 2, figure=fig)



ax0 = fig.add_subplot(gs[0, 0])

ax1 = fig.add_subplot(gs[0, 1])

ax2 = fig.add_subplot(gs[1, :])

ax3 = fig.add_subplot(gs[2, :])

ax4 = fig.add_subplot(gs[3, :])

ax5 = fig.add_subplot(gs[4, :])



pd.Series(users.groupby('Year').size()).plot(style='.-', ax=ax0, color='black')

pd.Series(messages.groupby('Year').size()).plot(style='.-', ax=ax1, color='black')

messages.groupby(['Year'])[['Question']].mean().plot(style='.-', ax=ax2, color='green')

messages.groupby(['Year'])[['Thank']].mean().plot(style='.-', ax=ax3, color='firebrick')

messages.groupby(['Year'])[['Upvote']].mean().plot(style='.-', ax=ax4, color='darkorange')

messages.groupby(['Year'])[['Promotion']].mean().plot(style='.-', ax=ax5, color='darkblue')



ax0.set_title('Number of new users per year', fontsize=14)

ax1.set_title('Number of comments per year', fontsize=14)

ax2.set_title('Average number of comments with a question', fontsize=14)

ax3.set_title('Average number of comments with a thank you', fontsize=14)

ax4.set_title('Average number of comments with a request of upvote', fontsize=14)

ax5.set_title('Average number of comments promoting a notebook', fontsize=14)



for ax in [ax2, ax3, ax4, ax5]:

    ax.legend().set_visible(False)





plt.show()
comm_kernels = kernels[kernels.ForumTopicId.notna()].copy()



comm_kernels = pd.merge(comm_kernels, messages[['ForumTopicId', 'Medal', 'Question', 'Thank', 'Upvote', 'Promotion']], on='ForumTopicId')



comm_kernels.head()
fig = plt.figure(figsize=(12, 24), facecolor='#f7f7f7') 

fig.subplots_adjust(top=0.95)

fig.suptitle('Evolution of Comments on Kaggle Notebooks', fontsize=18)



gs = GridSpec(5, 2, figure=fig)



ax0 = fig.add_subplot(gs[0, 0])

ax1 = fig.add_subplot(gs[0, 1])

ax2 = fig.add_subplot(gs[1, :])

ax3 = fig.add_subplot(gs[2, :])

ax4 = fig.add_subplot(gs[3, :])

ax5 = fig.add_subplot(gs[4, :])



pd.Series(comm_kernels.groupby('Year').size()).plot(style='.-', ax=ax0, color='black')

comm_kernels.groupby('Year').TotalComments.sum().plot(style='.-', ax=ax1, color='black')

comm_kernels.groupby(['Year'])[['Question']].mean().plot(style='.-', ax=ax2, color='green')

comm_kernels.groupby(['Year'])[['Thank']].mean().plot(style='.-', ax=ax3, color='firebrick')

comm_kernels.groupby(['Year'])[['Upvote']].mean().plot(style='.-', ax=ax4, color='darkorange')

comm_kernels.groupby(['Year'])[['Promotion']].mean().plot(style='.-', ax=ax5, color='darkblue')



ax0.set_title('Number of Notebooks created per year', fontsize=14)

ax1.set_title('Number of comments on Notebooks per year', fontsize=14)

ax2.set_title('Average number of comments with a question', fontsize=14)

ax3.set_title('Average number of comments with a thank you', fontsize=14)

ax4.set_title('Average number of comments with a request of upvote', fontsize=14)

ax5.set_title('Average number of comments promoting a notebook', fontsize=14)



for ax in [ax2, ax3, ax4, ax5]:

    ax.legend().set_visible(False)





plt.show()
text = messages.Message.values

wordcloud = WordCloud(

    width = 1000,

    height = 500,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()