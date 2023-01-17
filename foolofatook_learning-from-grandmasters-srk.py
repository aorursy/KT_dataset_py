import pandas as pd

import gc



import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode



from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



init_notebook_mode()
users=pd.read_csv('../input/meta-kaggle/Users.csv')

followers=pd.read_csv('../input/meta-kaggle/UserFollowers.csv')

kernels=pd.read_csv('../input/meta-kaggle/Kernels.csv')

achievement = pd.read_csv('../input/meta-kaggle/UserAchievements.csv')
username = "sudalairajkumar" 

userId = users[users['UserName'] == username]['Id'].values[0]

del users
kernelGrandmasterDate = achievement[(achievement['UserId'] == userId) & (achievement['AchievementType'] == 'Scripts')]['TierAchievementDate']

competitionsGrandmasterDate = achievement[(achievement['UserId'] == userId) & (achievement['AchievementType'] == 'Competitions')]['TierAchievementDate']

followers = followers[followers['FollowingUserId'] == userId]

kernels = kernels[kernels['AuthorUserId'] == userId]

del achievement

gc.collect()
def formatKernelName(name):

    words = name.split('-')

    new_words = []

    for word in words:

        new_words.append(word[0].upper() + word[1:])

    new_word = ' '.join(new_words)

    return new_word



kernels['Kernel Name'] = kernels['CurrentUrlSlug'].apply(lambda x : formatKernelName(str(x)))
mostViewed = kernels.sort_values(by = 'TotalViews', ascending = False)

fig = px.bar(mostViewed.iloc[:20][::-1], x = 'TotalViews', y = 'Kernel Name',

             orientation = 'h', color = 'TotalViews',

             template = 'plotly_white',color_continuous_scale='Blugrn')

fig.update_layout(title_text='<b>Kernels With the Most Views</b>', title_x=0.5)

fig.show()
print(f'SRK has received an average of {int(kernels["TotalViews"].mean())} number of views per kernel')
mostUpvoted = kernels.sort_values(by = 'TotalVotes', ascending = False)

fig = px.bar(mostUpvoted.iloc[:20][::-1], x = 'TotalVotes', y = 'Kernel Name',

             orientation = 'h',color = 'TotalVotes',

            template = 'plotly_white',color_continuous_scale='Burg')

fig.update_layout(title_text='<b>Kernels With the Most Upvotes</b>', title_x=0.5)

fig.show()
print(f'SRK has received an average of {int(kernels["TotalVotes"].mean())} number of votes per kernel')
kernels['Views Per Upvote'] = kernels['TotalViews'] / kernels['TotalVotes']

viewsPerUpvote = kernels.sort_values(by = 'Views Per Upvote')

fig = px.bar(viewsPerUpvote.iloc[:20][::-1], x = 'Views Per Upvote', y = 'Kernel Name', 

             orientation = 'h',hover_data = ['TotalVotes'],template = 'plotly_white',

             color = 'Views Per Upvote',color_continuous_scale='Agsunset')             

fig.update_layout(title_text = '<b>Top 20 Kernels With the Least Views Per Upvote Ration</b>', title_x = 0.5)

fig.show()
print(f'The average Views Per Upvote for SRK is {int(kernels["Views Per Upvote"].mean())}')
noMedals = kernels[kernels['MedalAwardDate'].isna()]



fig = go.Figure(data=[go.Table(

    header=dict(values=['<b>Kernel Name</b>', '<b>Number of Votes</b>', '<b>Number of Views</b>'],

                line_color='black',

                fill_color='yellow',

                align='left',

                font=dict(color='black', size=14)),

    cells=dict(values=[noMedals['Kernel Name'],

                      noMedals['TotalVotes'],

                      noMedals['TotalViews']],

               line_color='black',

               fill_color='white',

               align='left',

               font=dict(color='black', size=13)))

])



fig.update_layout(margin=dict(l=80, r=80, t=25, b=10),

                  title = { 'text' : '<b>Kernels That Did not get Any Medals</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#ff0d00')

fig.show()

fig = make_subplots(rows = 1, cols = 2, subplot_titles=['Total Votes vs Total Views', 'Total Votes vs Total Comments'])



temp = px.scatter(kernels, x = 'TotalViews', y = 'TotalVotes', trendline='ols', template='plotly_white')

trendline = temp.data[1]

trace0 = go.Scatter(x = kernels['TotalViews'], y = kernels['TotalVotes'],mode = 'markers')

fig.add_trace(trace0, row = 1, col = 1)

fig.add_trace(trendline, row = 1, col = 1)



temp1 = px.scatter(kernels, x = 'TotalComments', y = 'TotalVotes', trendline='ols', template = 'plotly_white')

trendline1 = temp1.data[1]

trace1 = go.Scatter(x = kernels['TotalComments'], y = kernels['TotalVotes'],mode = 'markers')

fig.add_trace(trace1, row = 1, col = 2)

fig.add_trace(trendline1, row = 1, col = 2)

fig.update_layout(showlegend = False, template = 'ggplot2')

fig.show()
kernelsBeforeGrandMaster = kernels[kernels['CreationDate'] < kernelGrandmasterDate.values[0]]

kernelsAfterGrandmaster = kernels[kernels['CreationDate'] >= kernelGrandmasterDate.values[0]]
print(f"Number of Notebooks Created Before Becoming a Grandmaster is {kernelsBeforeGrandMaster.shape[0]}")

print(f"Number of Notebooks Created After Becoming a Grandmaster is {kernelsAfterGrandmaster.shape[0]}")
print(f"The average number of upvotes for Kernels Created before His Becoming Grandmaster is {int(kernelsBeforeGrandMaster['TotalVotes'].mean())}")

print(f"The average number of upvotes for Kernels Created after His Becoming Grandmaster is {int(kernelsAfterGrandmaster['TotalVotes'].mean())}")
print(f"The average number of Views for Kernels Created before His Becoming Grandmaster is {int(kernelsBeforeGrandMaster['TotalViews'].mean())}")

print(f"The average number of Views for Kernels Created after His Becoming Grandmaster is {int(kernelsAfterGrandmaster['TotalViews'].mean())}")
kernelsBeforeGrandMaster = kernelsBeforeGrandMaster.sort_values(by = 'TotalVotes', ascending=False).iloc[:5]

kernelsAfterGrandmaster = kernelsAfterGrandmaster.sort_values(by = 'TotalVotes', ascending=False).iloc[:5]



fig = make_subplots(specs=[[{'type':'table'}, {'type':'table'}]],rows = 1, cols = 2, subplot_titles=['Most Upvoted Notebooks Before Grandmaster', 'Most Upvoted Notebooks after Grandmaster'])



trace0 = go.Table(

    header=dict(values=['<b>Kernel Name</b>', '<b>Number of Votes</b>', '<b>Number of Views</b>'],

                line_color='black',

                fill_color='yellow',

                align='left',

                font=dict(color='black', size=14)),

    cells=dict(values=[kernelsBeforeGrandMaster['Kernel Name'],

                      kernelsBeforeGrandMaster['TotalVotes'],

                      kernelsBeforeGrandMaster['TotalViews']],

               line_color='black',

               fill_color='white',

               align='left',

               font=dict(color='black', size=13)))



fig.add_trace(trace0, row = 1, col = 1)



trace1 = go.Table(

    header=dict(values=['<b>Kernel Name</b>', '<b>Number of Votes</b>', '<b>Number of Views</b>'],

                line_color='black',

                fill_color='yellow',

                align='left',

                font=dict(color='black', size=14)),

    cells=dict(values=[kernelsAfterGrandmaster['Kernel Name'],

                      kernelsAfterGrandmaster['TotalVotes'],

                      kernelsAfterGrandmaster['TotalViews']],

               line_color='black',

               fill_color='white',

               align='left',

               font=dict(color='black', size=13)))



fig.add_trace(trace1, row = 1, col = 2)



fig.update_layout(margin=dict(l=80, r=80, t=25, b=10), height = 400)

fig.show()

stopwords = set(STOPWORDS)

def createCorpus():

    corpus = ""

    for des in kernels['Kernel Name'].to_list():

        corpus = corpus + ' ' + des

    return corpus



def generateWordCloud():

    plt.subplots(figsize=(12,8))

    corpus = createCorpus()

    wordcloud = WordCloud(background_color='yellow',

                          contour_color='black', contour_width=4, 

                          width=1500, margin=10,

                          stopwords=stopwords,

                          height=1080

                         ).generate(corpus)

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
generateWordCloud()