# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd.options.mode.chained_assignment = None



chunk = pd.read_csv('../input/emails.csv', chunksize=500)

data = next(chunk)



data.info()

print(data.message[2])
def get_text(Series, row_num_slicer):

    """returns a Series with text sliced from a list split from each message. Row_num_slicer

    tells function where to slice split text to find only the body of the message."""

    result = pd.Series(index=Series.index)

    for row, message in enumerate(Series):

        message_words = message.split('\n')

        del message_words[:row_num_slicer]

        result.iloc[row] = message_words

    return result



def get_row(Series, row_num):

    """returns a single row split out from each message. Row_num is the index of the specific

    row that you want the function to return."""

    result = pd.Series(index=Series.index)

    for row, message in enumerate(Series):

        message_words = message.split('\n')

        message_words = message_words[row_num]

        result.iloc[row] = message_words

    return result



def get_address(df, Series, num_cols=1):

    """returns a specified email address from each row in a Series"""

    address = re.compile('[\w\.-]+@[\w\.-]+\.\w+')

    addresses = []

    result1 = pd.Series(index=df.index)

    result2 = pd.Series(index=df.index)

    result3 = pd.Series(index=df.index)

    for i in range(len(df)):

        for message in Series:

            correspondents = re.findall(address, message)

            addresses.append(correspondents)

            result1[i] = addresses[i][0]

        if num_cols >= 2:

            if len(addresses[i]) >= 3:

                result2[i] = addresses[i][1]

                if num_cols == 3:

                    if len(addresses[i]) >= 4:

                        result3[i] = addresses[i][2]

    return result1, result2, result3



def standard_format(df, Series, string, slicer):

    """Drops rows containing messages without some specified value in the expected locations. 

    Returns original dataframe without these values. Don't forget to reindex after doing this!!!"""

    rows = []

    for row, message in enumerate(Series):

        message_words = message.split('\n')

        if string not in message_words[slicer]:

            rows.append(row)

    df = df.drop(df.index[rows])

    return df
x = len(data.index)

headers = ['Message-ID: ', 'Date: ', 'From: ', 'To: ', 'Subject: ']

for i, v in enumerate(headers):

    data = standard_format(data, data.message, v, i)

data = data.reset_index()

print("Got rid of {} useless emails! That's {}% of the total number of messages in this dataset.".format(x - len(data.index), np.round(((x - len(data.index)) / x) * 100, decimals=2)))
data['text'] = get_text(data.message, 15)

data['date'] = get_row(data.message, 1)

data['senders'] = get_row(data.message, 2)

data['recipients'] = get_row(data.message, 3)

data['subject'] = get_row(data.message, 4)



data.date = data.date.str.replace('Date: ', '')

data.date = pd.to_datetime(data.date)



data.subject = data.subject.str.replace('Subject: ', '')



data['recipient1'], data['recipient2'], data['recipient3'] = get_address(data, data.recipients, num_cols=3)

data['sender'], x, y = get_address(data, data.senders)



del data['recipients']

del data['senders']

del data['file']

del data['message']



data = data[['date', 'sender', 'recipient1', 'recipient2', 'recipient3', 'subject', 'text']]



print(data.head())
import networkx as nx

import nxviz as nv



G = nx.from_pandas_dataframe(data, 'sender', 'recipient1', edge_attr=['date', 'subject'])

plot = nv.ArcPlot(G)

plot.draw()

plt.show() 
plot = nv.CircosPlot(G)

plot.draw()

plt.show()
plt.figure(figsize=(20,20))

pos = nx.spring_layout(G, k=.1)

nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')

plt.show()
cent = nx.degree_centrality(G)

name = []

centrality = []



for key, value in cent.items():

    name.append(key)

    centrality.append(value)



cent = pd.DataFrame()    

cent['name'] = name

cent['centrality'] = centrality

cent = cent.sort_values(by='centrality', ascending=False)



plt.figure(figsize=(10, 25))

_ = sns.barplot(x='centrality', y='name', data=cent[:15], orient='h')

_ = plt.xlabel('Degree Centrality')

_ = plt.ylabel('Correspondent')

_ = plt.title('Top 15 Degree Centrality Scores in Enron Email Network')

plt.show()
between = nx.betweenness_centrality(G)

name = []

betweenness = []



for key, value in between.items():

    name.append(key)

    betweenness.append(value)



bet = pd.DataFrame()

bet['name'] = name

bet['betweenness'] = betweenness

bet = bet.sort_values(by='betweenness', ascending=False)





plt.figure(figsize=(10, 25))

_ = sns.barplot(x='betweenness', y='name', data=bet[:15], orient='h')

_ = plt.xlabel('Degree Betweenness Centrality')

_ = plt.ylabel('Correspondent')

_ = plt.title('Top 15 Betweenness Centrality Scores in Hillary Clinton Email Network')

plt.show()