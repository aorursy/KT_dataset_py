import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt
from sklearn import preprocessing
%matplotlib inline
def getDataframeFromSql():
    conn = sqlite3.connect('../input/recommenderDb.sqlite')
    query = '''SELECT questionId, userId FROM vote'''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
df = getDataframeFromSql()
df.head(10)

def getItemEmbedings():
    conn = sqlite3.connect('../input/recommenderDb.sqlite')
    query = '''SELECT question_tag.questionId, question_tag.tagId, tag.subscriber_count FROM question_tag
                JOIN tag ON question_tag.tagId = tag.id'''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

itemsDf = getItemEmbedings()
itemsDf.head(10)
userTagsDf = pd.merge(df, itemsDf, on='questionId')
userTagsDf.head(10)
userTagCount = userTagsDf.groupby(['userId', 'tagId']).size().reset_index()
userTagCount.head(10)
def plot_bar_x(userId):
    plt.figure(figsize=(30,10))
    filteredDf = userTagCount[userTagCount['userId'] == userId]
#     print(filteredDf.groupby('tagId').head(10))
    tags = filteredDf['tagId'].unique()
    tags = [str(tag) for tag in tags]
    values = filteredDf[0]
    index = np.arange(len(tags))
    plt.bar(index, values)
    plt.xlabel('Tag Id', fontsize=30)
    plt.ylabel('No of votes', fontsize=30)
    plt.xticks(index, tags, fontsize=15)
    plt.title('User %s votes/tag' % (userId), fontsize=30)

plot_bar_x(4)