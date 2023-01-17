import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
# Sorry for this error, cause I made the dataset private.
data = pd.read_csv("../input/classifier_sample.csv")
# To make sure that I will not modify the orginal dataset, I copied this data into a variable 
sample = data.copy()
print(f'The dataset has {sample.shape[0]} rows and {sample.shape[1]} columns.')
sample.dtypes
sample_1a = sample.copy()
sample_1a.iloc[:, [1,2,3,4,6,7,8]].replace('', np.nan, inplace=True)
columns_list = list(sample_1a)
for i in columns_list:
    sample_1a = sample_1a.drop(sample_1a[sample_1a[i].isna()].index)
sample_1a.head()
print(f'After removing all rows that have empty value, the dataset has {sample_1a.shape[0]} rows and {sample_1a.shape[1]} columns.')
# change data type of 2 columns publication_time and verification_date to easily do the next step of cleaning
sample_1a['publication_time'] = pd.to_datetime(sample_1a['publication_time'], format="%Y-%m-%dT%H:%M:%S", utc = True)
sample_1a['verification_date'] = pd.to_datetime(sample_1a['verification_date'])

def convert_tobool(data):
    if data['is_relevant'] == True: return True
    else: return False
sample_1a['is_relevant'] = sample_1a.apply(convert_tobool, axis=1)
sample_1b = sample_1a.copy()
sample_1b['rank'] = sample_1b.sort_values(['duplicate_id', 'verification_date', 'publication_time'], ascending=[True, False, True])\
                .groupby(['duplicate_id']).cumcount() + 1
sample_1b = sample_1b[sample_1b['rank'] == 1]
sample_1b.head()
print(f'After removing duplicate rows based on given criteria, the dataset has {sample_1b.shape[0]} rows and {sample_1b.shape[1]} columns.')
by_topic = sample_1a.groupby('topic').agg({'publication_id': 'count'})
by_topic
ax = by_topic.sort_values('publication_id')\
    .plot(kind='barh', title ="Distribution of publication by topic", figsize=(12, 6), legend = False, fontsize=10)
ax.set_ylabel("Topic", fontsize=12)
ax.set_xlabel("Number of publications", fontsize=12)
plt.show()
by_lan_channel = sample_1a.groupby(['publication_channel', 'publication_language']).agg({'publication_id': 'count'})
by_lan_channel = by_lan_channel.reset_index()
by_lan_channel
by_lan_channel_vi = by_lan_channel.copy()
by_lan_channel_vi = by_lan_channel_vi.groupby(['publication_channel', 'publication_language'])\
                            ['publication_id'].sum().unstack('publication_language').fillna(0)
# by_lan_channel_vi
by_lan_channel_vi.plot(kind='bar', stacked=True,\
                              title ='Distribution of publication by Channel and Language'\
                              , figsize=(10, 5), rot = 90)
plt.xlabel('Publication Channel')
plt.ylabel('Number of publications')
plt.legend(loc = 'upper left')
plt.show()
by_lan_topic_relevant = sample_1a.groupby(['publication_language', 'topic', 'is_relevant'])\
                        .agg({'publication_id': 'count'})
by_lan_topic_relevant = by_lan_topic_relevant.reset_index()
by_lan_topic_relevant.head()
by_probability_relevant = sample_1a.groupby(['predicted_relevance_probability',\
                                           'is_relevant']).agg({'publication_id': 'count'})
by_probability_relevant = by_probability_relevant.reset_index()
f = plt.figure(figsize=(10,5))
for i in by_probability_relevant.is_relevant.unique():
    
    # Set up the plot   
    ax = f.add_subplot(2, 1, i+1)
    # Draw the plot
    ax.hist(by_probability_relevant['predicted_relevance_probability']\
            [by_probability_relevant['is_relevant'] == i],
           bins = 20, edgecolor='black')

    # Title and labels
    ax.set_title(f"Distribution of predicted_relevance_probability when is_relevant {i}", size = 12)
    ax.set_xlabel('Intervals of predicted relevance probability', size = 10)
    ax.set_ylabel('Number of publications', size= 10)

plt.tight_layout()
plt.show()
for i in by_lan_topic_relevant.is_relevant.unique():
    by_lan_topic_relevant_vi = by_lan_topic_relevant[by_lan_topic_relevant['is_relevant'] == i]
    by_lan_topic_relevant_vi = by_lan_topic_relevant_vi.groupby(['publication_language','topic'])\
                                ['publication_id'].sum().unstack('publication_language').fillna(0)
    by_lan_topic_relevant_vi.plot(kind='bar', stacked=True,\
                                  title =f"Distribution of topic by language when is_relevant {i}"\
                                  , figsize=(10, 4), rot = 90)
    plt.legend(loc = 'upper left')
    plt.show()

sample_3a = sample_1a.copy()
sample_3a.dtypes
sample_3a['prediction'] = np.where(sample_3a['predicted_relevance_probability'] >= 0.5, True, False)
sample_3a.head()
def cal_metrics(data):
    result = []
    recall_score_result = recall_score(data['is_relevant'],\
                         data['prediction'], average='binary')
    precision_score_result = precision_score(data['is_relevant'],\
                         data['prediction'], average='binary')
    f1_score_result = f1_score(data['is_relevant'],\
                         data['prediction'], average='binary')
    accuracy_score_result = accuracy_score(data['is_relevant'], data['prediction'])
    result.append(recall_score_result)
    result.append(precision_score_result)
    result.append(f1_score_result)
    result.append(accuracy_score_result)
    return result
performance_metrics_3a = cal_metrics(sample_3a)
performance_metrics_3a
sample_3b = sample_1a.copy()
sample_3b = sample_3b[sample_3b['publication_language'] == 'de']
threshold_list_3b = np.arange(0.4, 1, 0.1).tolist()
performance_metrics_3b  = []
for i in threshold_list_3b:
    sample_3b['prediction'] = np.where(sample_3b['predicted_relevance_probability'] >= i, True, False)
    sub_result = cal_metrics(sample_3b)
    sub_result.append(i)
    performance_metrics_3b.append(sub_result)
performance_metrics_3b = pd.DataFrame(performance_metrics_3b)
performance_metrics_3b.columns =['recall_score','precision_score','f1_score', 'accuracy_score', 'threshold']
performance_metrics_3b.sort_values('f1_score', ascending = False)
sample_3c = sample_1a.copy()
sample_3c = sample_3c[sample_3c['publication_language'] == 'en']
# warning: it may take around 2-3 mins to complete
threshold_list_3c = np.arange(0, 1.0001, 0.0001).tolist()

performance_metrics_3c = []

def cal_f1_score(data):
    result = []
    f1_score_result = f1_score(data['is_relevant'],\
                         data['prediction'], average='binary')
    result.append(f1_score_result)
    return result

for i in threshold_list_3c:
    sample_3c['prediction'] = np.where(sample_3c['predicted_relevance_probability'] >= i, True, False)
    sub_result_3c = cal_f1_score(sample_3c)
    sub_result_3c.append(i)
    performance_metrics_3c.append(sub_result_3c)
performance_metrics_3c = pd.DataFrame(performance_metrics_3c)
performance_metrics_3c.columns =['f1_score', 'threshold']
performance_metrics_3c.sort_values('f1_score', ascending = False).head(10)