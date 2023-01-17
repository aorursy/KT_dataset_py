# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from numpy import cov

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
nRowsRead = 12203 # specify 'None' if want to read whole file
# alexa.com_site_info.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df = pd.read_csv('/kaggle/input/sites-information-data-from-alexacom-dataset/alexa.com_site_info.csv', delimiter=',', nrows = nRowsRead)
df.dataframeName = 'alexa.com_site_info.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
def display_plot(item_1, item_2):
    x = df[item_1]
    y = df[item_2]
#     colors = (0, 0, 0)
    area = np.pi * 3

    # Plot
    plt.scatter(x, y, s=area, alpha=0.5)

    plt.title('Comaparison dataset columns')
    plt.xlabel(item_1)
    plt.ylabel(item_2)

    plt.plot()
    plt.show()
    
display_plot('comparison_metrics_data_bounce_rate_this_site_percentage', 'comparison_metrics_data_bounce_rate_comp_avg_percentage')
display_plot('comparison_metrics_search_traffic_this_site_percentage', 'comparison_metrics_data_bounce_rate_this_site_percentage')
display_plot('comparison_metrics_search_traffic_this_site_percentage', 'comparison_metrics_data_sites_linking_in_this_site_percentage')
display_plot('keyword_opportunities_breakdown_optimization_opportunities', 'keyword_opportunities_breakdown_keyword_gaps')
display_plot('keyword_opportunities_breakdown_optimization_opportunities', 'keyword_opportunities_breakdown_buyer_keywords')
display_plot('keyword_opportunities_breakdown_easy_to_rank_keywords', 'keyword_opportunities_breakdown_keyword_gaps')
display_plot('keyword_opportunities_breakdown_buyer_keywords', 'keyword_opportunities_breakdown_easy_to_rank_keywords')
display_plot('This_site_rank_in_global_internet_engagement', 'comparison_metrics_data_bounce_rate_this_site_percentage')
display_plot('This_site_rank_in_global_internet_engagement', 'keyword_opportunities_breakdown_keyword_gaps')
display_plot('This_site_rank_in_global_internet_engagement', 'keyword_opportunities_breakdown_buyer_keywords')
display_plot('This_site_rank_in_global_internet_engagement', 'keyword_opportunities_breakdown_easy_to_rank_keywords')
display_plot('This_site_rank_in_global_internet_engagement', 'comparison_metrics_data_bounce_rate_comp_avg_percentage')
display_plot('keyword_opportunities_breakdown_easy_to_rank_keywords', 'keyword_opportunities_breakdown_buyer_keywords')
display_plot('all_topics_keyword_gaps_Avg_traffic_parameter_3', 'all_topics_buyer_keywords_Avg_traffic_parameter_4')

# x = df['This_site_rank_in_global_internet_engagement']
# y = df['all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1']
# plt.plot(x, y)
display_plot('This_site_rank_in_global_internet_engagement', 'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1')

display_plot('keyword_opportunities_breakdown_optimization_opportunities', 'keyword_opportunities_breakdown_keyword_gaps')
display_plot('keyword_opportunities_breakdown_keyword_gaps', 'keyword_opportunities_breakdown_buyer_keywords')
display_plot('comparison_metrics_data_sites_linking_in_this_site_percentage', 'comparison_metrics_data_sites_linking_in_comp_avg_percentage')
display_plot('keyword_opportunities_breakdown_buyer_keywords', 'all_topics_keyword_gaps_Avg_traffic_parameter_4')
# display_plot('keyword_opportunities_breakdown_buyer_keywords', 'all_topics_buyer_keywords_Avg_traffic_parameter_4')
display_plot('all_topics_buyer_keywords_Avg_traffic_parameter_4', 'audience_overlap_sites_overlap_scores_parameter_4')
df.boxplot()
# df.hist()
# Display relations between columns for neural network input
def display_related_columns(critical_item, item_1, item_2, item_3):
    display_plot(critical_item, item_1)
    display_plot(critical_item, item_2)
    display_plot(critical_item, item_3)
## neural network inputs relation for all_topics_buyer_keywords_Avg_traffic_parameter_1
display_related_columns('all_topics_buyer_keywords_Avg_traffic_parameter_1', 'all_topics_keyword_gaps_Avg_traffic_parameter_1', 'audience_overlap_sites_overlap_scores_parameter_1', 'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1')
## neural network inputs relation for  audience_overlap_sites_overlap_scores_parameter_1
display_related_columns('audience_overlap_sites_overlap_scores_parameter_1', 'all_topics_keyword_gaps_Avg_traffic_parameter_1', 'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1', 'all_topics_buyer_keywords_Avg_traffic_parameter_1')
## neural network inputs relation for all_topics_keyword_gaps_Avg_traffic_parameter_1
display_related_columns('all_topics_keyword_gaps_Avg_traffic_parameter_1', 'all_topics_buyer_keywords_Avg_traffic_parameter_2', 'audience_overlap_sites_overlap_scores_parameter_2', 'comparison_metrics_search_traffic_Comp Avg_percentage')
## neural network inputs relation for comparison_metrics_search_traffic_Comp Avg_percentage
display_related_columns('comparison_metrics_search_traffic_Comp Avg_percentage', 'all_topics_keyword_gaps_Avg_traffic_parameter_1', 'all_topics_buyer_keywords_Avg_traffic_parameter_4', 'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1')
# dicts = df.to_dict()
# print(dicts)
# for item in dicts:
#     dicts[item].apply(lambda x : str(x) = 'NaN' if str(x) == '0' else str(x))

# print(dicts)
# df['This_site_rank_in_global_internet_engagement'] = pd.to_numeric(df['This_site_rank_in_global_internet_engagement'], error='coerse')

# df.describe()
# Determine correlation between columns :
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    #  and ('audience_overlap_' in col or 'keyword_gaps_Avg_' in col or '_relevance_to_site_parameter_1' in col or '_keywords_Avg_traffic_parameter_2' in col or 'This_site_rank_in_global_internet_engagement' in col or 'Daily_time_on_site' in col or 'keyword_opportunities_' in col or 'keyword_gaps_search_popularity_parameter_1' in col or '_easy_to_rank_keywords_search_pop_parameter_1' in col)
#     df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1 and ('audience_overlap_sites_overlap_scores_parameter_4' in col or '_keyword_gaps_Avg_traffic_parameter_3' in col or '_keyword_gaps_Avg_traffic_parameter_4' in col in col or 'comparison_metrics_data_' in col or '_relevance_to_site_parameter_1' in col or '_keywords_Avg_traffic_parameter_4' in col or 'rank_in_global_internet_engagement' in col or 'Daily_time_on_site' in col or 'keyword_opportunities_' in col or 'keyword_gaps_search_popularity_parameter_1' in col or '_easy_to_rank_keywords_search_pop_parameter_1' in col)]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
plotCorrelationMatrix(df, 14)
df.corr()
df_corr = df.corr(method ='pearson')
# df_corr

# Find maximum correlation between 2 columns

numeric_columns = []
str_counter = 0
for item in df:
    try:
        y = 0
        x = df[item][1] * 2
        z = x + y
        numeric_columns.append(item)
    except:
        str_counter += 1
print(str_counter)

for i in range(len(numeric_columns)):
    for j in range(i + 1, len(numeric_columns) - 1):
        if df[numeric_columns[i]].corr(df[numeric_columns[j]]) > 0.5 and df[numeric_columns[i]].corr(df[numeric_columns[j]]) < 0.6 and numeric_columns[i].split('parameter_')[0] not in numeric_columns[j]:
            print(numeric_columns[i] , numeric_columns[j], str(df[numeric_columns[i]].corr(df[numeric_columns[j]])))
        
x = np.sort(df['all_topics_keyword_gaps_Avg_traffic_parameter_1'])
y = np.sort(df['all_topics_buyer_keywords_Avg_traffic_parameter_2'])
plt.xlabel('all_topics_keyword_gaps_Avg_traffic_parameter_1')
plt.ylabel('all_topics_buyer_keywords_Avg_traffic_parameter_2')
plt.plot(x, y)
df['all_topics_keyword_gaps_Avg_traffic_parameter_1'] = np.sort(df['all_topics_keyword_gaps_Avg_traffic_parameter_1'])
df['all_topics_buyer_keywords_Avg_traffic_parameter_2'] = np.sort(df['all_topics_buyer_keywords_Avg_traffic_parameter_2'])
df['audience_overlap_sites_overlap_scores_parameter_1'] = np.sort(df['audience_overlap_sites_overlap_scores_parameter_1'])                                                                  
df['all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1'] = np.sort(df['all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1'])

df[['all_topics_keyword_gaps_Avg_traffic_parameter_1', 'all_topics_buyer_keywords_Avg_traffic_parameter_2','audience_overlap_sites_overlap_scores_parameter_1', 'all_topics_easy_to_rank_keywords_relevance_to_site_parameter_1']].plot()
df['This_site_rank_in_global_internet_engagement'] = np.sort(df['This_site_rank_in_global_internet_engagement'])
df['Daily_time_on_site'] = np.sort(df['Daily_time_on_site'])

# df.hist(column ='This_site_rank_in_global_internet_engagement', by ='Daily_time_on_site')
df.head(20)

numeric_columns = []
str_counter = 0
for item in df:
    try:
        y = 0
        x = df[item][1] * 2
        z = x + y
        numeric_columns.append(item)
    except:
        str_counter += 1
print(str_counter)

for i in range(len(numeric_columns)):
    for j in range(i + 1, len(numeric_columns) - 1):
        if df[numeric_columns[i]].corr(df[numeric_columns[j]]) > 0.6 and numeric_columns[i].split('r_')[0] not in numeric_columns[j]:
            print(numeric_columns[i], numeric_columns[j], )
            print(df[numeric_columns[i]].corr(df[numeric_columns[j]]))
# all_topics_keyword_gaps_Avg_traffic_parameter_2 all_topics_buyer_keywords_Avg_traffic_parameter_3
df.isnull().sum()
values = {'all_topics_keyword_gaps_Avg_traffic_parameter_2': int(df['all_topics_buyer_keywords_Avg_traffic_parameter_3'].mean()), 'all_topics_buyer_keywords_Avg_traffic_parameter_3': int(df['all_topics_buyer_keywords_Avg_traffic_parameter_3'].mean())}
values
df = df.fillna(value=values)
df
df.isnull().sum()
def create_plot(item_1, item_2):
    colors = (0,0,0)
    area = np.pi*3
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.gcf().set_size_inches(plt.gcf().get_size_inches()[0]*2, plt.gcf().get_size_inches()[1]*2, forward=True)
    # Plot
    # plt.subplot(8, 4, 1)
    plt.scatter(df[item_1], df[item_2])
    plt.xlabel(item_1)
    plt.ylabel(item_2)

create_plot('all_topics_keyword_gaps_Avg_traffic_parameter_2', 'all_topics_buyer_keywords_Avg_traffic_parameter_3')
new_df  = df[['all_topics_keyword_gaps_Avg_traffic_parameter_2', 'all_topics_buyer_keywords_Avg_traffic_parameter_3']]
new_df
### find perfect theta_0 and theta_1

theta_0 = -5.0625;
theta_1 = 0.20625;
y_pred = theta_0 + theta_1*new_df['all_topics_buyer_keywords_Avg_traffic_parameter_3'];

np.square(np.subtract(new_df['all_topics_keyword_gaps_Avg_traffic_parameter_2'],y_pred)).mean()/2
plt.scatter(new_df['all_topics_buyer_keywords_Avg_traffic_parameter_3'], new_df['all_topics_keyword_gaps_Avg_traffic_parameter_2'])
plt.plot(new_df['all_topics_buyer_keywords_Avg_traffic_parameter_3'] , y_pred, color = "g") 
plt.show()

