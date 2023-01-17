import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline



data = pd.read_csv('../input/HR_comma_sep.csv')




plt.figure(figsize = (8,8))

plt.subplot(1,2,1)

plt.plot(data.satisfaction_level[data.left == 1],data.last_evaluation[data.left == 1],'o', alpha = 0.1)

plt.ylabel('Last Evaluation')

plt.title('Employees who left')

plt.xlabel('Satisfaction level')



plt.subplot(1,2,2)

plt.title('Employees who stayed')

plt.plot(data.satisfaction_level[data.left == 0],data.last_evaluation[data.left == 0],'o', alpha = 0.11)

plt.xlim([0.4,1])

plt.ylabel('Last Evaluation')

plt.xlabel('Satisfaction level')
from sklearn.cluster import KMeans

kmeans_df =  data[data.left == 1].drop([ u'number_project',

       u'average_montly_hours', u'time_spend_company', u'Work_accident',

       u'left', u'promotion_last_5years', u'sales', u'salary'],axis = 1)

kmeans = KMeans(n_clusters = 3, random_state = 0).fit(kmeans_df)

kmeans.cluster_centers_
left = data[data.left == 1]

left['label'] = kmeans.labels_

plt.figure()

plt.xlabel('Satisfaction Level')

plt.ylabel('Last Evaluation')

plt.title('3 Clusters of employees who left')

plt.plot(left.satisfaction_level[left.label==0],left.last_evaluation[left.label==0],'o', alpha = 0.2, color = 'r')

plt.plot(left.satisfaction_level[left.label==1],left.last_evaluation[left.label==1],'o', alpha = 0.2, color = 'g')

plt.plot(left.satisfaction_level[left.label==2],left.last_evaluation[left.label==2],'o', alpha = 0.2, color = 'b')

plt.legend(['Winners','Frustrated','Bad Match'], loc = 3, fontsize = 15,frameon=True)
winners_hours_std = np.std(left.average_montly_hours[left.label == 0])

frustrated_hours_std = np.std(left.average_montly_hours[left.label == 1])

bad_match_hours_std = np.std(left.average_montly_hours[left.label == 2])

winners = left[left.label ==0]

frustrated = left[left.label == 1]

bad_match = left[left.label == 2]



def get_pct(df1,df2, value_list,feature):

    pct = []

    for value in value_list:

        pct.append(np.true_divide(len(df1[df1[feature] == value]),len(df2[df2[feature] == value])))

    return pct

columns = ['sales','winners','bad_match','frustrated']

winners_list = get_pct(winners,left,np.unique(left.sales),'sales')

frustrated_list = get_pct(frustrated,left,np.unique(left.sales),'sales')

bad_match_list = get_pct(bad_match,left,np.unique(left.sales),'sales')

plot_df = pd.DataFrame(columns = columns)

plot_df['sales'] = np.unique(left.sales)

plot_df['winners'] = winners_list

plot_df['bad_match'] = bad_match_list

plot_df['frustrated'] =frustrated_list

plot_df = plot_df.sort(columns = 'bad_match')







plt.figure()

values = np.unique(left.sales)

plt.bar(range(len(values)),plot_df.winners,width = 1, color = 'r',bottom=plot_df.bad_match + plot_df.frustrated)

plt.bar(range(len(values)),plot_df.frustrated, width = 1, color = 'g',bottom=plot_df.bad_match)

plt.bar(range(len(values)),plot_df.bad_match, width = 1, color = 'b')

plt.xticks(range(len(values))+ 0.5*np.ones(len(values)),plot_df.sales, rotation= 30)

plt.legend(['Winners','Frustrated','Bad Match'], loc = 3,frameon=True)



plt.title('Split of workers into the clusters for each category')
def get_num(df,value_list,feature):

    out = []

    for val in value_list:

        out.append(np.true_divide(len(df[df[feature] == val]),len(df)))

    return out



winners_list = get_num(winners,np.unique(left.sales),'sales')

frustrated_list = get_num(frustrated,np.unique(left.sales),'sales')

bad_match_list = get_num(bad_match,np.unique(left.sales),'sales')

plot_df = pd.DataFrame(columns = columns)

plot_df['sales'] = np.unique(left.sales)

plot_df['winners'] = winners_list

plot_df['bad_match'] = bad_match_list

plot_df['frustrated'] = frustrated_list

plot_df = plot_df.sort(columns = 'bad_match')



plt.figure()

values = np.unique(left.sales)

plt.bar(range(len(values)),plot_df.winners,width = 0.25, color = 'r')

plt.bar(range(len(values))+0.25*np.ones(len(values)),plot_df.frustrated, width = 0.25, color = 'g')

plt.bar(range(len(values))+0.5*np.ones(len(values)),plot_df.bad_match, width = 0.25, color = 'b')

plt.xticks(range(len(values))+ 0.5*np.ones(len(values)),plot_df.sales, rotation= 30)

plt.legend(['Winners','Frustrated','Bad Match'], loc = 2)



plt.title('% of Workers in each cluster')
plt.figure()

import seaborn as sns

sns.kdeplot(winners.average_montly_hours, color = 'r')

sns.kdeplot(bad_match.average_montly_hours, color ='b')

sns.kdeplot(frustrated.average_montly_hours, color ='g')

plt.legend(['Winners','Bad Match','Frustrated'])

plt.title('Hours per month distribution')
print('HR - Average monthly hours')

print (' "Winners" ',np.mean(left.average_montly_hours[(left.sales == 'hr') & (left.label == 0)]))

print (' "Frustrated" ',np.mean(left.average_montly_hours[(left.sales == 'hr') & (left.label == 1)]))

print (' "Bad Match" ',np.mean(left.average_montly_hours[(left.sales == 'hr') & (left.label == 2)]))

print('R&D -  Average monthly hours')

print (' "Winners" ',np.mean(left.average_montly_hours[(left.sales == 'RandD') & (left.label == 0)]))

print (' "Frustrated" ',np.mean(left.average_montly_hours[(left.sales == 'RandD') & (left.label == 1)]))

print (' "Bad Match"',np.mean(left.average_montly_hours[(left.sales == 'RandD') & (left.label == 2)]))
plt.figure()

plt.bar(0,np.mean(winners.number_project), color = 'r')

plt.bar(1,np.mean(frustrated.number_project), color = 'g')

plt.bar(2,np.mean(bad_match.number_project), color = 'b')

plt.title('Average Number of Projects')

plt.xticks([0.4,1.4,2.4],['Winners','Frustrated','Bad Match'])