# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

from math import sqrt

from pprint import pprint

from collections import Counter



from matplotlib import pyplot as plt

from scipy.spatial import distance_matrix

from scipy.spatial.distance import pdist, cityblock

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster #cut_tree



from IPython.display import HTML

import string
with open('../input/agenda-result.json') as data_file:    

    raw_agenda = json.load(data_file)

if raw_agenda[-1]=={}:

    del raw_agenda[-1]



with open('../input/mps.json') as data_file:    

    raw_mps = json.load(data_file)



with open('../input/factions.json') as data_file:    

    raw_fractions = json.load(data_file)



with open('../input/dates.txt') as data_file:    

    d = [l for l in data_file]



""" Deputy dimension, map identifier id_mp to a dictionary of properties """

mps = { d['id_mp']: d for d in raw_mps['mp'] }



""" Parliamentary Fractions dimension """

fractions = { f['id']: f for f in raw_fractions['faction'] }
TYPE_EVENT_VOTING = '0'

def get_agenda_df():

    date_agenda, id_event,fraction, id_mp, yes = [], [], [], [], []

    for day in raw_agenda:

        for question in day['question']:

            for event_question in question['event_question']:

                if event_question['type_event']==TYPE_EVENT_VOTING:

                    for result_event in event_question['result_event']:

                        for result_by_name in result_event['result_by_name']:

                            date_agenda.append( day['date_agenda'] )

                            fraction.append( result_by_name['faction'] )

                            id_mp.append( result_by_name['id_mp'] )

                            id_event.append( result_event['id_event'] )

                            yes.append( 1.0 if result_by_name['result']==1 else 0.0 )

    return pd.DataFrame({'date_agenda':date_agenda, 'fraction':fraction, 'id_mp':id_mp, 'id_event':id_event, 'yes':yes})

agenda = get_agenda_df()
num_events = len(set(agenda['id_event']))

print('Total vote events: {:,}'.format(num_events))

#agenda.head()
MPsXEvents_raw = agenda.pivot('id_mp','id_event','yes')

event_ids = list(set(agenda['id_event']))

# for clustering analysis align all the vectors to the same dimensionality:

MPsXEvents = MPsXEvents_raw.fillna(0, inplace=False)

for i in MPsXEvents.axes[0]:

    MPsXEvents.loc[i, 'name'] = mps[i]['name']



fractionsXevents = agenda.groupby(['fraction','id_event'], as_index=False)['yes'].mean().pivot('fraction','id_event','yes')

for i in fractionsXevents.axes[0]:

    fractionsXevents.loc[i, 'name'] = fractions[i]['name']

cutoff=30



f, sub_plots = plt.subplots(7, 2,figsize=(12,4), sharex='none', sharey='none')

f.set_size_inches(10,24)

sub_plots = [ x for a in sub_plots for x in a ]



for metric, methods in [

    ('cityblock', ['single','average','weighted','complete']),

    ('euclidean', ['ward','centroid','median'])]:

    for method in methods:

        Z = linkage(MPsXEvents[event_ids], method, metric)

#         clust_sizes = [ v_c[1] for v_c in Counter(fcluster(Z, cutoff, 'maxclust')).most_common()]

#         print( "{:<10} {:<10} {}".format(method, metric, clust_sizes))

        p = sub_plots.pop(0)

        p.hist([ int(r[2]) for r in Z if (r[0]>460 and r[1]>460) ], bins=10 )

        p.set_xlim(0, num_events)

        p.set_xlabel("{}, {} distance".format(method, metric))

        p.set_ylabel("{}\n # MPs".format(method))

        

        p = sub_plots.pop(0)

        p.set_xlabel("MP, (Cluster)".format(method, metric))

        p.set_ylabel("{}, {} dist.".format(method, metric))

        dendrogram(

            Z,

            truncate_mode='lastp',  # show only the last p merged clusters

            p=cutoff,  # show only the last p merged clusters

            show_leaf_counts=True,  # otherwise numbers in brackets are counts

            leaf_rotation=90.,  # rotates the x axis labels

            leaf_font_size=8.,  # font size for the x axis labels

            show_contracted=True,  # to get a distribution impression in truncated branches

            ax=p,

        )



plt.show()

# Credits to https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
cluster_count_cutoff=15

for linkage_type in ['average', 'weighted', 'complete']:

    Z = linkage(MPsXEvents[event_ids], linkage_type, metric='cityblock')

    plt.title('Hierarchical Clustering Dendrogram,\nlinkage type "{}"'.format(linkage_type))

    plt.xlabel('sample index')

    plt.ylabel('distance')

    dendrogram(

        Z,

        truncate_mode='lastp',  # show only the last p merged clusters

        p=cluster_count_cutoff,  # show only the last p merged clusters

        show_leaf_counts=True,  # otherwise numbers in brackets are counts

        leaf_rotation=90.,  # rotates the x axis labels

        leaf_font_size=15.,  # font size for the x axis labels

        show_contracted=True,  # to get a distribution impression in truncated branches

    )

    yval, ylabel = plt.yticks()

    #plt.yticks(yval, [ int(y**2) for y in yval])

    plt.figure(1).set_size_inches(14,5)

    #print([ l.get_text() for l in plt.xticks()[1] ])

    plt.show()

Z = linkage(MPsXEvents[event_ids], 'average', metric='cityblock')



MPsXEvents['cluster_number'] = fcluster(Z, cluster_count_cutoff, 'maxclust')



significant_clusters = [ c_id for c_id, n in Counter(MPsXEvents['cluster_number']).most_common() if n>10 ]

clustersXevents = MPsXEvents[ MPsXEvents.cluster_number.isin(significant_clusters)].groupby('cluster_number', as_index=True)[event_ids].mean()



def HTMLCrossDistances():

    def listDistances(html, vector):

        PINK = "#ffbbbb"

        GREEN = "#bbffbb"

        FORMAT_MAX = ' bgcolor='+GREEN

        FORMAT_MIN = ' bgcolor='+PINK

        vectorNoNAN = vector.dropna()

        if len(vectorNoNAN) / len(vector) > 0.05:

            # skip vectors of more than 95% NaN

            clustersNoNAN = clustersXevents.append(vector).dropna(axis=1).iloc[:-1]

            distances = []

            for c1, rC in clustersNoNAN.iterrows():

                raw_d = cityblock(vectorNoNAN, rC)

                d = round(( 1 - raw_d / len(vectorNoNAN)) * 100)

                distances += [d]

            for d in distances:

                if d == min(distances):

                    highlight = FORMAT_MIN

                elif d == max(distances):

                    highlight = FORMAT_MAX

                else:

                    highlight = ''

                html += ['<td align=right {}>{:g}</td>'.format(highlight, d)]

        else:

            html += ['<td colspan={}>Not enough data</td>'.format(len(clustersXevents.index))]

    

    html=[]

    html+=['<table style="width:100%">']

    html+=['<tr><th>MP/Fraction Name</th><th>Cluster</th>']



    for c, r in clustersXevents.iterrows():

        html += ['<th>C{}</th>'.format(c)]

    html += ['</tr>']



    for c0, r in clustersXevents.iterrows():

        html += ['<tr><td>Cluster {}</td><td>C{}</td>'.format(c0,c0)]

        listDistances(html, clustersXevents[event_ids].loc[c0] )

        html += ['</tr>']



    for fraction, r in fractionsXevents.iterrows():

        html += ['<tr><td>{}</td><td>F{}</td>'.format(r['name'],fraction)]

        listDistances(html, fractionsXevents[event_ids].loc[fraction] )

        html += ['</tr>']



    for mp_id, r in MPsXEvents.iterrows():

        html += ['<tr><td>{}</td><td>C{:g}</td>'.format(r['name'], r['cluster_number'])]

        listDistances(html, MPsXEvents_raw[event_ids].loc[mp_id] )

        html += ['</tr>']

        

    html+=['</table>']



    return HTML(''.join(html))



HTMLCrossDistances()