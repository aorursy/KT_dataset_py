# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import csv

import networkx as nx

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

import tensorflow as tf

from IPython.display import YouTubeVideo

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
frame_lvl_record = "../input/frame-sample/frame/train00.tfrecord"
print(os.listdir("../input/frame-sample/frame"))

print(os.listdir("../input/validate-sample/validate"))
sub = pd.read_csv('../input/sample_submission.csv')

sub.head()

print ("submission style ...")
vid_ids = []

labels = []



for example in tf.python_io.tf_record_iterator(frame_lvl_record):

    tf_example = tf.train.Example.FromString(example)

    vid_ids.append(tf_example.features.feature['id']

                   .bytes_list.value[0].decode(encoding='UTF-8'))

    labels.append(tf_example.features.feature['labels'].int64_list.value)
print('Number of videos in this tfrecord: ',len(vid_ids))

print ('Number of labels in this tfrecord: ', len (labels))

print('Picking a youtube video id:',vid_ids[15])
# With that video id, we can play the video

YouTubeVideo('UzXQaOLQVCU')
# due to execution time, we're only going to read the first video



feat_rgb = []

feat_audio = []



for example in tf.python_io.tf_record_iterator(frame_lvl_record):  

    tf_seq_example = tf.train.SequenceExample.FromString(example)

    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

    sess = tf.InteractiveSession()

    rgb_frame = []

    audio_frame = []

    # iterate through frames

    for i in range(n_frames):

        rgb_frame.append(tf.cast(tf.decode_raw(

                tf_seq_example.feature_lists.feature_list['rgb']

                  .feature[i].bytes_list.value[0],tf.uint8)

                       ,tf.float32).eval())

        audio_frame.append(tf.cast(tf.decode_raw(

                tf_seq_example.feature_lists.feature_list['audio']

                  .feature[i].bytes_list.value[0],tf.uint8)

                       ,tf.float32).eval())

        

        

    sess.close()

    

    feat_audio.append(audio_frame)

    feat_rgb.append(rgb_frame)

    break
print('The first video has %d frames' %len(feat_rgb[0]))
vocabulary = pd.read_csv('../input/vocabulary.csv')

vocabulary.head()
vocabulary.info()
from collections import Counter



label_mapping =  vocabulary[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']

print("we have {} unique labels in the dataset".format(len(vocabulary['Index'].unique())))
n = 30 # although, we'll only show those that appear in the 1,000 for this competition

top_n = Counter([item for sublist in labels for item in sublist]).most_common(n)

top_n_labels = [int(i[0]) for i in top_n]

top_n_label_names = [label_mapping[x] for x in top_n_labels if x in label_mapping] # filter out the labels that aren't in the 1,000 used for this competition

print(top_n_label_names)
labels_count_dict = dict(top_n)

labels_count_df = pd.DataFrame.from_dict(labels_count_dict, orient='index').reset_index()

labels_count_df.columns = ['label', 'count']

labels_count_df['label'] = labels_count_df['label'].map(label_mapping, na_action='ignore')

TOP_labels = list(labels_count_df['label'])[:n]

fig, ax = plt.subplots(figsize=(10,7))

sns.barplot(y='label', x='count', data=labels_count_df)

plt.title('Top {} labels with sample count'.format(n))
import networkx as nx

from itertools import combinations



G = nx.Graph()



G.clear()

for list_of_nodes in labels:

    filtered_nodes = set(list_of_nodes).intersection(set(top_n_labels) & 

                                                     set(vocabulary['Index'].unique()))  

    for node1,node2 in list(combinations(filtered_nodes,2)): 

        node1_name = label_mapping[node1]

        node2_name = label_mapping[node2]

        G.add_node(node1_name)

        G.add_node(node2_name)

        G.add_edge(node1_name, node2_name)



plt.figure(figsize=(9,9))

nx.draw_networkx(G, font_size="12")
plt.figure(figsize = (10,8))

vocabulary.groupby('Vertical1').TrainVideoCount.sum().plot(kind="bar")

plt.title("Average TrainVideoCount per vertical1")

plt.show()



plt.figure(figsize = (10,8))

vocabulary.groupby('Vertical1').Index.count().plot(kind="bar")

plt.title("Average number video per vertical1")

plt.show()
plt.figure(figsize = (10,8))

vocabulary.groupby('Vertical2').TrainVideoCount.sum().plot(kind="bar")

plt.title("Average TrainVideoCount per vertical2")

plt.show()



plt.figure(figsize = (10,8))

vocabulary.groupby('Vertical2').TrainVideoCount.count().plot(kind="bar")

plt.title("Average video number per vertical2")

plt.show()
plt.figure(figsize = (10,8))

vocabulary.groupby('Vertical3').TrainVideoCount.sum().plot(kind="bar")

plt.title("Average TrainVideoCount per vertical3")

plt.show()



plt.figure(figsize = (10,8))

vocabulary.groupby('Vertical3').TrainVideoCount.count().plot(kind="bar")

plt.title("Average video number per vertical3")

plt.show()
sns.lmplot(x='Index', y='TrainVideoCount', data=vocabulary , size=15)
vocabulary.groupby('Vertical1').corr()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='black',

                          stopwords=stopwords,

                          max_words=1000,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(vocabulary['WikiDescription']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - description")

plt.axis('off')

plt.show()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='black',

                          stopwords=stopwords,

                          max_words=1000,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(vocabulary['Name']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - Name")

plt.axis('off')

plt.show()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='black',

                          stopwords=stopwords,

                          max_words=1000,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(vocabulary['Vertical1']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - Vertical1")

plt.axis('off')

plt.show()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='black',

                          stopwords=stopwords,

                          max_words=1000,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(vocabulary['Vertical2']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - Vertical2")

plt.axis('off')

plt.show()
with open('../input/vocabulary.csv', 'r') as f:

  vocabularylist = list(csv.reader(f))



T1=[]



for l in vocabularylist:

    if l[5] != 'NaN' and l[6] !='NaN' and l[5] != '' and l[6] !='' and l[5] !=  l[6] :

        c1 = l[5]

        c2 = l[6]

        tuple = (c1, c2)

    if l[5] != 'NaN' and l[7] !='NaN' and l[5] != '' and l[7] !='' and l[5] !=  l[7] :

        c1 = l[5]

        c2 = l[7]

        tuple = (c1, c2)

    if l[6] != 'NaN' and l[7] !='NaN' and l[6] != '' and l[7] !='' and l[7] !=  l[6] :

        c1 = l[6]

        c2 = l[7]

        tuple = (c1, c2)

    T1.append(tuple)

    

edges = {k: T1.count(k) for k in set(T1)}

print ("List = ",len(edges), "elements")

edges
B = nx.DiGraph()

nodecolor=[]

for ed, weight in edges.items():

    if ed[0]!='Vertical2' and ed[0]!='Vertical3' and  ed[1]!='Vertical2' and ed[1]!='Vertical3':

        B.add_edge(ed[0], ed[1], weight=weight)

for k in B.nodes:

    if (k == "Beauty & Fitness"):

        nodecolor.append('blue')

    elif (k == "News"):

        nodecolor.append('Magenta')

    elif (k == "Food & Drink"):

        nodecolor.append('crimson')

    elif (k == "Health"):

        nodecolor.append('green')

    elif (k == "Science"):

        nodecolor.append('yellow')

    elif (k == "Business & Industrial"):

        nodecolor.append('cyan')

    elif (k == "Home & Garden"):

        nodecolor.append('darkorange')

    elif (k == "Travel"):

        nodecolor.append('slategrey')

    elif (k == "Arts & Entertainment"):

        nodecolor.append('red')

    elif (k == "Games"):

        nodecolor.append('grey')

    elif (k == "People & Society"):

        nodecolor.append('lightcoral')

    elif (k == "Shopping"):

        nodecolor.append('maroon')

    elif (k =="Computers & Electronics"):

        nodecolor.append('orangered')

    elif (k == "Hobbies & Leisure"):

        nodecolor.append('saddlebrown')

    elif (k == "Sports"):

        nodecolor.append('lawngreen')

    elif (k == "Real Estate"):

        nodecolor.append('deeppink')

    elif (k == "Finance"):

        nodecolor.append('navy')

    elif (k == "Reference"):

        nodecolor.append('royalblue')

    elif (k == "Autos & Vehicles"):

        nodecolor.append('turquoise')

    elif (k == "Internet & Telecom"):

        nodecolor.append('lime')

    elif (k == "Law & Government"):

        nodecolor.append('palegreen')

    elif (k == "Jobs & Education"):

        nodecolor.append('springgreen')

    elif (k == "Pets & Animals"):

        nodecolor.append('lightpink')

    elif (k == "Books & Literature"):

        nodecolor.append('lightpink')
plt.figure(figsize = (15,15))

nx.draw(B, pos=nx.circular_layout(B), node_size=1500, with_labels=True, node_color=nodecolor)

nx.draw_networkx_edge_labels(B, pos=nx.circular_layout(B), edge_labels=nx.get_edge_attributes(B, 'weight'))

plt.title('Weighted graph representing the relationship between the categories', size=20)

plt.show()
# analyse

print('Quick Review')

print (20*'...',"\n")

print("number of node : %s" % B.number_of_nodes())

print("number of arcs : %s" % B.number_of_edges())



# arc entrant

indeg = 0

for n in B.in_degree():

    indeg += n[1]



# arc sortant

outdeg = 0

for n in B.in_degree():

    outdeg += n[1]



print('')

print("the number of edges pointing to the node : %s" % indeg)

print("the number of edges pointing to the outside of the node : %s" % outdeg)



# passage en graphe non orienté

G = B.to_undirected()



# min et max de degree

listmindegre = (0, 10)

listmaxdegre = (0, 0)

for n in G.degree():

    if (listmindegre[1] > n[1]):

        listmindegre = n

    if (listmaxdegre[1] < n[1]):

        listmaxdegre = n



print('')

print("The node that has the minimal degree is : ", listmindegre)

print("The node that has the maximum degree is : ", listmaxdegre)

edgdesmax=0

for ed,w in G.edges.items():

    if(w['weight']>edgdesmax):

        edgdesmax=w['weight']

        edgdescat=ed

edgdescat

print("both category ",edgdescat[0]," and ",edgdescat[1]," has the big relationship weight( w = ",edgdesmax,")")

   

# centrality

listmincentrality = (0, 10)

listmaxcentrality = (0, 0)

for n in (nx.betweenness_centrality(G)).items():

    if (listmincentrality[1] > n[1]):

        listmincentrality = n

    elif (listmaxcentrality[1] < n[1]):

        listmaxcentrality = n



print('')

print("The node that has minimal centrality is : ", listmincentrality)

print("The node that has the maximum centrality is : ", listmaxcentrality)



# normalized

listminnormalized = (0, 10)

listmaxnormalized = (0, 0)

for n in (nx.degree_centrality(G)).items():

    if (listminnormalized[1] > n[1]):

        listminnormalized = n

    elif (listmaxnormalized[1] < n[1]):

        listmaxnormalized = n



print('')

print("The node that has the minimum (normalized) degree is : ", listminnormalized)

print("The node that has the maximal (normalized) degree is: ", listmaxnormalized)
cl = list(nx.find_cliques(G))

print("estimate number of cliques %s" % nx.graph_number_of_cliques(G))

print("click on who has maximum number %s" % nx.graph_clique_number(G))

print('')



print(">> possible cases of clique:\n")

for cl in nx.find_cliques(G):

    if len(cl)==2 or len(cl)==3:

        print(cl)
pathlengths = []



for v in G.nodes():

    spl = nx.single_source_shortest_path_length(G, v)

    for p in spl.values():

        pathlengths.append(p)

print("average of the shortest paths %s" % round((sum(pathlengths) / len(pathlengths)), 3))



print('')



print("density : %s" % round(nx.density(G), 3))

print("diameter :", nx.diameter(G.subgraph(max(nx.connected_components(G), key=len))))



# eccentricity

listmineccentricity = (0, 10)

listmaxeccentricity = (0, 0)

for n in (nx.eccentricity(G.subgraph(max(nx.connected_components(G), key=len)))).items():

    if (listmineccentricity[1] > n[1]):

        listmineccentricity = n

    elif (listmaxeccentricity[1] < n[1]):

        listmaxeccentricity = n



print('')

print("The node that has the minimal eccentricity is : ", listmineccentricity)

print("The node that has the maximum eccentricity is : ", listmaxeccentricity)

print('')



print("center : %s" % nx.center(G.subgraph(max(nx.connected_components(G), key=len))))

print("periphery : %s" % nx.periphery(G.subgraph(max(nx.connected_components(G), key=len))))
plt.figure(figsize = (15,15))

nx.draw_random(B,  node_size=1500, with_labels=True, node_color=nodecolor)

nx.draw_networkx_edge_labels(B, pos=nx.circular_layout(B), edge_labels=nx.get_edge_attributes(B, 'weight'))

plt.title('Weighted graph representing the relationship between the categories', size=20)

plt.show()