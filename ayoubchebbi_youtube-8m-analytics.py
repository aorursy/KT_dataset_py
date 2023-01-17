import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import networkx as nx
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
vocabulary = pd.read_csv('../input/vocabulary.csv')
vocabulary.head()
vocabulary.describe()
vocabulary.info()
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
vocabulary.groupby('Vertical2').TrainVideoCount.count().plot(kind="bar")
plt.title("Average video number per vertical3")
plt.show()
sns.lmplot(x='Index', y='TrainVideoCount', data=vocabulary , size=15)
plt.figure(figsize = (10,8))
sns.heatmap(vocabulary.groupby('Vertical1').corr(), annot=True )
plt.show()
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
print('')
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


# recherche des cliques
print('')
cl = list(nx.find_cliques(G))
print("estimate number of cliques %s" % nx.graph_number_of_cliques(G))
print("click on who has maximum number %s" % nx.graph_clique_number(G))
print('')

print("possible cases of clique ")
for cl in nx.find_cliques(G):
    if len(cl)==2 or len(cl)==3:
        print(cl)

# plus courts chemins
pathlengths = []

for v in G.nodes():
    spl = nx.single_source_shortest_path_length(G, v)
    for p in spl.values():
        pathlengths.append(p)
print('')
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
from arcgis.gis import GIS