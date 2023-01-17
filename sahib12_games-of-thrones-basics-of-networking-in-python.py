# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
d1=pd.read_csv('/kaggle/input/book1.csv')
print(d1.shape)
#Printing head

d1.head()
print(d1['Type'].unique())
print(d1['Target'].value_counts())
# Importing modules

import networkx as nx



# Creating an empty graph object

b1 = nx.Graph()
for _, edge in d1[:50].iterrows():

    b1.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])
print("Total number of nodes: ", int(b1.number_of_nodes())) 
print("Total number of edges: ", int(b1.number_of_edges())) 
print("List of all nodes: ", list(b1.nodes())) 
print("List of all edges: ", list(b1.edges(data = True))) 

print("Degree for all nodes: ", dict(b1.degree())) 
  

print("List of all nodes we can go to in a single step from node of Jaime-Lannister	: ", 

                                                 list(b1.neighbors('Jaime-Lannister'))) 
# fixing the size of the figure 

plt.figure(figsize =(20, 20)) 

pos = nx.fruchterman_reingold_layout(b1) 

node_color = [b1.degree(v) for v in b1] 

# node colour is a list of degrees of nodes 



node_size = 100

# size of node 

edge_width = [0.15 * b1[u][v]['weight'] for u, v in b1.edges()] 

# width of edge is a list of weight of edges 



nx.draw_networkx(b1,pos, node_size = node_size, 

                node_color = 'green', alpha = 0.7, 

                with_labels = True, width = edge_width, 

                edge_color ='.4', cmap = plt.cm.Blues) 



plt.axis('off') 

plt.tight_layout(); 

# fixing the size of the figure 

plt.figure(figsize =(20, 20)) 

pos = nx.shell_layout(b1) 

node_color = [b1.degree(v) for v in b1] 

# node colour is a list of degrees of nodes 



node_size = 100

# size of node 

edge_width = [0.15 * b1[u][v]['weight'] for u, v in b1.edges()] 

# width of edge is a list of weight of edges 



nx.draw_networkx(b1,pos, node_size = node_size, 

                node_color = 'green', alpha = 0.7, 

                with_labels = True, width = edge_width, 

                edge_color ='.4', cmap = plt.cm.Blues) 



plt.axis('off') 

plt.tight_layout(); 

# fixing the size of the figure 

plt.figure(figsize =(20, 20)) 

pos = nx.kamada_kawai_layout(b1) 

node_color = [b1.degree(v) for v in b1] 

# node colour is a list of degrees of nodes 



node_size = 100

# size of node 

edge_width = [0.15 * b1[u][v]['weight'] for u, v in b1.edges()] 

# width of edge is a list of weight of edges 



nx.draw_networkx(b1,pos, node_size = node_size, 

                node_color = 'green', alpha = 0.7, 

                with_labels = True, width = edge_width, 

                edge_color ='.4', cmap = plt.cm.Blues) 



plt.axis('off') 

plt.tight_layout(); 

# fixing the size of the figure 

plt.figure(figsize =(20, 20)) 

pos = nx.spring_layout(b1) 

node_color = [b1.degree(v) for v in b1] 

# node colour is a list of degrees of nodes 



node_size = 100

# size of node 

edge_width = [0.15 * b1[u][v]['weight'] for u, v in b1.edges()] 

# width of edge is a list of weight of edges 



nx.draw_networkx(b1,pos, node_size = node_size, 

                node_color = 'green', alpha = 0.7, 

                with_labels = True, width = edge_width, 

                edge_color ='.4', cmap = plt.cm.Blues) 



plt.axis('off') 

plt.tight_layout(); 

# Creating a list of networks for all the books

books = [b1]
# Calculating the degree centrality of book 1

deg_cen_book1 = nx.degree_centrality(books[0])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book1 =  sorted(deg_cen_book1.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 10 of book1 and book5

for i in range(len(sorted_deg_cen_book1)):

    print(sorted_deg_cen_book1[i][0])# Printing out the top 5 of book1
book_fnames = ['/kaggle/input/book2.csv','/kaggle/input/book3.csv','/kaggle/input/book4.csv','/kaggle/input/book5.csv']

for book_fname in book_fnames:

    book = pd.read_csv(book_fname)

    G_book = nx.Graph()

    for _, edge in book.iterrows():

        G_book.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])

    books.append(G_book)
# Calculating the degree centrality of book 1

deg_cen_book1 = nx.degree_centrality(books[1])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book1 =  sorted(deg_cen_book1.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 5 of book1 and book5

for i in range(len(sorted_deg_cen_book1)):

    print(sorted_deg_cen_book1[i][0])# Printing out the top 5 of book1
# Calculating the degree centrality of book 1

deg_cen_book1 = nx.degree_centrality(books[2])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book1 =  sorted(deg_cen_book1.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 5 of book1 and book5

for i in range(len(sorted_deg_cen_book1)):

    print(sorted_deg_cen_book1[i][0])# Printing out the top 5 of book1
# Calculating the degree centrality of book 1

deg_cen_book1 = nx.degree_centrality(books[3])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book1 =  sorted(deg_cen_book1.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 5 of book1 and book5

for i in range(len(sorted_deg_cen_book1)):

    print(sorted_deg_cen_book1[i][0])# Printing out the top 5 of book1
# Calculating the degree centrality of book 1

deg_cen_book1 = nx.degree_centrality(books[4])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book1 =  sorted(deg_cen_book1.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 5 of book1 and book5

for i in range(len(sorted_deg_cen_book1)):

    print(sorted_deg_cen_book1[i][0])# Printing out the top 5 of book1
closeness=nx.closeness_centrality(b1)

sorted_clo_cen_book1=sorted(closeness.items(), key=lambda item: item[1],reverse=True)[:5]



# Printing out the top 5 of book1 

for i in range(len(sorted_clo_cen_book1)):

    print(sorted_clo_cen_book1[i][0])# Printing out the top 5 of book1
# Calculating the degree centrality of book 2

deg_cen_book2 = nx.closeness_centrality(books[1])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book2 =  sorted(deg_cen_book2.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 5 of book2

for i in range(len(sorted_deg_cen_book2)):

    print(sorted_deg_cen_book2[i][0])# Printing out the top 5 of book2
# Calculating the degree centrality of book 3

deg_cen_book3 = nx.closeness_centrality(books[2])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book3 =  sorted(deg_cen_book3.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 5 of book3

for i in range(len(sorted_deg_cen_book3)):

    print(sorted_deg_cen_book3[i][0])# Printing out the top 5 of book3
# Calculating the degree centrality of book 4

deg_cen_book4 = nx.closeness_centrality(books[3])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book4 =  sorted(deg_cen_book4.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 5 of book4

for i in range(len(sorted_deg_cen_book4)):

    print(sorted_deg_cen_book4[i][0])# Printing out the top 5 of book4
# Calculating the degree centrality of book 5

deg_cen_book5 = nx.closeness_centrality(books[4])



# Sorting the dictionaries according to their degree centrality and storing the top 5

sorted_deg_cen_book5 =  sorted(deg_cen_book5.items(), key=lambda x:x[1], reverse=True)[0:5]



# Printing out the top 5 of  book5

for i in range(len(sorted_deg_cen_book5)):

    print(sorted_deg_cen_book5[i][0])# Printing out the top 5 of book5
# print('ECENTRICITY')

# data=nx.eccentricity(books[0])

# book_ec=pd.DataFrame(data.items(),columns=['Characters','Shortest_Path'])

# book_ec.head(10)
print('ECENTRICITY')

data=nx.eccentricity(books[1])

book1_ec=pd.DataFrame(data.items(),columns=['Characters','Shortest_Path'])

book1_ec.head(10)
print("Diameter: ", nx.diameter(books[1])) 

print("Radius: ", nx.radius(books[1])) 

print("Preiphery: ", list(nx.periphery(books[1]))) 

print("Center: ", list(nx.center(books[1]))) 
print('ECENTRICITY')

data=nx.eccentricity(books[2])

book_ec=pd.DataFrame(data.items(),columns=['Characters','Shortest_Path'])

book_ec.head(10)
print("Diameter: ", nx.diameter(books[2])) 

print("Radius: ", nx.radius(books[2])) 

print("Preiphery: ", list(nx.periphery(books[2]))) 

print("Center: ", list(nx.center(books[2]))) 
print('ECENTRICITY')

data=nx.eccentricity(books[3])

book_ec=pd.DataFrame(data.items(),columns=['Characters','Shortest_Path'])

book_ec.head(10)
print("Diameter: ", nx.diameter(books[3])) 

print("Radius: ", nx.radius(books[3])) 

print("Preiphery: ", list(nx.periphery(books[3]))) 

print("Center: ", list(nx.center(books[3]))) 
print('ECENTRICITY')

data=nx.eccentricity(books[4])

book_ec=pd.DataFrame(data.items(),columns=['Characters','Shortest_Path'])

book_ec.head(10)
print("Diameter: ", nx.diameter(books[4])) 

print("Radius: ", nx.radius(books[4])) 

print("Preiphery: ", list(nx.periphery(books[4]))) 

print("Center: ", list(nx.center(books[4]))) 
%matplotlib inline



# Creating a list of closeness centrality of all the books

evolution = [nx.closeness_centrality(book) for book in books]

 

# Creating a DataFrame from the list of degree centralities in all the books

degree_evol_df = pd.DataFrame.from_records(evolution)



# Plotting the closeness centrality evolution

degree_evol_df[['Aegon-I-Targaryen', 'Tyrion-Lannister', 'Jon-Snow','Daenerys-Targaryen','Aemon-Targaryen-(Maester-Aemon)']].plot()
# Creating a list of pagerank of all the characters in all the books

evolution_pagerank = [nx.pagerank(book) for book in books]



# Making a DataFrame from the list

pagerank_evol_df = pd.DataFrame.from_records(evolution_pagerank)



# Finding the top 4 characters in every book

set_of_char = set()

for i in range(5):

    set_of_char |= set(list(pagerank_evol_df.T[i].sort_values(ascending=False)[0:5].index))

list_of_char = list(set_of_char)



# Plotting the top characters

pagerank_evol_df[list_of_char].plot(figsize=(20, 13))
# Creating a list of pagerank, betweenness centrality, degree centrality

# of all the characters in the fifth book.

import seaborn  as sns

measures = [nx.pagerank(books[4]), 

            nx.closeness_centrality(books[4]), 

            nx.degree_centrality(books[4])]



# Creating the correlation DataFrame

cor5 = pd.DataFrame.from_records(measures)



# Calculating the correlation

corr5=cor5.T.corr()

print(corr5)

ax = sns.heatmap(

    corr5, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=256),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=360,

    horizontalalignment='right'

);
# Creating a list of pagerank, betweenness centrality, degree centrality

# of all the characters in the fifth book.

import seaborn  as sns

measures = [nx.pagerank(books[0]), 

            nx.closeness_centrality(books[0]), 

            nx.degree_centrality(books[0])]



# Creating the correlation DataFrame

cor1 = pd.DataFrame.from_records(measures)





# Finding the most important character in the fifth book,  

# according to degree centrality, betweenness centrality and pagerank.

p_rank, c_cent, d_cent = cor1.idxmax(axis=1)



# Printing out the top character accoding to the Page Rank

print(p_rank)
# Creating a list of pagerank, betweenness centrality, degree centrality

# of all the characters in the fifth book.

import seaborn  as sns

measures = [nx.pagerank(books[1]), 

            nx.closeness_centrality(books[1]), 

            nx.degree_centrality(books[1])]



# Creating the correlation DataFrame

cor1 = pd.DataFrame.from_records(measures)





# Finding the most important character in the fifth book,  

# according to degree centrality, betweenness centrality and pagerank.

p_rank, c_cent, d_cent = cor1.idxmax(axis=1)



# Printing out the top character accoding to the Page Rank

print(p_rank)
# Creating a list of pagerank, betweenness centrality, degree centrality

# of all the characters in the fifth book.

import seaborn  as sns

measures = [nx.pagerank(books[2]), 

            nx.closeness_centrality(books[2]), 

            nx.degree_centrality(books[2])]



# Creating the correlation DataFrame

cor1 = pd.DataFrame.from_records(measures)





# Finding the most important character in the fifth book,  

# according to degree centrality, betweenness centrality and pagerank.

p_rank, c_cent, d_cent = cor1.idxmax(axis=1)



# Printing out the top character accoding to the Page Rank

print(p_rank)
# Creating a list of pagerank, betweenness centrality, degree centrality

# of all the characters in the fifth book.

import seaborn  as sns

measures = [nx.pagerank(books[3]), 

            nx.closeness_centrality(books[3]), 

            nx.degree_centrality(books[3])]



# Creating the correlation DataFrame

cor1 = pd.DataFrame.from_records(measures)





# Finding the most important character in the fifth book,  

# according to degree centrality, betweenness centrality and pagerank.

p_rank, c_cent, d_cent = cor1.idxmax(axis=1)



# Printing out the top character accoding to the Page Rank

print(p_rank)
# Finding the most important character in the fifth book,  

# according to degree centrality, betweenness centrality and pagerank.

p_rank, c_cent, d_cent = cor5.idxmax(axis=1)



# Printing out the top character accoding to the Page Rank

print(p_rank)