# Importing modules

import pandas as pd



# Reading in datasets/book1.csv

book1 = pd.read_csv('../input/book1.csv')



# Printing out the head of the dataset

book1.head()
# Importing modules

import networkx as nx



# Creating an empty graph object

G_book1 = nx.Graph()
# Iterating through the DataFrame to add edges

for _, edge in book1.iterrows():

    G_book1.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])





# Creating a list of networks for all the books

books = [G_book1]

book_fnames = ['../input/book2.csv', '../input/book3.csv', '../input/book4.csv', '../input/book5.csv']

for book_fname in book_fnames:

    book = pd.read_csv(book_fname)

    G_book = nx.Graph()

    for _, edge in book.iterrows():

        G_book.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])

    books.append(G_book)
# Make a simple drawing of graph

nx.draw(G_book1)
# Calculating the degree centrality of book 1

deg_cen_book1 = nx.degree_centrality(books[0])



# Calculating the degree centrality of book 5

deg_cen_book5 = nx.degree_centrality(books[4])



# Sorting the dictionaries according to their degree centrality and storing the top 10

sorted_deg_cen_book1 = sorted(deg_cen_book1.items(), key= lambda x: x[1], reverse=True)



# Sorting the dictionaries according to their degree centrality and storing the top 10

sorted_deg_cen_book5 = sorted(deg_cen_book5.items(), key= lambda x: x[1], reverse=True)



# Printing out the top 10 of book1 and book5

print('Top10 for book1:\n', sorted_deg_cen_book1[:10])

print('\nTop10 for book5:\n', sorted_deg_cen_book5[:10])
%matplotlib inline

import matplotlib.pyplot as plt



# Creating a list of degree centrality of all the books

evol = [nx.degree_centrality(book) for book in books]

 

# Creating a DataFrame from the list of degree centralities in all the books

degree_evol_df = pd.DataFrame(evol)



# Plotting the degree centrality evolution of Eddard-Stark, Tyrion-Lannister and Jon-Snow

plt.style.use('fivethirtyeight')

degree_evol_df[['Eddard-Stark', 'Tyrion-Lannister', 'Jon-Snow']].plot()

plt.show()


# Creating a list of betweenness centrality of all the books just like we did for degree centrality

evol = [nx.betweenness_centrality(book, weight='weight') for book in books]



# Making a DataFrame from the list

betweenness_evol_df = pd.DataFrame(evol).fillna(0)



# Finding the top 4 characters in every book

set_of_char = set()

for i in range(5):

    set_of_char = set(list(betweenness_evol_df.T[i].sort_values(ascending=False)[0:4].index))

list_of_char = list(set_of_char)



# Plotting the evolution of the top characters

betweenness_evol_df[list_of_char].plot(figsize=(13, 7))

plt.show()
# Creating a list of pagerank of all the characters in all the books

evol = [nx.pagerank(book) for book in books]



# Making a DataFrame from the list

pagerank_evol_df = pd.DataFrame(evol).fillna(0)

# Finding the top 4 characters in every book

set_of_char = set()

for i in range(5):

    set_of_char |= set(list(pagerank_evol_df.T[i].sort_values(ascending=False)[0:4].index))

list_of_char = list(set_of_char)



# Plotting the top characters

pagerank_evol_df[list_of_char].plot(figsize=(13, 7))



# Setting text font parameters to facilitate finding the 4 top characters

plt.text(4.05, 0.06, 'Jon Snow', size=20)

plt.text(4.05, 0.045, 'Daenerys', size=20)

plt.text(4.05, 0.034, 'Stannis', size=20)

plt.text(4.05, 0.029, 'Tyrion', size=20)

plt.text(4.05, 0.0035, 'Eddard Stark', size=20)

plt.text(0.2, 0.063, 'Eddard Stark', size=15)

plt.legend(loc=(1.2,0.3))

plt.show()
# Creating a list of pagerank, betweenness centrality, degree centrality

# of all the characters in the fifth book.

measures = [nx.pagerank(books[4]), 

            nx.betweenness_centrality(books[4], weight='weight'), 

            nx.degree_centrality(books[4])]



# Creating the correlation DataFrame

cor = pd.DataFrame(measures, index=['pageRank', 'betweeness', 'degree'])



# Calculating the correlation

cor.T.corr()
# Finding the most important character in the fifth book,  

# according to degree centrality, betweenness centrality and pagerank.

p_rank, b_cent, d_cent = cor.idxmax(axis=1)



# Format the results and print them

results = """Important character(s) according to:

\t- page rank is: {0}

\t- betweeness centrality is: {1} 

\t- degree centrality is: {2}"""



print(results.format(p_rank, b_cent, d_cent))