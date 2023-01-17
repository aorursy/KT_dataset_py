! pip install turicreate

import turicreate
# Reading the data and creating an SFrame of the data

people = turicreate.SFrame.read_csv('../input/documentreterival/people_wiki.csv')



# Exploring dataset

people.head()
people.shape
obama = people[people['name'] == 'Barack Obama']

obama['text']
BradPitt = people[people['name'] == 'Brad Pitt'] 

BradPitt['text']
# Creating new 'word_count' column for obama article

obama['word_count'] = turicreate.text_analytics.count_words(obama['text'])

obama
# let's view some words count in 'word_count' column for obama

obama['word_count'][:30]
# Creating a table containg two columns 'word' and 'count' for obama using its 'word_count' column

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name= ['word', 'count'])

obama_word_count_table.head(5)
obama_word_count_table.sort('count', ascending= False)
# Creating 'word_count' column for for entire corpus of article (i.e. for all rows 'text' column in the dataframe)

people['word_count'] = turicreate.text_analytics.count_words(people['text'])

people.head(5)
# Computing 'tfidf' (i.e 'Term Frequency-Inverse Document Frequecy') using 'text' column for dataset 

people['tfidf'] = turicreate.text_analytics.tf_idf(people['text'])

people.head(5)
obama = people[people['name'] == 'Barack Obama']

obama[['tfidf']].stack('tfidf', new_column_name= ['word', 'count']).sort('count', ascending= False)
BradPitt = people[people['name'] == 'Brad Pitt']

BradPitt[['tfidf']].stack('tfidf', new_column_name= ['word', 'count']).sort('count', ascending= False)
clinton = people[people['name'] == 'Bill Clinton']

beckham = people[people['name'] == 'David Beckham']
turicreate.distances.cosine(obama['tfidf'][0], clinton['tfidf'][0])
turicreate.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])
knn_model = turicreate.nearest_neighbors.create(people, features= ['tfidf'], label= 'name')
knn_model.query(obama)
EltonJohn = people[people['name'] == 'Elton John']

EltonJohn
# top five word according to word counts for 'Elton John'

EltonJohn_word_count_table = EltonJohn[['word_count']].stack('word_count', new_column_name= ['word', 'count'])

EltonJohn_word_count_table.sort('count', ascending= False).head(5)
# top five word according to tfidf for 'Elton John'

EltonJohn_tfidf_table = EltonJohn[['tfidf']].stack('tfidf', new_column_name= ['word', 'tfidf'])

EltonJohn_tfidf_table.sort('tfidf', ascending= False).head(5)
victoria = people[people['name'] == 'Victoria Beckham']

paul = people[people['name'] == 'Paul McCartney']
# Cosine distance between 'Elton John' and 'Victoria Beckham'

turicreate.distances.cosine(EltonJohn['tfidf'][0], victoria['tfidf'][0])
# Cosine distance between 'Elton John' and 'Paul McCartney'

turicreate.distances.cosine(EltonJohn['tfidf'][0], paul['tfidf'][0])
knn_model.query(EltonJohn)