## importing necessary packges

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

try:

    !pip install tensorflow-gpu

    import tensorflow as tf

except:

    !pip install tensorflow

    import tensorflow as tf

def import_data():

    ## using ranking dataset

    ## and the books dataset

    dataset = pd.read_csv('../input/ratings.csv')

    books = pd.read_csv('../input/books.csv')

    return dataset,books

    
datasets,book=import_data()

book=book[['id','original_title','authors','isbn','original_publication_year']]

book.head()
def split_data(dataset):

    train,test = train_test_split(dataset)

    return train,test
train,test = split_data(datasets)
def extract_book_and_user(dataset):

    n_user = len(dataset.user_id.unique())

    n_books = len(dataset.book_id.unique())

    return n_user,n_books
n_user,n_books = extract_book_and_user(datasets)
print (n_user)

print (n_books)
def nural_net_model_function():

    from keras.layers import Input, Embedding, Flatten, Dot, Dense

    from keras.models import Model

    

    ## book input as a vector

    ## adding functional approach

    book_input = Input(shape=[1],name='Book-Input') ## adding lebel name

    ## adding embedding layer for onehot encoding

    book_embadding = Embedding(n_books+1,5,name='Bok-embadding')(book_input)

    #***

        #Embedding layer is used for making hot encoding like input

        ## for nural network and multiple input system

    #***

    

    ##adding flatten layer 

    book_vec = Flatten(name='Flatten-books')(book_embadding)

    

    

    ## adding user functionality

    user_input = Input(shape=[1], name="User-Input")

    ## adding embedding layers

    user_embedding = Embedding(n_user+1, 5, name="User-Embedding")(user_input)

    ## adding flatten layer

    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    

    ## making product for making a relation between two Embadding layer

    

    prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec]) ## relation between two Embedding

    

    x_train = [user_input, book_input]

    y_train = prod

    ## creating moddel

    model = Model(x_train,y_train)

    

    OPTIMIZER='adam'

    ERROR_FUNCTION='mean_squared_error'

    model.compile(OPTIMIZER,ERROR_FUNCTION)

    model.fit([train.user_id,train.book_id],train.rating,epochs=10,verbose=1,)

    return model

    

    

    

    

    
model = nural_net_model_function()
model.save('regresssion.model.h5') ## saving the whole structure and the weight
def get_unique_value():

    book_data = np.array(list(set(datasets.book_id)))

    return book_data
book_data = get_unique_value()
def setting_user(user_id):

    user = np.array([user_id for i in range(len(book_data))])

    return user
#user = int(input())  ## if you want you can ask for user id here

## seeting random value like

user = setting_user(1)
user
predictions = model.predict([user, book_data])

## extracting the value from multiple array

predictions = np.array([item[0] for item in predictions])
## getting the top 5 book id

def get_recommended_book_id(predictions):

    ## argsort() returns the sorted valus's previous index

    recommended_book_ids = (-predictions).argsort()[:5]

    return recommended_book_ids
recomended_book = get_recommended_book_id(predictions)
def get_recommended_book_name(book,recomended_book):

    books_index=book['id'].isin(recomended_book)

    value=book[books_index]

    return value

    
name = get_recommended_book_name(book,recomended_book)
name
recomended_book