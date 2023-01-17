from keras.engine.saving import load_model

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.preprocessing.text import Tokenizer

from keras import layers

import pandas as pd

import numpy as np



import tensorflow as tf



config = tf.ConfigProto()

config.gpu_options.allow_growth = True

# Now when creating your session pass this config to it.

sess = tf.Session(config=config)









# csv_in = ["WellsFargoMobile - app reviews.csv","Discover Mobile - app reviews.csv","Citi Mobile - app reviews.csv","Chase Mobile - app reviews.csv", "Amex - app reviews.csv"]  # paths of CSVs to 'concentrate'

# csv_out = "banks_credit.csv"

csv_in = ["WellsFargoMobile - app reviews.csv","Discover Mobile - app reviews.csv","Citi Mobile - app reviews.csv","Chase Mobile - app reviews.csv", "Amex - app reviews.csv"]  # paths of CSVs to 'concentrate'

csv_out = "banks_credit.csv"



skip_header = False

with open(csv_out, "w") as dest:

    for csv in csv_in:

        with open(csv, "r") as src:

            if skip_header:  # skip the CSV header in consequent files

                next(src)

            for line in src:

                dest.write(line)

                if line[-1] != "\n":  # if not present, write a new line after each row

                    dest.write("\n")

            skip_header = True  # make sure only the first CSV header is included
# whats_new_data = pd.read_csv('/Users/jeffrey/Desktop/HugeProject/WhatsAppMessenger - app reviews.csv', sep=',')

# new file- create new data

banks_credit_all = pd.read_csv('/Users/jeffrey/Desktop/HugeProject/banks_credit.csv', sep=',')

banks_credit_all_sample=banks_credit_all.sample(n=1000,replace=False,random_state=1)

print (len(banks_credit_all_sample))

# new file- create new data

# whats_new_data.Body = whats_new_data.Body.astype(str)

banks_credit_all.Body = banks_credit_all.Body.astype(str)

# X_new = whats_new_data.Body.values

X_new=banks_credit_all.Body.values



np_ids = np.array(amex_new_data["Review.ID"])

np_ids = np.reshape(np_ids, [np_ids.__len__(), 1])

np_ids = np.squeeze(np_ids)



np.savetxt('/Users/jeffrey/Desktop/HugeProject/id_prediction.csv', np_ids, delimiter=',', fmt='%s')





def create_embedding_matrix(filepath, word_index, embedding_dim):

    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index

    embedding_matrix = np.zeros((vocab_size, embedding_dim))



    with open(filepath) as f:

        for line in f:

            word,*vector = line.split()

            if word in word_index:

                idx = word_index[word]

                embedding_matrix[idx] = np.array(

                    vector, dtype=np.float32)[:embedding_dim]



    return embedding_matrix



# Embedding- represents wods as dense word bector

def create_model(vocab_size, embedding_dim, maxlen,embedding_matrix):

    model = Sequential()

    model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen,trainable=True))

    model.add(layers.Conv1D(128, 5, activation='relu'))

    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(10, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    return model







def grid_create_model(num_filters, kernel_size,vocab_size, embedding_dim, maxlen):

    model = Sequential()

    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True))

    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))

    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(10, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    return model
#  embedding_dim = 50  # 100, 200, 300

def run_model_configuration(embedding_dim,train_size):

    # input data for training



    updatedall_data=pd.read_csv('/Users/jeffrey/Desktop/HugeProject/combined training data v2.csv', sep=',')

#     print (updatedall_data.head())

    updatedall_data_sample=updatedall_data.sample(n=train_size,replace=False,random_state=1)



    sentences = updatedall_data_sample['body'].values





    paymentlabels = updatedall_data_sample[['reviewid','payment']]

#     tokenizer_obj = Tokenizer()

#     sentences=[str(i) for i in sentences]

#     tokenizer_obj.fit_on_texts(sentences)





# # sentences

# label_list = pd.read_csv('/Users/glennartz/Documents/UX Classifier/Training Data/single topics/label list v1.csv', sep=',')

# label list is every category parameter



    # set model parameters

    test_size = .2

    num_words = 5000

    maxlen = 100

    batch_size = 200

    num_epochs = 2

    embedding_dim=embedding_dim



# for i in range(len(paymentlabels)):

#     y = review_data[paymentlabels.Payment[i]].values

#     print (y)

    y=paymentlabels.payment

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=test_size, random_state=1000)



    tokenizer_obj = Tokenizer()

    sentences=[str(i) for i in sentences]

    tokenizer_obj.fit_on_texts(sentences)

    embedding_matrix = create_embedding_matrix('/Users/jeffrey/Desktop/HugeProject/glove.6B.'+str(embedding_dim)+'d.txt', tokenizer_obj.word_index, embedding_dim)

    

    # pad sequences

    max_length = max([len(s.split()) for s in sentences])

    

    # define vocabulary size

    vocab_size = len(tokenizer_obj.word_index) + 1

    print (vocab_size)

    

    sentences_train=[str(i) for i in sentences_train]

    sentences_test=[str(i) for i in sentences_test]

    

    X_train_tokens =  tokenizer_obj.texts_to_sequences(sentences_train)

    X_test_tokens = tokenizer_obj.texts_to_sequences(sentences_test)

    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')

    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')



    print (embedding_dim)

    model = create_model(vocab_size, embedding_dim, max_length,embedding_matrix)

    model.fit(X_train_pad, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test_pad, y_test), verbose=True)

    score, acc = model.evaluate(X_test_pad, y_test, batch_size=batch_size)

    print('Test score:', score)

    print('Test accuracy:', acc)

    print("Accuracy: {0:.2%}".format(acc))

    #     model.save('/Users/glennartz/Documents/UX Classifier/Saved models/' + label_list.label[i] + '_model_ulta.h5')  # creates a HDF5 file 'my_model.h5'

    # model = load_model('/Users/glennartz/Documents/Project Z/saved_models/' + label_list.label[i] + '_model.h5')





    test_samples_tokens = tokenizer_obj.texts_to_sequences(X_new)

    test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)



    #predict

    predictions = model.predict(x=test_samples_tokens_pad)



    np_predictions = np.array(predictions)

    np_predictions = np.squeeze(np_predictions)





    # pred_data = pd.DataFrame({'Review.ID': np_ids, 'labels': np_classes, 'pred': np_predictions})

    pred_data = pd.DataFrame({'Review.ID': np_ids, 'pred': np_predictions})



    np.savetxt('/Users/jeffrey/Desktop/HugeProject/' + "bank" + '_predictions.csv', pred_data, delimiter=',', fmt='%s')



    del model  # deletes the existing model

























run_model_configuration(50,1000)
# wells_review_data = pd.read_csv('/Users/jeffrey/Desktop/HugeProject/WellsFargoMobile - app reviews.csv', sep=',')

updatedall_data=pd.read_csv('/Users/jeffrey/Desktop/HugeProject/combined training data v2.csv', sep=',')

# banks_credit=pd.read_csv('/Users/jeffrey/Desktop/HugeProject/banks_credit.csv', sep=',')

updatedall_data_sample=updatedall_data.sample(n=1000,replace=False,random_state=1)







# add embedding weights?
print (len(banks_credit))
# input data for training

all_data.head()

# sentences1 = wells_review_data['Body'].values

sentences = banks_credit_sample['Body'].values

# # paymentlabels=wells_review_data['Payment'].values

# # wells_review_data.head()



paymentlabels = banks_credit_sample[['Review.ID','Payment']]

# # sentences



print (len(sentences))

print (len(sentences1))

# print(paymentlabels)
whats_new_data = pd.read_csv('/Users/jeffrey/Desktop/HugeProject/WhatsAppMessenger - app reviews.csv', sep=',')

# new file- create new data

amex_new_data = pd.read_csv('/Users/jeffrey/Desktop/HugeProject/Amex - app reviews.csv', sep=',')

# new file- create new data

# whats_new_data.Body = whats_new_data.Body.astype(str)

amex_new_data.Body = amex_new_data.Body.astype(str)

# X_new = whats_new_data.Body.values

X_new=amex_new_data.Body.values



np_ids = np.array(amex_new_data["Review.ID"])

np_ids = np.reshape(np_ids, [np_ids.__len__(), 1])

np_ids = np.squeeze(np_ids)



np.savetxt('/Users/jeffrey/Desktop/HugeProject/id_prediction.csv', np_ids, delimiter=',', fmt='%s')



embedding_dim = 50 

tokenizer_obj = Tokenizer()

sentences=[str(i) for i in sentences]

tokenizer_obj.fit_on_texts(sentences)



embedding_matrix = create_embedding_matrix('/Users/jeffrey/Desktop/HugeProject/glove.6B.'+str(embedding_dim)+'d.txt', tokenizer_obj.word_index, embedding_dim)



embedding_matrix
# label_list = pd.read_csv('/Users/glennartz/Documents/UX Classifier/Training Data/single topics/label list v1.csv', sep=',')

# label list is every category parameter



# set model parameters

test_size = .2

num_words = 5000

embedding_dim = 50  # 100, 200, 300

maxlen = 100

batch_size = 200

num_epochs = 2



# what is new category in this case

# new data are category

# whats_new_data = pd.read_csv('/Users/jeffrey/Desktop/HugeProject/WhatsAppMessenger - app reviews.csv', sep=',')

# new file- create new data

# whats_new_data.body = new_data.body.astype(str)

# X_new = whats_new_data.body.values



# np_ids = np.array(whats_new_data.reviewid)

# np_ids = np.reshape(np_ids, [np_ids.__len__(), 1])

# np_ids = np.squeeze(np_ids)



# np.savetxt('/Users/glennartz/Documents/UX Classifier/Predictions/ids_predictions.csv', np_ids, delimiter=',', fmt='%s')



# for i in range(len(paymentlabels)):

#     y = review_data[paymentlabels.Payment[i]].values

#     print (y)

y=paymentlabels.Payment

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=test_size, random_state=1000)



tokenizer_obj = Tokenizer()

sentences=[str(i) for i in sentences]

tokenizer_obj.fit_on_texts(sentences)



embedding_matrix = create_embedding_matrix('/Users/jeffrey/Desktop/HugeProject/glove.6B.'+str(embedding_dim)+'d.txt', tokenizer_obj.word_index, embedding_dim)



# pad sequences

max_length = max([len(s.split()) for s in sentences])



# define vocabulary size

vocab_size = len(tokenizer_obj.word_index) + 1



sentences_train=[str(i) for i in sentences_train]

sentences_test=[str(i) for i in sentences_test]

X_train_tokens =  tokenizer_obj.texts_to_sequences(sentences_train)

X_test_tokens = tokenizer_obj.texts_to_sequences(sentences_test)





X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')

X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')





model = create_model(vocab_size, embedding_dim, max_length)

model.fit(X_train_pad, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test_pad, y_test), verbose=True)

score, acc = model.evaluate(X_test_pad, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)

print("Accuracy: {0:.2%}".format(acc))

#     model.save('/Users/glennartz/Documents/UX Classifier/Saved models/' + label_list.label[i] + '_model_ulta.h5')  # creates a HDF5 file 'my_model.h5'

# model = load_model('/Users/glennartz/Documents/Project Z/saved_models/' + label_list.label[i] + '_model.h5')





test_samples_tokens = tokenizer_obj.texts_to_sequences(X_new)

test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)



#predict

predictions = model.predict(x=test_samples_tokens_pad)



np_predictions = np.array(predictions)

np_predictions = np.squeeze(np_predictions)





# pred_data = pd.DataFrame({'Review.ID': np_ids, 'labels': np_classes, 'pred': np_predictions})

pred_data = pd.DataFrame({'Review.ID': np_ids, 'pred': np_predictions})



np.savetxt('/Users/jeffrey/Desktop/HugeProject/' + "bank" + '_predictions.csv', pred_data, delimiter=',', fmt='%s')



del model  # deletes the existing model























# figure grid search out

# Group by similar topics

# keep track of sample sizes and run models 

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import RandomizedSearchCV



# Main settings

epochs = 20

# amount of time the model takes it goes through the whole training- 1000 rows batch size is 200 5 iterations

embedding_dim = 50

maxlen = 100

# reset embedding dim, maxlen

output_file = 'data/output.txt'



# Run grid search for each source (yelp, amazon, imdb)

# for source, frame in df.groupby('source'):

# print('Running grid search for data set :', source)

# sentences = all_data['sentence'].values

sentences = all_data['Body'].values

sentences=[str(i) for i in sentences]

# y = df['label'].values

y=paymentlabels.Payment

# Train-test split

sentences_train, sentences_test, y_train, y_test = train_test_split(

    sentences, y, test_size=0.25, random_state=1000)







sentences_train=[str(i) for i in sentences_train]

sentences_test=[str(i) for i in sentences_test]

# Tokenize words

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)

X_test = tokenizer.texts_to_sequences(sentences_test)



# Adding 1 because of reserved 0 index

vocab_size = len(tokenizer.word_index) + 1



# Pad sequences with zeros

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



# Parameter grid for grid search

param_grid = dict(num_filters=[32, 64, 128],

                  kernel_size=[3, 5, 7],

                  vocab_size=[vocab_size],

                  embedding_dim=[embedding_dim],

                  maxlen=[maxlen])

model = KerasClassifier(build_fn=grid_create_model,

                        epochs=epochs, batch_size=10,

                        verbose=False)

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,

                          cv=4, verbose=1, n_iter=5)



print (X_train.shape,y_train.shape)

grid_result = grid.fit(X_train, y_train)



# Evaluate testing set

test_accuracy = grid.score(X_test, y_test)



# Save and evaluate results

prompt = input(f'finished {source}; write to file and proceed? [y/n]')

# if prompt.lower() not in {'y', 'true', 'yes'}:

#     break

with open(output_file, 'a') as f:

    s = ('Running {} data set\nBest Accuracy : '

         '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')

    output_string = s.format(

        source,

        grid_result.best_score_,

        grid_result.best_params_,

        test_accuracy)

    print(output_string)

    f.write(output_string)