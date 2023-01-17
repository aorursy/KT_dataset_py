import random

import pandas as pd

import numpy as np



def chunks(l, n):

    n = max(1, n)

    return (l[i:i+n] for i in range(0, len(l), n))



def normalize(lst):

    s = sum(lst)

    normed = [itm/s for itm in lst]

    normed[-1] = (normed[-1] + (1-sum(normed)))#pad last value with what ever difference neeeded to make sum to exactly 1

    return normed



"""This routine develops the models vocabulary and vocab_probs is also built, representing the inverse probability 

of encounting a paticular subreddit in the given dataset, which is then used to bias the selection of rarer

subreddits as labels to 

smooth the distribution of training labels across all subreddits in the vocabulary"""



df = pd.read_csv("../input/reddit_data.csv").head(100000) #reduce dataset size to first 100,000 comments

vocab_counts = df["subreddit"].value_counts()

tmp_vocab = list(vocab_counts.keys())

total_counts = sum(vocab_counts.values)

inv_prob = [total_counts/vocab_counts[sub] for sub in tmp_vocab]

vocab = ["Unseen-Sub"] + tmp_vocab #build place holder, Unseen-Sub, for all subs not in vocab

tmp_vocab_probs = normalize(inv_prob)

#force probs sum to 1 by adding differenc to "Unseen-sub" probability

vocab_probs = [1-sum(tmp_vocab_probs)] + tmp_vocab_probs

print("Vocab size = " + str(len(vocab)))



#Subredit Sequencing

sequence_chunk_size = 15

def remove_repeating_subs(df):

    cache_data = {}

    prev_usr = None

    past_sub = None

    for comment_data in df.itertuples():

        current_usr = comment_data[1]

        if current_usr != prev_usr:#New user found in sorted comment data, begin sequence extraction for new user

            if prev_usr != None and prev_usr not in cache_data.keys():#dump sequences to cache for previous user if not in cache

                cache_data[prev_usr] = usr_sub_seq

            usr_sub_seq = [comment_data[2]] #initialize user sub sequence list with first sub for current user

            past_sub = comment_data[2]

        else:#if still iterating through the same user, add new sub to sequence if not a repeat

            if comment_data[2] != past_sub:#Check that next sub comment is not a repeat of the last interacted with sub,

                                            #filtering out repeated interactions

                usr_sub_seq.append(comment_data[2])

                past_sub = comment_data[2]

        prev_usr = current_usr #update previous user to being the current one before looping to next comment

    return cache_data



def build_training_sequences(usr_data):

    train_seqs = []

    #split user sub sequences into provided chunks of size sequence_chunk_size

    for usr,usr_sub_seq in usr_data.items():

        comment_chunks = chunks(usr_sub_seq,sequence_chunk_size)

        for chnk in comment_chunks:

            #for each chunk, filter out potential labels to select as training label, filter by the top subs filter list

            filtered_subs = [vocab.index(sub) for sub in chnk]

            if filtered_subs:

                #randomly select the label from filtered subs, using the vocab probability distribution to smooth out

                #representation of subreddit labels

                filter_probs = normalize([vocab_probs[sub_indx] for sub_indx in filtered_subs])

                label = np.random.choice(filtered_subs,1,p=filter_probs)[0]

                #build sequence by ensuring users sub exists in models vocabulary and filtering out the selected

                #label for this subreddit sequence

                chnk_seq = [vocab.index(sub) for sub in chnk if sub in vocab and vocab.index(sub) != label] 

                train_seqs.append([chnk_seq,label,len(chnk_seq)]) 

    return train_seqs
pp_user_data = remove_repeating_subs(df)

train_data = build_training_sequences(pp_user_data)

seqs,lbls,lngths = zip(*train_data)

train_df = pd.DataFrame({'sub_seqs':seqs,

                         'sub_label':lbls,

                         'seq_length':lngths})

train_df.head()
import tensorflow as tf

import tflearn

from tflearn.data_utils import to_categorical, pad_sequences

import numpy as np



def train_model(train,test,vocab_size,n_epoch=2,n_units=128,dropout=0.6,learning_rate=0.0001):



    trainX = train['sub_seqs']

    trainY = train['sub_label']

    testX =  test['sub_seqs']

    testY =  test['sub_label']



    # Sequence padding

    trainX = pad_sequences(trainX, maxlen=sequence_chunk_size, value=0.,padding='post')

    testX = pad_sequences(testX, maxlen=sequence_chunk_size, value=0.,padding='post')



    # Converting labels to binary vectors

    trainY = to_categorical(trainY, nb_classes=vocab_size)

    testY = to_categorical(testY, nb_classes=vocab_size)



    # Network building

    net = tflearn.input_data([None, 15])

    net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128,trainable=True)

    net = tflearn.lstm(net, n_units=n_units, dropout=dropout,weights_init=tflearn.initializations.xavier(),return_seq=False)

    net = tflearn.fully_connected(net, vocab_size, activation='softmax',weights_init=tflearn.initializations.xavier())

    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,

                             loss='categorical_crossentropy')



    # Training

    model = tflearn.DNN(net, tensorboard_verbose=2)



    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=False,

              batch_size=512,n_epoch=n_epoch)

    

    return model
split_perc=0.8

train_len, test_len = np.floor(len(train_df)*split_perc), np.floor(len(train_df)*(1-split_perc))

train, test = train_df.ix[:train_len-1], train_df.ix[train_len:train_len + test_len]

model = train_model(train,test,len(vocab))
from sklearn.manifold import TSNE

#retrieve the embedding layer fro mthe model by default name 'Embedding'

embedding = tflearn.get_layer_variables_by_name("Embedding")[0]

finalWs = model.get_weights(embedding)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

lowDWeights = tsne.fit_transform(finalWs)
from bokeh.plotting import figure, show, output_notebook,output_file

from bokeh.models import ColumnDataSource, LabelSet



#control the number of labelled subreddits to display

sparse_labels = [lbl if random.random() <=0.01 else '' for lbl in vocab]

source = ColumnDataSource({'x':lowDWeights[:,0],'y':lowDWeights[:,1],'labels':sparse_labels})





TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"



p = figure(tools=TOOLS)



p.scatter("x", "y", radius=0.1, fill_alpha=0.6,

          line_color=None,source=source)



labels = LabelSet(x="x", y="y", text="labels", y_offset=8,

                  text_font_size="10pt", text_color="#555555", text_align='center',

                 source=source)

p.add_layout(labels)



#output_file("embedding.html")

output_notebook()

show(p)
from tensorflow.python.framework import graph_util

def freeze_graph(model):

    # We precise the file fullname of our freezed graph

    output_graph = "/tmp/frozen_model.pb"



    # Before exporting our graph, we need to precise what is our output node

    # This is how TF decides what part of the Graph he has to keep and what part it can dump

    # NOTE: this variable is plural, because you can have multiple output nodes

    output_node_names = "InputData/X,FullyConnected/Softmax"



    # We clear devices to allow TensorFlow to control on which device it will load operations

    clear_devices = True

    

    # We import the meta graph and retrieve a Saver

    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)



    # We retrieve the protobuf graph definition

    graph = model.net.graph

    input_graph_def = graph.as_graph_def()



    # We start a session and restore the graph weights

    # We use a built-in TF helper to export variables to constants

    sess = model.session

    output_graph_def = graph_util.convert_variables_to_constants(

        sess, # The session is used to retrieve the weights

        input_graph_def, # The graph_def is used to retrieve the nodes 

        output_node_names.split(",") # The output node names are used to select the usefull nodes

    ) 



    # Finally we serialize and dump the output graph to the filesystem

    with tf.gfile.GFile(output_graph, "wb") as f:

        f.write(output_graph_def.SerializeToString())

    print("%d ops in the final graph." % len(output_graph_def.node))

freeze_graph(model)
def load_graph(frozen_graph_filename):

    # We load the protobuf file from the disk and parse it to retrieve the 

    # unserialized graph_def

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:

        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())



    # Then, we can use again a convenient built-in function to import a graph_def into the 

    # current default Graph

    with tf.Graph().as_default() as graph:

        tf.import_graph_def(

            graph_def, 

            input_map=None, 

            return_elements=None, 

            name="prefix", 

            op_dict=None, 

            producer_op_list=None

        )

    return graph

grph = load_graph("/tmp/frozen_model.pb")
x = grph.get_tensor_by_name('prefix/InputData/X:0')

y = grph.get_tensor_by_name("prefix/FullyConnected/Softmax:0")



# We launch a Session

with tf.Session(graph=grph) as sess:

    # Note: we didn't initialize/restore anything, everything is stored in the graph_def

    y_out = sess.run(y, feed_dict={

        x: [[1]*sequence_chunk_size] 

    })

    print(y_out) 