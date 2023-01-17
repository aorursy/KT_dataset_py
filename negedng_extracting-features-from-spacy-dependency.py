import pandas as pd

import numpy as np

import keras



import spacy



from collections import defaultdict

from sklearn.metrics import log_loss





import matplotlib.pyplot as plt

%matplotlib inline



import time

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
nlp = spacy.load('en_core_web_lg')
class Graph():

    def __init__(self):

        """

        self.edges is a dict of all possible next nodes

        e.g. {'X': ['A', 'B', 'C', 'E'], ...}

        self.weights has all the weights between two nodes,

        with the two nodes as a tuple as the key

        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}

        """

        self.edges = defaultdict(list)

        self.weights = {}

    

    def add_edge(self, from_node, to_node, weight, back_penalty=1):

        # Note: assumes edges are bi-directional

        self.edges[from_node].append(to_node)

        self.edges[to_node].append(from_node)

        self.weights[(from_node, to_node)] = weight

        self.weights[(to_node, from_node)] = weight*back_penalty



def dijsktra(graph, initial, end):

    # shortest paths is a dict of nodes

    # whose value is a tuple of (previous node, weight)

    shortest_paths = {initial: (None, 0)}

    current_node = initial

    visited = set()

    

    while current_node != end:

        visited.add(current_node)

        destinations = graph.edges[current_node]

        weight_to_current_node = shortest_paths[current_node][1]



        for next_node in destinations:

            weight = graph.weights[(current_node, next_node)] + weight_to_current_node

            if next_node not in shortest_paths:

                shortest_paths[next_node] = (current_node, weight)

            else:

                current_shortest_weight = shortest_paths[next_node][1]

                if current_shortest_weight > weight:

                    shortest_paths[next_node] = (current_node, weight)

        

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}

        if not next_destinations:

            raise Exception("Something is wrong")

        # next node is the destination with the lowest weight

        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    

    # Work back through destinations in shortest path

    path = []

    dist = 0

    while current_node is not None:

        path.append(current_node)

        next_node = shortest_paths[current_node][0]

        dist += shortest_paths[current_node][1]

        current_node = next_node

    # Reverse path

    path = path[::-1]

    return path, dist

        
def get_rank(token):

    """Step up with token.head until it reaches the root. Returns with step number and root"""

    i = 0

    next_token = token

    while(next_token!=next_token.head):

        i+=1

        next_token=next_token.head

    return i, next_token



def child_count(token):

    cc = 0

    for child in token.children:

        cc+=1

    return cc
def build_answers(data):

    answers = []

    for i in range(len(data)):

        dataNext = data.loc[i]

        Acoref = dataNext["A-coref"]

        Bcoref = dataNext["B-coref"]

        answerNext = [int(Acoref), int(Bcoref), 1-int(Acoref or Bcoref)]

        answers.append(answerNext)

    return np.vstack(answers)
def build_features(data):

    """Generates features from input data"""

    features = []

    sum_good = 0

    for i in range(0,len(data)):

        fi = []

        dataNext = data.loc[i]

        text = dataNext["Text"]

        #print(visualise(dataNext))

        doc=nlp(text)

        Aoff = dataNext["A-offset"]

        Boff = dataNext["B-offset"]

        Poff = dataNext["Pronoun-offset"]

        lth = len(text)

        

        for token in doc:

            if(token.idx==Aoff):

                Atoken = token

            if(token.idx==Boff):

                Btoken = token

            if(token.idx==Poff):

                Ptoken=token

        Arank, Aroot = get_rank(Atoken)

        Brank, Broot = get_rank(Btoken)

        Prank, Proot = get_rank(Ptoken)

        

        graph = Graph()

        

        for token in doc:

            graph.add_edge(token, token.head, 1, 4)

        

        sent_root = []

        for sent in doc.sents:

            sent_root.append(sent.root)

        for j in range(len(sent_root)-1):

            graph.add_edge(sent_root[j], sent_root[j+1],1, 4)

        try:

            _, Alen = dijsktra(graph, Atoken, Ptoken)

        except:

            Alen = 300

        try:

            _, Blen = dijsktra(graph, Btoken, Ptoken)

        except:

            Blen = 300

        

        

        sent_num = len(sent_root)

        for i in range(len(sent_root)):

            if Aroot == sent_root[i]:

                Atop = i

            if Broot == sent_root[i]:

                Btop = i

            if Proot == sent_root[i]:

                Ptop = i

        

        fi.append(Aoff/lth)#0

        fi.append(Boff/lth)#1

        fi.append(Poff/lth)#2



        fi.append(1.0*Atop/sent_num)#3

        fi.append(1.0*Btop/sent_num)#4

        fi.append(1.0*Ptop/sent_num)#5



        fi.append(Arank/10)#6

        fi.append(Brank/10)#7

        fi.append(Prank/10)#8

        

        #fi.append(Atoken.similarity(Ptoken))#9

        #fi.append(Btoken.similarity(Ptoken))#10

        

        #fi.append(Alen/300)#9

        #fi.append(Blen/300)#10

        

        #fi.append(child_count(Aroot))#11

        #fi.append(child_count(Broot))#12

        #fi.append(child_count(Proot))#13

        

        features.append(fi)

    return np.vstack(features)



def swap_raws(data, i, j):

    """Swap the ith and jth column of the data"""

    new_data = np.copy(data)

    temp = np.copy(new_data[:, i])

    new_data[:,i] = new_data[:,j]

    new_data[:,j] = temp

    return new_data
def build_model(featnum):

    model = keras.models.Sequential()



    model.add(keras.layers.Dense(units=64, activation='relu', input_dim=featnum))

    model.add(keras.layers.Dense(units=64, activation='relu'))

    model.add(keras.layers.Dense(units=64, activation='relu'))



    

    model.add(keras.layers.Dense(units=3, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.02, decay=1e-4, momentum=0.7, nesterov=True)

    model.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])

    return model
!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv

!ls
test_data = pd.read_csv('gap-development.tsv', sep='\t')

val_data = pd.read_csv('gap-validation.tsv', sep='\t')

dev_data = pd.read_csv('gap-test.tsv', sep='\t')
dev_answ = build_answers(dev_data)

val_answ = build_answers(val_data)

test_answ = build_answers(test_data)
print("Feature building started ", time.ctime())

dev_feat = build_features(dev_data)

print("Developement ready", time.ctime())

val_feat = build_features(val_data)

print("Validation ready", time.ctime())

test_feat = build_features(test_data)

print("Test ready", time.ctime())
#Flip A and B in dev

dev_feat_p = swap_raws(dev_feat, 0, 1)

dev_feat_p = swap_raws(dev_feat_p, 3, 4)

dev_feat_p = swap_raws(dev_feat_p, 6, 7)

#dev_feat_p = swap_raws(dev_feat_p, 9, 10)

#dev_feat_p = swap_raws(dev_feat_p, 11, 12)



dev_feat = np.concatenate((dev_feat, dev_feat_p),0)
dev_answ_p = swap_raws(dev_answ,0,1)

dev_answ = np.concatenate((dev_answ, dev_answ_p),0)
model = build_model(np.shape(dev_feat)[1])

earlyStopping = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

history = model.fit(dev_feat, dev_answ,

                    validation_data = (val_feat, val_answ),

                    epochs=100000, batch_size=2000, verbose=0,

                    callbacks=[earlyStopping],

                    shuffle=True)
plt.plot(history.history['loss'], label="loss")

plt.plot(history.history['val_loss'], label="val_loss")

plt.legend()

plt.show()
test_predict = model.predict(test_feat)
print("Test score:", log_loss(test_answ,test_predict))
submission = pd.read_csv("../input/sample_submission_stage_1.csv", index_col = "ID")

submission["A"]=test_predict[:,0]

submission["B"]=test_predict[:,1]

submission["NEITHER"]=test_predict[:,2]

submission.to_csv("submission_11f.csv")