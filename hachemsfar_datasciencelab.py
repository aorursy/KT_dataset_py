# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pickle

from sklearn.manifold import TSNE

import os

import numpy

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import pandas as pd

import seaborn as sns





import math



import numpy

from sklearn.metrics import f1_score, accuracy_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.linear_model import LogisticRegression



import torch

from torch.autograd import Variable

import torch.optim as optim

import random

import numpy as np

import nltk

import time

import networkx as nx



import torch

import torch.nn as nn

import numpy as np

import random
# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
final_loss_list=[]





class LossNegSampling(nn.Module):

    

    def __init__(self, num_nodes, emb_dim,nb_labels, sequence_length, context_size, no_of_sequences_per_node):

        

        super(LossNegSampling, self).__init__()



        self.V=num_nodes

        self.dim = emb_dim

        self.t=1

        self.gamma_o = 0.01 # Initial Clustering Weight Rate 0.1, 0.01, 0.001

        self.gamma = 0.01

        self.l = sequence_length

        self.w = context_size

        self.N = no_of_sequences_per_node



        self.embedding_u = nn.Embedding(num_nodes, emb_dim) #  embedding  u

        self.embedding_com = nn.Embedding(nb_labels, emb_dim) #  embedding  community centers

       

        self.logsigmoid = nn.LogSigmoid()

    

        initrange = (2.0 / (num_nodes + emb_dim))**0.5 # Xavier init 2.0/sqrt(num_nodes+emb_dim)

        self.embedding_u.weight.data.uniform_(-initrange, initrange) # init u

            

        self.nb_labels= nb_labels

        self.lr_o = 0.01 # Initial Learning Rate 0.01, 0.005

        self.lr_f = 0.001 # 0.001 0.0005

        self.lr = 0.01

  

        inits=[]

        for k in range(nb_labels):

            rnd_node=torch.tensor([random.randint(0,num_nodes-1)])

            vec=self.embedding_u(rnd_node).data.cpu().numpy()[0]

            inits.append(vec)



        self.embedding_com.weight.data.copy_(torch.from_numpy(np.array(inits))) ##init_communities

        

        #for i in range(1,5):

        #    print(i, self.embedding_com.weight[i])

    def calculate_node_labels(self):



        u_embed = self.embedding_u.weight.data

        c_embed = self.embedding_com.weight.data



        n = u_embed.shape[0]

        d = u_embed.shape[1]        



        k = c_embed.shape[0]



        z = u_embed.reshape(n,1,d)

        z = z.repeat(1,k,1)   

     

        mu = c_embed.reshape(1,k,d)

        mu = mu.repeat(n,1,1)

        

        dist = (z-mu).norm(2,dim=2).reshape((n,k))



        cluster_choice=torch.argmin(dist,dim=1)



        return cluster_choice



    def calculate_new_cluster_centers(self, node_labels):

        nodes_per_cluster = {}

        for i in range(self.nb_labels):

            nodes_per_cluster[i]=[]



        for i in range(len(node_labels)):

            a = int(node_labels[i])

            nodes_per_cluster[a].append(self.embedding_u.weight.data[i])



        for i in range(self.nb_labels):

            npc = nodes_per_cluster[i]

            w=0

            for j in range(len(npc)):

                w+=npc[j]

            w=w/len(npc)

            if type(w)!=int:

                self.embedding_com.weight.data[i]=w





        

    def forward(self, u_node, v_node, negative_nodes):

        # self.t=self.t+1





        # self.t+=len(u_node)

        # self.gamma=self.gamma_o*(10**((-self.t*math.log10(self.gamma_o))/(self.l*self.w*self.V*self.N)))           

        # self.lr = self.lr_o - ((self.lr_o-self.lr_f)*(self.t/(self.l*self.w*self.V*self.N))) 



        u_embed = self.embedding_u(u_node) # B x 1 x Dim  edge (u,v)

        v_embed = self.embedding_u(v_node) # B x 1 x Dim  

                           

        negs = -self.embedding_u(negative_nodes) # B x K x Dim  neg samples

     

        positive_score=  v_embed.bmm(u_embed.transpose(1, 2)).squeeze(2) # Bx1

        negative_score= torch.sum(negs.bmm(u_embed.transpose(1, 2)).squeeze(2), 1).view(negative_nodes.size(0), -1) # BxK -> Bx1

             

        sum_all = self.logsigmoid(positive_score)+ self.logsigmoid(negative_score)

            

        loss= -torch.mean(sum_all)





        c_embed = self.embedding_com.weight.data



        n = u_embed.shape[0]

        d = u_embed.shape[2]

        # z = u_embed.repeat(1,self.nb_labels,1)



        k = c_embed.shape[0]



        z = u_embed.reshape(n,1,d)

        z = z.repeat(1,k,1)   

     

        mu = c_embed.reshape(1,k,d)

        mu = mu.repeat(n,1,1)



        # mu = self.embedding_com.weight.repeat(n,1,1)

        

        dist = (z-mu).norm(2,dim=2).reshape((n,k))



        loss2= self.logsigmoid((dist.min(dim=1)[0]**2)).mean()

        # loss2= (dist.min(dim=1)[0]**2).mean()



        # cluster_choice=torch.argmin(dist,dim=1)



        final_loss=loss+(loss2*self.gamma)



        # return final_loss, cluster_choice

        return final_loss



    

    def get_emb(self, input_node):

        

        embeds = self.embedding_u(input_node) ### u



        return embeds
torch.cuda.is_available()

gpus = [0]

torch.cuda.set_device(gpus[0])
USE_CUDA = True # torch.cuda.is_available()

#gpus = [0]

#torch.cuda.set_device(gpus[0])

    

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

        

class lineEmb():

    

    def __init__(self, edge_file, social_edges=None, name='wiki', emb_size= 2,  

                     alpha=5, epoch=5, batch_size= 256, shuffel=True , neg_samples=5,

                      sequence_length=21, context_size=10, no_of_sequences_per_node=15):

    

        self.emb_size = emb_size

        self.shuffel = shuffel

    

        self.neg_samples = neg_samples

        self.batch_size=batch_size

        self.epoch=epoch

        self.alpha= alpha

     

        self.name=name

        self.G = nx.read_edgelist(edge_file)

        self.social_edges= social_edges



        self.sequence_length=sequence_length

        self.context_size=context_size

        self.no_of_sequences_per_node=no_of_sequences_per_node

        

        self.index2word = dict()

        self.word2index = dict()

        self.build_vocab()  

        

        self.emb_file= './emb/%s_size_%d_line.emb'%(self.name, self.emb_size)  

                 

    def getBatch(self, batch_size, train_data):

        

        if self.shuffel==True:

            random.shuffle(train_data)

        

        sindex = 0

        eindex = batch_size

        while eindex < len(train_data):

            batch = train_data[sindex: eindex]

            temp = eindex

            eindex = eindex + batch_size

            sindex = temp

            yield batch

        

        if eindex >= len(train_data):

            batch = train_data[sindex:]

            

            yield batch

                

    def prepare_node(self, node, word2index):

        return Variable(LongTensor([word2index[str(node)]]))

    

             

    def prepare_sequence(self, seq, word2index):

        idxs = list(map(lambda w: word2index[w], seq))



        return Variable(LongTensor(idxs))

    

    def build_vocab(self):

        self.social_nodes=[]



        for u,v in self.social_edges:

            self.social_nodes.append(u)

            self.social_nodes.append(v)   

            

        self.all_nodes= list(set(self.social_nodes))

        

        self.word2index = {}

        for vo in self.all_nodes:

            if self.word2index.get(vo) is None:

                self.word2index[str(vo)] = len(self.word2index)



        self.index2word = {v:k for k, v in self.word2index.items()}

               

    def prepare_trainData(self, sequences):    

        print('prepare training data ...')

        self.train_data = []

        for sequence in sequences:

            for i in range(self.context_size * 2 + 1):

                if i != self.context_size:

                    self.train_data.append((sequence[self.context_size], sequence[i]))

        u_p = []

        v_p = []

        tr_num=0    

        for tr in self.train_data:

            u_p.append(self.prepare_node(tr[0], self.word2index).view(1, -1))

            v_p.append(self.prepare_node(tr[1], self.word2index).view(1, -1))

            tr_num+=1

        train_samples = list(zip(u_p, v_p))

        print(len(train_samples), 'samples are ready ...')

        return train_samples

        



    def negative_sampling(self, targets,  k):

        batch_size = targets.size(0)

        neg_samples = []

         

        for i in range(batch_size):

             

            nsample = []

            target_index = targets[i].data.cpu().tolist()[0] if USE_CUDA else targets[i].data.tolist()[0]

            v_node= self.index2word[target_index]

            

            while len(nsample) < k: # num of sampling

                

                neg = random.choice(self.all_nodes)

                if (neg != v_node): 

                    nsample.append(neg)

                else:   

                    continue

               

            neg_samples.append(self.prepare_sequence(nsample, self.word2index).view(1, -1))

        

        return torch.cat(neg_samples)        





    def random_walk_sample(self, no_of_sequences_per_node, sequence_length):

        walks = []

        half_walk = int(sequence_length/2)

        for node in self.all_nodes:

            for i in range(no_of_sequences_per_node):

                node_sequence = []

                walk_1 = self.capture_sequence(node_sequence, node, half_walk)

                walk_1.reverse()

                node_sequence = []

                walk_2 = self.capture_sequence(node_sequence, node, half_walk)

                total_random_walk = walk_1+[node]+walk_2

                walks.append(total_random_walk)

        # random.shuffle(walks)

        # return walks

        flatten = lambda list: [item for sublist in list for item in sublist]

        windows = flatten([list(nltk.ngrams(c, self.context_size * 2 + 1)) for c in walks])

        #random.shuffle(windows)

        return windows



    #Recursive Function 

    def capture_sequence(self, walk, node, counter):

        if counter==0:

            return walk

        else:

            counter-=1

            connected_nodes = list(self.G[node])

            random_neighbor_node = random.randint(0,len(connected_nodes)-1)

            next_node=connected_nodes[random_neighbor_node]

            walk.append(next_node)

            return self.capture_sequence(walk, next_node, counter)

    



    def train (self,nb_labels):

        

        train_data= self.prepare_trainData(self.random_walk_sample(self.no_of_sequences_per_node, self.sequence_length))

        

        final_losses = []

        model = LossNegSampling(len(self.all_nodes), self.emb_size, nb_labels,

         self.sequence_length, self.context_size, self.no_of_sequences_per_node)

        

        if USE_CUDA:

           model = model.cuda()

           

        optimizer = optim.Adam(model.parameters(), lr=model.lr) #Learning Rate changed dynamically 

       

        self.epoches=[]

                

        for epoch in range(self.epoch):

            

            t1=time.time()           



            f = open("gamma_values.txt", "a")

            f.write(str(model.gamma)+"\n")

            f.close()

            f = open("alpha_values.txt", "a")

            f.write(str(model.lr)+"\n")

            f.close()



            for i,  batch in enumerate(self.getBatch(self.batch_size, train_data)):

            

                inputs, targets= zip(*batch)



                model.t+=len(inputs)

                model.gamma=model.gamma_o*(10**((-model.t*math.log10(model.gamma_o))/(model.l*model.w*model.V*model.N)))           

                model.lr = model.lr_o - ((model.lr_o-model.lr_f)*(model.t/(model.l*model.w*model.V*model.N))) 

                # The changing of the learning rate

                for param in optimizer.param_groups:

                    param['lr'] = model.lr

               

                inputs= torch.cat(inputs) # B x 1

                targets=torch.cat( targets) # B x 1

    

                negs = self.negative_sampling(targets , self.neg_samples)

    

                model.zero_grad()



                node_labels = model.calculate_node_labels()

                model.calculate_new_cluster_centers(node_labels)



                final_loss  = model(inputs, targets, negs)

                final_loss.backward()

                optimizer.step()



                final_losses.append(final_loss.data.cpu().numpy())



            t2= time.time()

            final_loss_list.append(np.mean(final_losses))

            print(self.name, ' Epoch Number: ', epoch,' loss: %0.3f '%np.mean(final_losses),' Alpha: ', model.lr,' Gamma: ', model.gamma ,' t: ', model.t,' l: ', model.l,' w: ', model.w,' N: ', model.N)                                               



            self.validate(model)



        final_emb={}

        normal_emb={}

        for w in self.all_nodes:



            normal_emb[w]=model.get_emb(self.prepare_node(w, self.word2index))

            vec=[float(i) for i in normal_emb[w].data.cpu().numpy()[0]]



            final_emb[int(w)]=vec

        

        return final_emb



    def validate(self, model):

        final_emb={}

        normal_emb={}

        for w in self.all_nodes:

            normal_emb[w]=model.get_emb(self.prepare_node(w, self.word2index))

            vec=[float(i) for i in normal_emb[w].data.cpu().numpy()[0]]

            final_emb[int(w)]=vec

        node_classification(final_emb, "/kaggle/input/data-science-lab-data-sets/cora-label.txt", "cora_GEMSEC", 128)

class TopKRanker(OneVsRestClassifier):



    def predict(self, X, top_k_list):

        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))

        all_labels = []

        for i, k in enumerate(top_k_list):

            probs_ = probs[i, :]

            labels = self.classes_[probs_.argsort()[-k:]].tolist()

            probs_[:] = 0

            probs_[labels] = 1

            all_labels.append(probs_)

        return numpy.asarray(all_labels)





class Classifier(object):



    def __init__(self, embeddings, clf, name):

        self.embeddings = embeddings

        self.clf = TopKRanker(clf)

        self.binarizer = MultiLabelBinarizer(sparse_output=True)

        self.name = name



    def train(self, X, Y, Y_all):

        self.binarizer.fit(Y_all)

        X_train = [self.embeddings[x] for x in X]

        Y = self.binarizer.transform(Y)

        self.clf.fit(X_train, Y)



    def evaluate(self, X, Y):

        top_k_list = [len(l) for l in Y]

        Y_ = self.predict(X, top_k_list)

        Y = self.binarizer.transform(Y)



        averages = ["micro", "macro", "samples", "weighted"]

        results = {}

        for average in averages:

            results[average] = f1_score(Y, Y_, average=average)



        results['acc'] = accuracy_score(Y, Y_)

        return results



    def predict(self, X, top_k_list):

        X_ = numpy.asarray([self.embeddings[x] for x in X])

        Y = self.clf.predict(X_, top_k_list=top_k_list)

        return Y



    def split_train_evaluate(self, X, Y, train_precent, seed=0):

        state = numpy.random.get_state()



        training_size = int(train_precent * len(X))

        numpy.random.seed(seed)

        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))

        X_train = [X[shuffle_indices[i]] for i in range(training_size)]

        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]

        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]

        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]



        self.train(X_train, Y_train, Y)

        numpy.random.set_state(state)

        return self.evaluate(X_test, Y_test)
def read_node_label(embeddings, label_file, skip_head=False):

        fin = open(label_file, 'r')

        X = []

        Y = []

        label = {}



        for line in fin:

            a = line.strip('\n').split(' ')

            label[a[0]] = a[1]



        fin.close()

        for i in embeddings:

            X.append(i)

            Y.append(label[str(i)])



        return X, Y





def node_classification(embeddings, label_path,name, size):

        X, Y = read_node_label(embeddings, label_path )



        f_c = open('%s_classification_%d.txt' % (name, size), 'w')



        all_ratio = []



        for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

            # print(" Training classifier using {:.2f}% nodes...".format(tr_frac * 100))

            clf = Classifier(embeddings=embeddings, clf=LogisticRegression(), name=name)

            results = clf.split_train_evaluate(X, Y, tr_frac)

            avg = 'macro'

            f_c.write(name + ' train percentage: ' + str(tr_frac) + ' F1-' + avg + ' ' + str('%0.5f' % results[avg]))

            all_ratio.append(results[avg])

            f_c.write('\n')

        f_c = open('%s_classification_%d.txt' % (name, size), 'r')

        print(f_c.read()) 





def plot_embeddings(embeddings,label_file,name):

    X, Y = read_node_label(embeddings, label_file)



    emb_list = []

    for k in X:

        emb_list.append(embeddings[k])

    emb_list = numpy.array(emb_list)



    model = TSNE(n_components=2)

    node_pos = model.fit_transform(emb_list)



    color_idx = {}

    for i in range(len(X)):

        color_idx.setdefault(Y[i][0], [])

        color_idx[Y[i][0]].append(i)



    for c, idx in color_idx.items():

        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)

    plt.legend()



    plt.savefig('%s.png' % name)  # or '%s.pdf'%name

    plt.show()
f = open("gamma_values.txt", "w")

f.write("Gamma Values:"+"\n")

f.close()



f = open("alpha_values.txt", "w")

f.write("Alpha Values:"+"\n")

f.close()





for name in ['cora-edgelist']:



    edge_file= '/kaggle/input/data-science-lab-data-sets/%s.txt'%name

    #label_file= './%s/%s-label.txt'%(name, name)

    

    

    f_social= open(edge_file, 'r')

    

    nb_labels = 7 #5

    social_edges=[]

    

    for line in f_social:

        

        a=line.strip('\n').split(' ')

        social_edges.append((a[0],a[1]))



    

    for size in [128]: #50, 100, 200



        model= lineEmb( edge_file,  social_edges, name,  emb_size= size, alpha=5, epoch=15, batch_size=256, shuffel=True)

    

        embeddings= model.train(nb_labels)

    

    print('\n')

#node_classification(embeddings, "/kaggle/input/cora-label.txt", "cora_GEMSEC", 128)

#plot_embeddings(embeddings, "/kaggle/input/cora-label.txt", "cora_GEMSEC")