# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import time

import sys

from collections import Counter

#import numpy as np



# Let's tweak our network from before to model these phenomena

class SentimentNetwork:

    def __init__(self, reviews,labels,min_count = 10,polarity_cutoff = 0.1,hidden_nodes = 10, learning_rate = 0.1):

       

        np.random.seed(1)

    

        self.pre_process_data(reviews, polarity_cutoff, min_count)

        

        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

        

        

    def pre_process_data(self,reviews, polarity_cutoff,min_count):

        

        positive_counts = Counter()

        negative_counts = Counter()

        total_counts = Counter()



        for i in range(len(reviews)):

            if(labels[i] == 'robot'):

                for word in reviews[i].lower().split(" "):

                    positive_counts[word] += 1

                    total_counts[word] += 1

            else:

                for word in reviews[i].lower().split(" "):

                    negative_counts[word] += 1

                    total_counts[word] += 1



        pos_neg_ratios = Counter()



        for term,cnt in list(total_counts.most_common()):

            if(cnt >= 50):

                pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)

                pos_neg_ratios[term] = pos_neg_ratio



        for word,ratio in pos_neg_ratios.most_common():

            if(ratio > 1):

                pos_neg_ratios[word] = np.log(ratio)

            else:

                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

        

        review_vocab = set()

        for review in reviews:

            for word in review.split(" "):

                if(total_counts[word] > min_count):

                    if(word in pos_neg_ratios.keys()):

                        if((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):

                            review_vocab.add(word)

                    else:

                        review_vocab.add(word)

        self.review_vocab = list(review_vocab)

        

        label_vocab = set()

        for label in labels:

            label_vocab.add(label)

        

        self.label_vocab = list(label_vocab)

        

        self.review_vocab_size = len(self.review_vocab)

        self.label_vocab_size = len(self.label_vocab)

        

        self.word2index = {}

        for i, word in enumerate(self.review_vocab):

            self.word2index[word] = i

        

        self.label2index = {}

        for i, label in enumerate(self.label_vocab):

            self.label2index[label] = i

         

        

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        # Set number of nodes in input, hidden and output layers.

        self.input_nodes = input_nodes

        self.hidden_nodes = hidden_nodes

        self.output_nodes = output_nodes



        # Initialize weights

        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))

    

        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 

                                                (self.hidden_nodes, self.output_nodes))

        

        self.learning_rate = learning_rate

        

        self.layer_0 = np.zeros((1,input_nodes))

        self.layer_1 = np.zeros((1,hidden_nodes))

        

    def sigmoid(self,x):

        return 1 / (1 + np.exp(-x))

    

    

    def sigmoid_output_2_derivative(self,output):

        return output * (1 - output)

    

    def update_input_layer(self,review):



        # clear out previous state, reset the layer to be all 0s

        self.layer_0 *= 0

        for word in review.split(" "):

            self.layer_0[0][self.word2index[word]] = 1



    def get_target_for_label(self,label):

        if(label == 'robot'):

            return 1

        else:

            return 0

        

    def train(self, training_reviews_raw, training_labels):

        

        training_reviews = list()

        for review in training_reviews_raw:

            indices = set()

            for word in review.split(" "):

                if(word in self.word2index.keys()):

                    indices.add(self.word2index[word])

            training_reviews.append(list(indices))

        

        assert(len(training_reviews) == len(training_labels))

        

        correct_so_far = 0

        

        start = time.time()

        

        for i in range(len(training_reviews)):

            

            review = training_reviews[i]

            label = training_labels[i]

            

            #### Implement the forward pass here ####

            ### Forward pass ###



            # Input Layer



            # Hidden layer

            # layer_1 = self.layer_0.dot(self.weights_0_1)

            self.layer_1 *= 0

            for index in review:

                self.layer_1 += self.weights_0_1[index]

            

            # Output layer

            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))



            #### Implement the backward pass here ####

            ### Backward pass ###



            # Output error

            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.

            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)



            # Backpropagated error

            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer

            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error



            # Update the weights

            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step

            

            for index in review:

                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step



            if(layer_2 >= 0.5 and label == 'robot'):

                correct_so_far += 1

            if(layer_2 < 0.5 and label == 'human'):

                correct_so_far += 1

            

            reviews_per_second = i / float(time.time() - start)

            

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")

        

    

    def test(self, testing_reviews, testing_labels):

        

        correct = 0

        

        start = time.time()

        

        for i in range(len(testing_reviews)):

            pred = self.run(testing_reviews[i])

            if(pred == testing_labels[i]):

                correct += 1

            

            reviews_per_second = i / float(time.time() - start)

            

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \

                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \

                            + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")

    

    def run(self, review):

        

        # Input Layer





        # Hidden layer

        self.layer_1 *= 0

        unique_indices = set()

        for word in review.lower().split(" "):

            if word in self.word2index.keys():

                unique_indices.add(self.word2index[word])

        for index in unique_indices:

            self.layer_1 += self.weights_0_1[index]

        

        # Output layer

        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        

        if(layer_2[0] >= 0.5):

            return "robot"

        else:

            return "human"

        
data = pd.read_csv('../input/rdany_conversations_2016-03-01.csv')

messages = data["text"]

labels = data["source"]

mlp = SentimentNetwork(messages,labels,hidden_nodes=30,min_count=5,polarity_cutoff=0.01,learning_rate=0.01)
mlp.train(messages,labels)

mlp.train(messages,labels)

mlp.train(messages,labels)

mlp.train(messages,labels)
print (mlp.run("[start]"))

print (mlp.run("hi there!"))
mlp.weights_0_1[mlp.word2index["hi"]]
def get_most_similar_words(focus = "hi"):

    most_similar = Counter()



    for word in mlp.word2index.keys():

        most_similar[word] = np.dot(mlp.weights_0_1[mlp.word2index[word]],mlp.weights_0_1[mlp.word2index[focus]])

    

    return most_similar.most_common()
get_most_similar_words("hi")[:10]
get_most_similar_words("bad")[:10]
words = Counter()

robot_words = Counter()

human_words = Counter()





for k,i in data.iterrows():

    for word in i["text"].split(" "):

        words[word] += 1

        if i["source"] == "robot":

            robot_words[word] += 1

        else:

            human_words[word] += 1



robot_human_ratios = Counter()



for term,cnt in list(words.most_common()):

    if(cnt > 10):

        robot_human_ratio = robot_words[term] / float(human_words[term]+1)

        robot_human_ratios[term] = robot_human_ratio



for word,ratio in robot_human_ratios.most_common():

    if(ratio > 1):

        robot_human_ratios[word] = np.log(ratio)

    else:

        robot_human_ratios[word] = -np.log((1 / (ratio+0.01)))
import matplotlib.colors as colors



words_to_visualize = list()

for word, ratio in robot_human_ratios.most_common():

    if(word in mlp.word2index.keys()):

        words_to_visualize.append(word)

    

#for word, ratio in list(reversed(robot_human_ratios.most_common()))[0:500]:

#    if(word in mlp.word2index.keys()):

#        words_to_visualize.append(word)
pos = 0

neg = 0



colors_list = list()

vectors_list = list()

for word in words_to_visualize:

    if word in robot_human_ratios.keys():

        vectors_list.append(mlp.weights_0_1[mlp.word2index[word]])

        if(robot_human_ratios[word] > 0):

            pos+=1

            colors_list.append("#00ff00")

        else:

            neg+=1

            colors_list.append("#000000")
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)

words_top_ted_tsne = tsne.fit_transform(vectors_list)
from bokeh.models import ColumnDataSource, LabelSet

from bokeh.plotting import figure, show, output_file

from bokeh.io import output_notebook

output_notebook()
p = figure(tools="pan,wheel_zoom,reset,save",

           toolbar_location="above",

           title="vector T-SNE for most polarized words")



source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],

                                    x2=words_top_ted_tsne[:,1],

                                    names=words_to_visualize))



p.scatter(x="x1", y="x2", size=8, source=source,color=colors_list)



word_labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,

                  text_font_size="8pt", text_color="#555555",

                  source=source, text_align='center')

p.add_layout(word_labels)



show(p)



# green indicates robot words, black indicates human words
mlp.run("hi there! how are you?")
mlp.run("i don't know what to do")