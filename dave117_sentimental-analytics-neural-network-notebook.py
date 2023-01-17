#DISCLAIMER:
# Machine Learning is about lot's of reading, so please do not afraid of reading.
# Programming is just a tool, idea of application is bigger than it so understand properly, Use open resources for that, You can use right side Tab for searching!

r = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1].lower(),r.readlines()))
r.close()

l = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),l.readlines()))
l.close()
# check Trustfullness of data
print("Total No. of Reviews : " ,len(reviews))
print("Total No. of Labels : ",len(labels))
# print text data and labels for understanding seperation and other operations
def print_reviews_lables(n):
  print(labels[n] + "\t|\t" + reviews[n][:100] + ".(conti.)..")

print("Labels.txt \t| \t reviews.txt")
print("=====================================")
print_reviews_lables(0)
print_reviews_lables(100)
print_reviews_lables(1000)
print_reviews_lables(24999) # last element 

from collections import Counter
import numpy as np

# define positive,negative and total counters
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
# define loop for counting words
for i in range(len(reviews)):
  if(labels[i]=="POSITIVE"):
    #count positive
    for words in reviews[i].split(" "):
      positive_counts[words] +=1
      total_counts[words] +=1
  else:
    # count negative
    for words in reviews[i].split(" "):
      negative_counts[words] +=1
      total_counts[words] +=1

# Do not try all print at a time in single cell, network has limit of 10000 bytes/sec
print("Positive : ",positive_counts.most_common())
print(len(positive_counts))
print("Negative : ",negative_counts.most_common())
print(len(negative_counts))
print("Total : " ,total_counts.most_common())
print(len(total_counts))
type(total_counts)
#@title Limit Digit [default = 100] { run: "auto", form-width: "30%", display-mode: "both" }
pos_neg_ratios = Counter()
Limit = 150 #@param {type:"slider", min:0, max:1000, step:50}
for term,cnt in list(total_counts.most_common()):
  if(cnt>Limit):
    pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1) # ensurity for only posible in positive_counts
    pos_neg_ratios[term] = pos_neg_ratio
# difference between last two terms
print(type(pos_neg_ratio),pos_neg_ratio)
print(type(pos_neg_ratios[term]),pos_neg_ratios[term])
print(pos_neg_ratios.most_common)
# check the ratio of diffeent word and make understanding
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["a"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
print("Pos-to-neg ratio for 'happy' = {}".format(pos_neg_ratios["happy"]))
print("Pos-to-neg ratio for 'down' = {}".format(pos_neg_ratios["down"]))
print("Pos-to-neg ratio for 'interesting' = {}".format(pos_neg_ratios["interesting"]))
print("Pos-to-neg ratio for 'enough' = {}".format(pos_neg_ratios["enough"]))
print("Pos-to-neg ratio for 'perfection' = {}".format(pos_neg_ratios["perfection"]))
print("Pos-to-neg ratio for 'pointless' = {}".format(pos_neg_ratios["pointless"]))
for word , ratio in pos_neg_ratios.most_common():
  if(ratio>1):
    pos_neg_ratios[word] = np.log(ratio)
  else:
    pos_neg_ratios[word] = -np.log((1/(ratio+0.01)))  
# recheck our distribution
print(pos_neg_ratios.most_common) 
# check the ratio of diffeent word and make understanding
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["a"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
print("Pos-to-neg ratio for 'happy' = {}".format(pos_neg_ratios["happy"]))
print("Pos-to-neg ratio for 'down' = {}".format(pos_neg_ratios["down"]))
print("Pos-to-neg ratio for 'interesting' = {}".format(pos_neg_ratios["interesting"]))
print("Pos-to-neg ratio for 'enough' = {}".format(pos_neg_ratios["enough"]))
print("Pos-to-neg ratio for 'perfection' = {}".format(pos_neg_ratios["perfection"]))
print("Pos-to-neg ratio for 'pointless' = {}".format(pos_neg_ratios["pointless"]))
# Most positive labels
print("Positive first 30 : " , pos_neg_ratios.most_common()[0:30])
print("===================")
print("Negative first 30 : " , pos_neg_ratios.most_common()[:-31:-1])
len(pos_neg_ratios)
#  Generate Vocabulary of our words
vocab = set(total_counts.keys())
vocab_size = len(vocab)
print(vocab_size)
type(vocab)
# create input layer with zeros with the size of set 
layer_0 = np.zeros((1,vocab_size))
layer_0.shape
word2index = {}
for i,word in enumerate(vocab):
  word2index[word] = i

# let's display index mapping
print(len(word2index))
#word2index

def update_input_layer(customer_review):
  global layer_0

  layer_0 *= 0 # reset the layer inputs to zero 

  # now let's count how many time individual word used in customer_review
  for word in customer_review.split(" "):
    layer_0[0][word2index[word]] +=1
update_input_layer(reviews[0])
layer_0
def get_label_target(label):
  if(label == 'POSITIVE'):
    return 1
  else:
    return 0
print(labels[0])
get_label_target(labels[0])
print(labels[1])
get_label_target(labels[1])
import time 
import sys 
import numpy as np

# Encapsulate our neural network in class
class Sentimental_Network:
    def __init__(self,reviews,labels, hidden_nodes = 10 ,learning_rate = 0.01): # Main method and methods will be called out
        np.random.seed(77)                                                      # to generate same radom data everytime (output-syncronism)
        self.pre_process_data(reviews,labels)
        self.network_architecture(len(self.review_vocab),hidden_nodes,1,learning_rate)

# generate supporting methods

    def pre_process_data(self,reviews,labels):
        review_vocab = set()                  # generate set and populate it with words in reviews
        for review in reviews:
          for word in review.split(" "):
            review_vocab.add(word)
        self.review_vocab = list(review_vocab) # track list of generated set "coverting to list of operations"
        
        labels_vocab = set()                  # use same set method for labels
        for label in labels:
          labels_vocab.add(label)
        self.labels_vocab = list(labels_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.labels_vocab_size = len(self.labels_vocab)

        #g
        self.word2index = {}               # generate dictionary of words used in reviews and indexing
        for i, word in enumerate(self.review_vocab):
          self.word2index[word] = i

        self.label2index = {}
        for label in enumerate(self.labels_vocab):
          self.label2index[label] = i

    def network_architecture(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # initialize weight
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        # generate input layer with zero values
        self.layer_0 = np.zeros((1,input_nodes))


    def update_input_layer(self,reviews):                                       # count word uniqueness and return layer_0 as input_layer
        self.layer_0 *= 0 
        for word in reviews.split(" "):
          if(word in self.word2index.keys()):
            self.layer_0[0][self.word2index[word]] +=1

    def get_label_target(seld,labels):                                          # convert positive/negative to 1/0
        if(labels == 'POSITIVE'):
          return 1
        else:
          return 0

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivation(self,output):
        return output * (1-output)

    # train network flow
    def train(self,training_reviews,training_labels):
        try:
          assert(len(training_reviews) == len(training_labels))
        except:
          print("=====================\n Number of Input data does not match with Number of Targets\n =================================")
        
        correct_ans = 0                                                         # to track correct labels for reveiws

        ### Forward Pass ###
        start = time.time()

        for i in range(len(training_reviews)):
            review = training_reviews[i]
            labels = training_labels[i]

            self.update_input_layer(review)
            layer_1 = self.layer_0.dot(self.weights_0_1)
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))


            ### Backward Pass ###
            layer_2_error = layer_2 - self.get_label_target(labels)      # Get total error
            delta_layer_2 = layer_2_error * self.sigmoid_derivation(layer_2) 

            layer_1_error = delta_layer_2.dot(self.weights_1_2.T)               # Back propogated error to inner layers (Mathematicallly: distribute error to each weight)
            delta_layer_1 = layer_1_error                                       # No activation function

            ## update weights ##
            self.weights_1_2 -= layer_1.T.dot(delta_layer_2) * self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(delta_layer_1) * self.learning_rate

            # check correctness
            if(layer_2 >= 0.5 and labels == 'POSITIVE'):
              correct_ans +=1
            elif(layer_2< 0.5 and labels == 'NEGATIVE'):
              correct_ans +=1
        

            end = time.time()
            elapsed_time = float(end - start)                                   # time mapping 
            reviews_per_second = i/elapsed_time if elapsed_time>0 else 0        # here, i is last checked review (if process fully complete then all)


            print("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                  + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                  + " #Correct:" + str(correct_ans)\
                   + " #Trained:" + str(i+1) \
                   + " Training Accuracy:" + str(correct_ans * 100 / float(i+1))[:4] + "%") 

            if (i/2500 == 0):
              print("Too Slow - less than 2500")     

    def test(self,testing_reviews,testing_labels):
        correct_ans = 0
        start = time.time()

        for i in range(len(testing_reviews)):
          pred = self.run(testing_reviews[i])# use run method because we do not need backpropagation in testing 
          if(pred==testing_labels[i]):
            correct_ans +=1
          
          end = time.time()
          elapsed_time = float(end-start)
          reviews_per_second = i/elapsed_time if elapsed_time>0 else 0        # here, i is last checked review (if process fully complete then all)


          sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                  + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                  + " #Correct:" + str(correct_ans)\
                   + " #Tested:" + str(i+1) \
                   + " Testing Accuracy:" + str(correct_ans * 100 / float(i+1))[:4] + "%") 
          
    def run(self,reviews):
        self.update_input_layer(reviews)
        layer_1 = self.layer_0.dot(self.weights_0_1)
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        if(layer_2[0] >= 0.5):
          return 'POSITIVE'
        else:
          return 'NEGATIVE'
model_0 = Sentimental_Network(reviews[:-1000],labels[:-1000], learning_rate=0.1)
print(model_0)
model_0.train(reviews[:-1000],labels[:-1000])
model_0.test(reviews[-1000:],labels[-1000:])
model_1 = Sentimental_Network(reviews[:-1000],labels[:-1000], learning_rate=0.01)
model_1.train(reviews[:-1000],labels[:-1000])
model_1.test(reviews[-1000:],labels[-1000:])
model_2 = Sentimental_Network(reviews[:-1000],labels[:-1000], learning_rate=0.001)
model_2.train(reviews[:-1000],labels[:-1000])
model_2.test(reviews[-1000:],labels[-1000:])
model_0.test(reviews[-1000:],labels[-1000:])
model_1.test(reviews[-1000:],labels[-1000:])
model_2.test(reviews[-1000:],labels[-1000:])
update_input_layer(reviews[0])
print("Total counts for different word : ",layer_0)
review_counter = Counter()
for word in reviews[0].split(" "):
  review_counter[word] +=1
review_counter.most_common()
import time 
import sys 
import numpy as np

# Encapsulate our neural network in class
class Sentimental_Network:
    def __init__(self,reviews,labels, hidden_nodes = 10 ,learning_rate = 0.01): # Main method and methods will be called out
        np.random.seed(77)                                                      # to generate same radom data everytime (output-syncronism)
        self.pre_process_data(reviews,labels)
        self.network_architecture(len(self.review_vocab),hidden_nodes,1,learning_rate)

# generate supporting methods

    def pre_process_data(self,reviews,labels):
        review_vocab = set()                                                    # generate set and populate it with words in reviews
        for review in reviews:
          for word in review.split(" "):
            review_vocab.add(word)
        self.review_vocab = list(review_vocab)                                  # track list of generated set " coverting to list of operations"
        
        labels_vocab = set()                                                    # use same set method for labels
        for label in labels:
          labels_vocab.add(label)
        self.labels_vocab = list(labels_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.labels_vocab_size = len(self.labels_vocab)

        #g
        self.word2index = {}                                                    # generate dictionary of words used in reviews and indexing
        for i, word in enumerate(self.review_vocab):
          self.word2index[word] = i

        self.label2index = {}
        for label in enumerate(self.labels_vocab):
          self.label2index[label] = i

    def network_architecture(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # initialize weight
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        # generate input layer with zero values
        self.layer_0 = np.zeros((1,input_nodes))


    def update_input_layer(self,reviews):                                       # count word uniqueness and return layer_0 as input_layer
        self.layer_0 *= 0 
        for word in reviews.split(" "):
          if(word in self.word2index.keys()):
            self.layer_0[0][self.word2index[word]] =1

    def get_label_target(seld,labels):                                          # convert positive/negative to 1/0
        if(labels == 'POSITIVE'):
          return 1
        else:
          return 0

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivation(self,output):
        return output * (1-output)

    # train network flow
    def train(self,training_reviews,training_labels):
        try:
          assert(len(training_reviews) == len(training_labels))
        except:
          print("=====================\n Number of Input data does not match with Number of Targets\n =================================")
        
        correct_ans = 0                                                         # to track correct labels for reveiws

        ### Forward Pass ###
        start = time.time()

        for i in range(len(training_reviews)):
            review = training_reviews[i]
            labels = training_labels[i]

            self.update_input_layer(review)
            layer_1 = self.layer_0.dot(self.weights_0_1)
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))


            ### Backward Pass ###
            layer_2_error = layer_2 - self.get_label_target(labels)      # Get total error
            delta_layer_2 = layer_2_error * self.sigmoid_derivation(layer_2) 

            layer_1_error = delta_layer_2.dot(self.weights_1_2.T)               # Back propogated error to inner layers (Mathematicallly: distribute error to each weight)
            delta_layer_1 = layer_1_error                                       # No activation function

            ## update weights ##
            self.weights_1_2 -= layer_1.T.dot(delta_layer_2) * self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(delta_layer_1) * self.learning_rate

            # check correctness
            if(layer_2 >= 0.5 and labels == 'POSITIVE'):
              correct_ans +=1
            elif(layer_2< 0.5 and labels == 'NEGATIVE'):
              correct_ans +=1
        

            end = time.time()
            elapsed_time = float(end - start)                                   # time mapping 
            reviews_per_second = i/elapsed_time if elapsed_time>0 else 0        # here, i is last checked review (if process fully complete then all)


            print("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                  + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                  + " #Correct:" + str(correct_ans)\
                   + " #Trained:" + str(i+1) \
                   + " Training Accuracy:" + str(correct_ans * 100 / float(i+1))[:4] + "%") 

            if (i/2500 == 0):
              print("Too Slow - less than 2500")     

    def test(self,testing_reviews,testing_labels):
        correct_ans = 0
        start = time.time()

        for i in range(len(testing_reviews)):
          pred = self.run(testing_reviews[i])                                    # use run method because we do not need backpropagation in testing 
          if(pred==testing_labels[i]):
            correct_ans +=1
          
          end = time.time()
          elapsed_time = float(end-start)
          reviews_per_second = i/elapsed_time if elapsed_time>0 else 0        # here, i is last checked review (if process fully complete then all)


          sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                  + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                  + " #Correct:" + str(correct_ans)\
                   + " #Tested:" + str(i+1) \
                   + " Testing Accuracy:" + str(correct_ans * 100 / float(i+1))[:4] + "%") 
          
    def run(self,reviews):
        self.update_input_layer(reviews)
        layer_1 = self.layer_0.dot(self.weights_0_1)
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        if(layer_2[0] >= 0.5):
          return 'POSITIVE'
        else:
          return 'NEGATIVE'
# Let's check our model_2 as it had best accuracy so far... 
model_2 = Sentimental_Network(reviews[:-1000],labels[:-1000], learning_rate=0.001)
model_2.train(reviews[:-1000],labels[:-1000])

model_2.test(reviews[-1000:],labels[-1000:])
#===========================
# Applying general procedure
#============================
layer_0  = np.zeros(10)
layer_0 
# define 1 on some place to represent existance of that index word in particular review
layer_0[4] = 1
layer_0[8] =1
layer_0
# create layer_1 by multiplication of weight matrix
weight_0_1 = np.random.randn(10,5) # as layer_1 has 5 neurons(assume)
weight_0_1
# create layer_1
layer_1 = layer_0.dot(weight_0_1)
layer_1
#============================
# Applying hypothesis_2:
#===========================
indices = [4,8]
layer_1  = np.zeros(5) # we assume 5 neurons for layer_1
for index in indices:
  layer_1 += (1*weight_0_1[index])
layer_1
import time 
import sys 
import numpy as np

# Encapsulate our neural network in class
class Sentimental_Network:
    def __init__(self,reviews,labels, hidden_nodes = 10 ,learning_rate = 0.01): # Main method and methods will be called out
        np.random.seed(77)                                                      # to generate same radom data everytime (output-syncronism)
        self.pre_process_data(reviews,labels)
        self.network_architecture(len(self.review_vocab),hidden_nodes,1,learning_rate)

# generate supporting methods

    def pre_process_data(self,reviews,labels):
        review_vocab = set()                                                    # generate set and populate it with words in reviews
        for review in reviews:
          for word in review.split(" "):
            review_vocab.add(word)
        self.review_vocab = list(review_vocab)                                  # track list of generated set " coverting to list of operations"
        
        labels_vocab = set()                                                    # use same set method for labels
        for label in labels:
          labels_vocab.add(label)
        self.labels_vocab = list(labels_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.labels_vocab_size = len(self.labels_vocab)

        #g
        self.word2index = {}                                                    # generate dictionary of words used in reviews and indexing
        for i, word in enumerate(self.review_vocab):
          self.word2index[word] = i

        self.label2index = {}
        for label in enumerate(self.labels_vocab):
          self.label2index[label] = i

    def network_architecture(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # initialize weight
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))

# remove layer_0 creation and generate layer_1 instead...

        self.layer_1 = np.zeros((1,hidden_nodes))


# remove update_layer as we do not need

    def get_label_target(seld,labels):                                          # convert positive/negative to 1/0
        if(labels == 'POSITIVE'):
          return 1
        else:
          return 0

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivation(self,output):
        return output * (1-output)

    # train network flow
    def train(self,training_reviews_raw,training_labels):                       # generate training_reviews_raw as we need to update trainint_review_raw
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                  indices.add(self.word2index[word])
            training_reviews.append(list(indices))


        try:
          assert(len(training_reviews) == len(training_labels))
        except:
          print("=====================\n Number of Input data does not match with Number of Targets\n =================================")
        
        correct_ans = 0                                                         # to track correct labels for reveiws

        ### Forward Pass ###
        start = time.time()

        for i in range(len(training_reviews)):
            review = training_reviews[i]
            labels = training_labels[i]

            self.layer_1 *= 0
            for index in review:
              self.layer_1 += self.weights_0_1[index]
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))


            ### Backward Pass ###
            layer_2_error = layer_2 - self.get_label_target(labels)      # Get total error
            delta_layer_2 = layer_2_error * self.sigmoid_derivation(layer_2) 

            layer_1_error = delta_layer_2.dot(self.weights_1_2.T)               # Back propogated error to inner layers (Mathematicallly: distribute error to each weight)
            delta_layer_1 = layer_1_error                                       # No activation function

            ## update weights ##
            self.weights_1_2 -= self.layer_1.T.dot(delta_layer_2) * self.learning_rate
            for index in review:
              self.weights_0_1[index] -= delta_layer_1[0] * self.learning_rate  # update weight to index positions (relevant)
          

            # check correctness
            if(layer_2 >= 0.5 and labels == 'POSITIVE'):
              correct_ans +=1
            elif(layer_2< 0.5 and labels == 'NEGATIVE'):
              correct_ans +=1
        

            end = time.time()
            elapsed_time = float(end - start)                                   # time mapping 
            reviews_per_second = i/elapsed_time if elapsed_time>0 else 0        # here, i is last checked review (if process fully complete then all)


            print("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                  + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                  + " #Correct:" + str(correct_ans)\
                   + " #Trained:" + str(i+1) \
                   + " Training Accuracy:" + str(correct_ans * 100 / float(i+1))[:4] + "%") 

            if (i/2500 == 0):
              print("Too Slow - less than 2500")     

    def test(self,testing_reviews,testing_labels):
        correct_ans = 0
        start = time.time()

        for i in range(len(testing_reviews)):
          pred = self.run(testing_reviews[i])                                    # use run method because we do not need backpropagation in testing 
          if(pred==testing_labels[i]):
            correct_ans +=1
          
          end = time.time()
          elapsed_time = float(end-start)
          reviews_per_second = i/elapsed_time if elapsed_time>0 else 0        # here, i is last checked review (if process fully complete then all)


          sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                  + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                  + " #Correct:" + str(correct_ans)\
                   + " #Tested:" + str(i+1) \
                   + " Testing Accuracy:" + str(correct_ans * 100 / float(i+1))[:4] + "%") 
          
    def run(self,reviews):
        self.layer_1 *= 0
        unique_indices = set()
        for word in reviews.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]                
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if(layer_2[0] >= 0.5):
          return 'POSITIVE'
        else:
          return 'NEGATIVE'
# Let's check our model_2 as it had best accuracy so far... 
model_2 = Sentimental_Network(reviews[:-1000],labels[:-1000], learning_rate=0.001)
model_2.train(reviews[:-1000],labels[:-1000])
model_2.test(reviews[-1000:],labels[-1000:])
# Let's check our model_2 as it had best accuracy so far... 
model_2 = Sentimental_Network(reviews[:-1000],labels[:-1000], learning_rate=0.1)
model_2.train(reviews[:-1000] * 2,labels[:-1000] *2)
model_2.test(reviews[-1000:],labels[-1000:])
# Understanding:
# First review positive label common words
pos_neg_ratios.most_common()[0:30]

# Negavie label most common words
pos_neg_ratios.most_common()[:-31:-1]
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()
hist, edges = np.histogram(list(map(lambda x:x[1],pos_neg_ratios.most_common())), density=True, bins=100, normed=True)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="Word Positive/Negative Affinity Distribution")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
frequency_frequency = Counter()

for word, cnt in total_counts.most_common():
    frequency_frequency[cnt] += 1
hist, edges = np.histogram(list(map(lambda x:x[1],frequency_frequency.most_common())), density=True, bins=100, normed=True)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="The frequency distribution of the words in our corpus")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
import time 
import sys 
import numpy as np

# Encapsulate our neural network in class
class Sentimental_Network:
    def __init__(self,reviews,labels,polarity_cutoff = 0.1,min_counts=10,hidden_nodes = 10 ,learning_rate = 0.1): # Main method and methods will be called out
        np.random.seed(77)                                                      # to generate same radom data everytime (output-syncronism)
        self.pre_process_data(reviews,labels,polarity_cutoff,min_counts)
        self.network_architecture(len(self.review_vocab),hidden_nodes,1,learning_rate)

# generate supporting methods

    def pre_process_data(self,reviews,labels,polarity_cutoff,min_counts):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
          if(labels[i]=='POSITIVE'):
            for word in reviews[i].split(" "):
              positive_counts[word] +=1
              total_counts[word] += 1
          else:
            for word in reviews[i].split(" "):
              negative_counts[word] +=1
              total_counts[word] += 1
         
        pos_neg_ratios = Counter()
        for term, cnt in list(total_counts.most_common()):
          if(cnt>50):
            pos_neg_ratio = positive_counts[term]/float(negative_counts[term]+1)
            pos_neg_ratios[term] = pos_neg_ratio
        
        for word, ratio in pos_neg_ratios.most_common():
          if(ratio>1):
            pos_neg_ratios[word] = np.log(ratio)
          else:
            pos_neg_ratios[word] = -np.log((1/(ratio+0.01)))

#================================================================================
        review_vocab = set()                                                    # generate set and populate it with words in reviews
        for review in reviews:
          for word in review.split(" "):                                        # set condition for performance improvement
            if(total_counts[word]>min_counts):
              if(word in pos_neg_ratios.keys()):
                if((pos_neg_ratios[word]>=polarity_cutoff) or (pos_neg_ratios[word]<= -polarity_cutoff)):
                  review_vocab.add(word)
            else: 
              review_vocab.add(word)

        self.review_vocab = list(review_vocab)                                  # track list of generated set " coverting to list of operations"
        
        labels_vocab = set()                                                    # use same set method for labels
        for label in labels:
          labels_vocab.add(label)
        self.labels_vocab = list(labels_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.labels_vocab_size = len(self.labels_vocab)

        #g
        self.word2index = {}                                                    # generate dictionary of words used in reviews and indexing
        for i, word in enumerate(self.review_vocab):
          self.word2index[word] = i

        self.label2index = {}
        for label in enumerate(self.labels_vocab):
          self.label2index[label] = i

    def network_architecture(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # initialize weight
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))

# remove layer_0 creation and generate layer_1 instead...

        self.layer_1 = np.zeros((1,hidden_nodes))


# remove update_layer as we do not need

    def get_label_target(seld,labels):                                          # convert positive/negative to 1/0
        if(labels == 'POSITIVE'):
          return 1
        else:
          return 0

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivation(self,output):
        return output * (1-output)

    # train network flow
    def train(self,training_reviews_raw,training_labels):                       # generate training_reviews_raw as we need to update trainint_review_raw
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                  indices.add(self.word2index[word])
            training_reviews.append(list(indices))


        try:
          assert(len(training_reviews) == len(training_labels))
        except:
          print("=====================\n Number of Input data does not match with Number of Targets\n =================================")
        
        correct_ans = 0                                                         # to track correct labels for reveiws

        ### Forward Pass ###
        start = time.time()

        for i in range(len(training_reviews)):
            review = training_reviews[i]
            labels = training_labels[i]

            self.layer_1 *= 0
            for index in review:
              self.layer_1 += self.weights_0_1[index]
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))


            ### Backward Pass ###
            layer_2_error = layer_2 - self.get_label_target(labels)      # Get total error
            delta_layer_2 = layer_2_error * self.sigmoid_derivation(layer_2) 

            layer_1_error = delta_layer_2.dot(self.weights_1_2.T)               # Back propogated error to inner layers (Mathematicallly: distribute error to each weight)
            delta_layer_1 = layer_1_error                                       # No activation function

            ## update weights ##
            self.weights_1_2 -= self.layer_1.T.dot(delta_layer_2) * self.learning_rate
            for index in review:
              self.weights_0_1[index] -= delta_layer_1[0] * self.learning_rate  # update weight to index positions (relevant)
          

            # check correctness
            if(layer_2 >= 0.5 and labels == 'POSITIVE'):
              correct_ans +=1
            elif(layer_2< 0.5 and labels == 'NEGATIVE'):
              correct_ans +=1
        

            end = time.time()
            elapsed_time = float(end - start)                                   # time mapping 
            reviews_per_second = i/elapsed_time if elapsed_time>0 else 0        # here, i is last checked review (if process fully complete then all)


            print("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                  + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                  + " #Correct:" + str(correct_ans)\
                   + " #Trained:" + str(i+1) \
                   + " Training Accuracy:" + str(correct_ans * 100 / float(i+1))[:4] + "%") 

            if (i/2500 == 0):
              print("Too Slow - less than 2500")     

    def test(self,testing_reviews,testing_labels):
        correct_ans = 0
        start = time.time()

        for i in range(len(testing_reviews)):
          pred = self.run(testing_reviews[i])                                    # use run method because we do not need backpropagation in testing 
          if(pred==testing_labels[i]):
            correct_ans +=1
          
          end = time.time()
          elapsed_time = float(end-start)
          reviews_per_second = i/elapsed_time if elapsed_time>0 else 0        # here, i is last checked review (if process fully complete then all)


          sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                  + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                  + " #Correct:" + str(correct_ans)\
                   + " #Tested:" + str(i+1) \
                   + " Testing Accuracy:" + str(correct_ans * 100 / float(i+1))[:4] + "%") 
          
    def run(self,reviews):
        self.layer_1 *= 0
        unique_indices = set()
        for word in reviews.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]                
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if(layer_2[0] >= 0.5):
          return 'POSITIVE'
        else:
          return 'NEGATIVE'
# Let's check our model_2 as it had best accuracy so far... 
model_2 = Sentimental_Network(reviews[:-1000],labels[:-1000],min_counts=20,polarity_cutoff=0.05,learning_rate=0.01)
model_2.train(reviews[:-1000],labels[:-1000])
model_2.test(reviews[-1000:],labels[-1000:])
model_2 = Sentimental_Network(reviews[:-1000],labels[:-1000],polarity_cutoff=0.8,min_counts=20,learning_rate=0.1)
model_2.train(reviews[:-1000],labels[:-1000])
model_2.test(reviews[-1000:],labels[-1000:])
mlp_full = Sentimental_Network(reviews[:-1000],labels[:-1000],min_counts=0,polarity_cutoff=0,learning_rate=0.01)
mlp_full.train(reviews[:-1000],labels[:-1000])
def get_most_similar_words(focus = "horrible"):
    most_similar = Counter()

    for word in mlp_full.word2index.keys():
        most_similar[word] = np.dot(mlp_full.weights_0_1[mlp_full.word2index[word]],mlp_full.weights_0_1[mlp_full.word2index[focus]])
    
    return most_similar.most_common()
get_most_similar_words("excellent")
get_most_similar_words("terrible")
import matplotlib.colors as colors

words_to_visualize = list()
for word, ratio in pos_neg_ratios.most_common(500):
    if(word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)
    
for word, ratio in list(reversed(pos_neg_ratios.most_common()))[0:500]:
    if(word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)
pos = 0
neg = 0

colors_list = list()
vectors_list = list()
for word in words_to_visualize:
    if word in pos_neg_ratios.keys():
        vectors_list.append(mlp_full.weights_0_1[mlp_full.word2index[word]])
        if(pos_neg_ratios[word] > 0):
            pos+=1
            colors_list.append("#00ff00")
        else:
            neg+=1
            colors_list.append("#000000")
    
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(vectors_list)
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="vector T-SNE for most polarized words")

source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],
                                    x2=words_top_ted_tsne[:,1],
                                    names=words_to_visualize,
                                    color=colors_list))

p.scatter(x="x1", y="x2", size=8, source=source, fill_color="color")

word_labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
p.add_layout(word_labels)

show(p)

# green indicates positive words, black indicates negative words
# Keep Learning, Enjoy Empowering