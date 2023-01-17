# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import re
import nltk
import pickle
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/articles1.csv")
data2 = pd.read_csv("../input/articles2.csv")
data3 = pd.read_csv("../input/articles3.csv")
text = data["content"]
text2 = data["content"]
# text3 = data["content"]
text = "".join(text) + "".join(text2) #+ "".join(text3)
del data, data2,  text2 #, data3, text3
text = text.lower()
text = re.sub(r'[^a-zA-Z .]', '', text)


word_list = re.sub(r'[^a-zA-Z ]', '', text)
word_list_raw = word_list.split()
word_list = list(set(word_list_raw))
word_list_len = len(word_list)
word_list_raw_len = len(word_list_raw)
freq_1gram = nltk.FreqDist(word_list_raw)

del word_list_raw

print(word_list_len)
text = text.split(".")
bigrams = []

for i in text :
    words = i.split()
    for j in range(len(words)-1) :
        bigrams.append((words[j],words[j+1]))

# with open('bigrams.pickle', 'wb') as f:
#     pickle.dump(bigrams, f, pickle.HIGHEST_PROTOCOL)
# with open('bigrams.pickle', 'rb') as f:
#     bigrams = pickle.load(f)

cfreq_2gram = nltk.ConditionalFreqDist(bigrams)
lm = nltk.ConditionalProbDist(cfreq_2gram, nltk.MLEProbDist)
# lm["my"].prob("own")

with open('language_model.pickle','wb') as f :
    pickle.dump(lm,f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(freq_1gram,f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(word_list_raw_len,f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(word_list,f, pickle.HIGHEST_PROTOCOL)
    

print("pickling done")
# def lv_distance(a, b):
#     string1 = a
#     string2 = b
#     distance = 0
#     n1 = len(string1)
#     n2 = len(string2)
    
#     if n1 >= n2:
#         for i in range(n1):
#             if i < n2:
#                 if string1[i] != string2[i]:
#                     distance += 1
#             else:
#                 distance += 1
#     else:
#         for i in range(n2):
#             if i < n1:
#                 if string2[i] != string1[i]:
#                     distance -= 1
#             else:
#                 distance -= 1
    
    
        
#     return abs(distance) + 0.5
# import heapq
# class PriorityQueue:
#     """
#       Implements a priority queue data structure. Each inserted item
#       has a priority associated with it and the client is usually interested
#       in quick retrieval of the lowest-priority item in the queue. This
#       data structure allows O(1) access to the lowest-priority item.
#     """
#     def  __init__(self):
#         self.heap = []
#         self.count = 0

#     def push(self, item, priority):
#         entry = (priority, self.count, item)
#         heapq.heappush(self.heap, entry)
#         self.count += 1

#     def pop(self):
#         (_, _, item) = heapq.heappop(self.heap)
#         return item

#     def isEmpty(self):
#         return len(self.heap) == 0

#     def update(self, item, priority):
#         # If item already in priority queue with higher priority, update its priority and rebuild the heap.
#         # If item already in priority queue with equal or lower priority, do nothing.
#         # If item not in priority queue, do the same thing as self.push.
#         for index, (p, c, i) in enumerate(self.heap):
#             if i == item:
#                 if p <= priority:
#                     break
#                 del self.heap[index]
#                 self.heap.append((priority, c, item))
#                 heapq.heapify(self.heap)
#                 break
#         else:
#             self.push(item, priority)
# print("started ucs0")
# sentence = "when it comes"
# sentence = sentence.split()
# def get_neighbours(state,sentence_words) :
#     to_ret = []
#     if len(state[0]) != 0 :
#         for i in word_list :
#             last_word= state[0][-1]
#             dist = (1-lm[last_word].prob(i)) * lv_distance(sentence_words[len(state[0])],i)
#             new_state = (state[0]+(i,), state[1] + dist)
#             to_ret.append(new_state)
    
#     else :
#         for i in word_list :
# #             dist =  (1)* lv_distance(sentence_words[len(state[0])],i)
#             dist =  (1-(freq_1gram[i] / word_list_raw_len))* lv_distance(sentence_words[len(state[0])],i)
#             new_state = ( (i,), dist)
#             to_ret.append(new_state)
#     return to_ret

# print("started ucs1")
# fringe = PriorityQueue()
# initial_state = ((),0)
# fringe.push(initial_state, initial_state[1])
# while not fringe.isEmpty() :
#     curr_state = fringe.pop()
#     if len(curr_state[0]) == len(sentence) :
#         final_state = curr_state
#         break
#     neighbours = get_neighbours(curr_state,sentence)
#     for i in neighbours :
#         fringe.push(i,i[1])

# print(final_state)