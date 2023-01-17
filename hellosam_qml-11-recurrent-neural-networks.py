import torch
from torch import nn
words = ['I', 'am', 'from', 'France', 'and', 'I', 'speak', 'French', '.']

num_words = len(words)
embedding_size = 5
# just random embeddings for now
word_embeddings = torch.randn(num_words, embedding_size)
word_embeddings
word_embeddings
# Recurrent neural network
# Idea: start with a basic feedforward network
hidden_size = 3
layer = nn.Linear(embedding_size, hidden_size)
# Transform word by word
state_list = []
# I am from France
for word_emb in word_embeddings:
    state = layer(word_emb)
    state_list.append(state)
state_list
# Add some 'memory': keep a hidden state running

memory_list = []
memory = torch.randn(3)

state2memory = nn.Linear(hidden_size, hidden_size)
input2memory = nn.Linear(embedding_size, hidden_size)
for word_emb in word_embeddings:
    state = input2memory(word_emb)
    new_memory = state2memory(memory)
    # update memory with state and input
    memory = torch.tanh(new_memory + state)
    
    memory_list.append(memory)
memory_list
# memory_list is the sequence of memory
memory_list