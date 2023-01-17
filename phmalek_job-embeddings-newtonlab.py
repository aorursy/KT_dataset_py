with open('/kaggle/input/job-data/description_data.txt','r') as f:
    text = f.read()
from utils import create_lookup_tables
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text.lower())
# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
print(words[:100])
print('the number of words in the document: {}'.format(len(words)))
print('the number of unique words in the document: {}'.format(len(set(words))))
vocab_to_int, int_to_vocab = create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]
from collections import Counter
import random
import numpy as np

threshold = 1e-5
word_counts = Counter(int_words)

total_count = len(int_words)
freqs = {word: count/total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}

train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

print(train_words[:30])
print([int_to_vocab[i] for i in train_words[:30]])
def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]
    
    return list(target_words)
# test get_target

int_text = list(range(10))
print('Input: ', int_text)
idx=5 # word index of interest

target = get_target(int_text, idx=idx, window_size=5)
print('Target: ', target) 
def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words)//batch_size #number of batches
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y
def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):

    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities
import torch
from torch import nn
import torch.optim as optim
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # Initialize embedding tables with uniform distribution
        # I believe this helps with convergence
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors
    
    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    
    def forward_noise(self, batch_size, n_samples):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)
        
        device = "cuda" if model.out_embed.weight.is_cuda else "cpu"
        noise_words = noise_words.to(device)
        
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vectors
class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        
        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_ps = self.log_softmax(scores)
        
        return log_ps
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        
        batch_size, embed_size = input_vectors.shape
        
        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get our noise distribution
# Using word frequencies calculated earlier in the notebook
word_freqs = np.array(sorted(freqs.values(), reverse=True))
unigram_dist = word_freqs/word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

# instantiating the model
embedding_dim = 300
model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device)

# using the loss that we defined
criterion = NegativeSamplingLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 1500
steps = 0
epochs = 20

# train for some number of epochs
for e in range(epochs):
    
    # get our input, target batches
    for input_words, target_words in get_batches(train_words, 512):
        steps += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # input, output, and noise vectors
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        noise_vectors = model.forward_noise(inputs.shape[0], 5)

        # negative sampling loss
        loss = criterion(input_vectors, output_vectors, noise_vectors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss stats
        if steps % print_every == 0:
            print("Epoch: {}/{}".format(e+1, epochs))
            print("Loss: ", loss.item()) # avg batch loss at this point in training
            valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...\n")
checkpoint = {'model': SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device),
              'criterion' : NegativeSamplingLoss(),
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, '/kaggle/working/checkpoint.pth')
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
embeddings = model.out_embed.weight.to('cpu').data.numpy()
viz_words = 200
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])
fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
import torch
from torch import nn
import torch.optim as optim
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model
model = load_checkpoint('/kaggle/working/checkpoint.pth')
print(model)
#test it:
def find_similar_words(embedding, word, k=6,device='cuda'):

    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    word_int = vocab_to_int[word]
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array([word_int])

    valid_examples = torch.LongTensor(valid_examples).to('cuda')
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
    
    closest_idxs = np.array(similarities.topk(k)[1].to('cpu'))[0]
    return [int_to_vocab[i] for i in closest_idxs]
topk = find_similar_words(model.in_embed, 'cleaning',6, device='cpu')
print(topk)
import json
def read_json(file,max_lines):

	with open(file, "r") as infile:
		num_lines = 0
		for l in infile:
			try:
				j = json.loads(l)
				yield j

				num_lines += 1
				if num_lines >= max_lines:
					break

			except Exception as error:
				print(error)
file = '/kaggle/input/jobs-trimmed/jobs_data.json'
j = read_json(file,1)
data= next(j)
#prepare embeddings for each job:
job_descr_list = []
for job in data:
    descr_txt = job['description']
    tokens = word_tokenize(descr_txt.lower())
    words = [word for word in tokens if word.isalpha()]
    job_descr_list.append(words)
sub_sampled_job_descr = []
for i, job in enumerate(job_descr_list):
    int_job = [vocab_to_int[w] for w in job if w in vocab_to_int.keys()]
    sub_int_job = subsampler(int_job,1)
    sub_sampled_job_descr.append(sub_int_job)
embeddings = []
#for lst in sub_sampled_job_descr:
for i, job in enumerate(job_descr_list):
    lst = [vocab_to_int[w] for w in job if w in vocab_to_int.keys()]    
    lst_tensor = torch.LongTensor(lst).to(device)
    lst_embeddings = model.in_embed(lst_tensor)
    magnitudes = lst_embeddings.pow(2).sum(dim=1).sqrt().unsqueeze(0).T
    lst_embeddings = lst_embeddings/magnitudes
    embeddings.append(lst_embeddings)
def get_search_score(idx, word): 
    word_int = vocab_to_int[word]
    word_tensor = torch.LongTensor(np.array([word_int])).to('cuda')
    word_embedding = model.in_embed(word_tensor)
    sim = torch.mm(word_embedding, embeddings[idx].t())
    sim=sim.to('cpu').data.numpy()
    score = np.mean(sim)
    return score
def get_jobs(word):
    scores = []
    for i in range(len(embeddings)):
        scores.append(get_search_score(i,word))
    scores_dic = {}
    for i,score in enumerate(scores):
        scores_dic[score] = i
    sorted_idx = [scores_dic[key] for key in sorted(scores_dic.keys(), reverse=True)]
    sorted_scores = [key for key in sorted(scores_dic.keys(), reverse=True)]
    return sorted_idx, sorted_scores
found_idx,found_scores = get_jobs('clean')
for i in found_idx[:10]:
    print(data[i]['title'])
found_idx,found_scores = get_job_ids('python')
for i in found_idx[:10]:
    print(data[i]['title'])
found_idx,found_scores = get_job_ids('java')
for i in found_idx[:10]:
    print(data[i]['title'])
found_idx,found_scores = get_job_ids('phd')
for i in found_idx[:10]:
    print(data[i]['title'])