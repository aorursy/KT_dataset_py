import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F

import csv

import random

import re

import os

import unicodedata

import codecs

import itertools
USE_CUDA = torch.cuda.is_available()

device = torch.device("cuda" if USE_CUDA else "cpu")
USE_CUDA
device
lines_filepath=os.path.join("../input/movie_lines.txt")

conv_filepath=os.path.join("../input/movie_conversations.txt")
#visuilize some lines

with open (lines_filepath,'rb') as file:

    lines = file.readlines()

for line in lines[:8]:

    print(line)

    print(line.strip())

    
# Splits each line of the file into a dictionary of fields [lineID,chartcterID,movi ID,charcter, text]

line_fields=["lineID","chartcterID","movieID","charcter","text"]

lines = {}

with open(lines_filepath, 'r', encoding='iso-8859-1') as f:

    for line in f:

        values = line.split(" +++$+++ ")

            # Extract fields

        lineObj = {}

        for i, field in enumerate(line_fields):

            lineObj[field] = values[i]

        lines[lineObj['lineID']] = lineObj

            

        
lines
list(lines.items())[0]
lines['L194']
# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*

conv_fields=["charcter1ID","charcter2ID","movieID","utteranceIDs"]

conversations = []

   

with open(conv_filepath, 'r', encoding='iso-8859-1') as f:

    for line in f:

        values = line.split(" +++$+++ ")

        # Extract fields

        convObj = {}

        for i, field in enumerate(conv_fields):

            

            convObj[field] = values[i]

        # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")

        lineIds = eval(convObj["utteranceIDs"])

        # Reassemble lines

        convObj["lines"] = []

        for lineId in lineIds:

            convObj["lines"].append(lines[lineId])

        conversations.append(convObj)
conversations[0]
len(conversations[0]["lines"])
conversations[0]["lines"]
conversations[0]["lines"][0]
conversations[0]["lines"][0]["text"].strip()
# Extracts pairs of sentences from conversations



qa_pairs = []

for conversation in conversations:

    # Iterate over all the lines of the conversation

    for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)

        inputLine = conversation["lines"][i]["text"].strip()

        targetLine = conversation["lines"][i+1]["text"].strip()

        # Filter wrong samples (if one of the lists is empty)

        if inputLine and targetLine:

            qa_pairs.append([inputLine, targetLine])
qa_pairs
len(qa_pairs)
# Define path to new file

datafile = os.path.join("formatted_movie_lines.txt")

delimiter = '\t'

# Unescape the delimiter

delimiter = str(codecs.decode(delimiter, "unicode_escape"))



# Write new csv file

print("\nWriting newly formatted file...")

with open(datafile, 'w', encoding='utf-8') as outputfile:

    writer = csv.writer(outputfile, delimiter=delimiter)

    for pair in qa_pairs :

        writer.writerow(pair)

print("Done writing to file")
#visulize some lines

datafile = os.path.join("formatted_movie_lines.txt")

with open(datafile,'rb') as file:

    lines=file.readlines()

for line in lines[:8]:

    print(line)
# Default word tokens

PAD_token = 0  # Used for padding short sentences

SOS_token = 1  # Start-of-sentence token

EOS_token = 2  # End-of-sentence token



class Vocabulary:

    def __init__(self, name):

        self.name = name

        self.trimmed = False

        self.word2index = {}

        self.word2count = {}

        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}

        self.num_words = 3  # Count SOS, EOS, PAD



    def addSentence(self, sentence):

        for word in sentence.split(' '):

            self.addWord(word)



    def addWord(self, word):

        if word not in self.word2index:

            self.word2index[word] = self.num_words

            self.word2count[word] = 1

            self.index2word[self.num_words] = word

            self.num_words += 1

        else:

            self.word2count[word] += 1



    # Remove words below a certain count threshold

    def trim(self, min_count):

        if self.trimmed:

            return

        self.trimmed = True



        keep_words = []



        for k, v in self.word2count.items():

            if v >= min_count:

                keep_words.append(k)



        print('keep_words {} / {} = {:.4f}'.format(

            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)

        ))



        # Reinitialize dictionaries

        self.word2index = {}

        self.word2count = {}

        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}

        self.num_words = 3 # Count default tokens



        for word in keep_words:

            self.addWord(word)
#turn aunicode string to plain AscII

def unicodeToAscii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s)if unicodedata.category(c) != 'Mn')
#Test the function 

unicodeToAscii("Montréal")
#Processing the text

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())

    #replace any !? 

    s = re.sub(r"([.!?])", r" \1", s)

    #remove

    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    s = re.sub(r"\s+", r" ", s).strip()

    return s
#test normlizing

normalizeString("aya!!@ 346")
# Read query/response pairs and return a voc object

datafile = os.path.join("formatted_movie_lines.txt")

print("Reading the process lines .. please wait")

# Read the file and split into lines

lines = open(datafile, encoding='utf-8').read().strip().split('\n')

# Split every line into pairs and normalize

pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

print("Done Reading")

voc =Vocabulary("CornellMovie-Dialog corpus")

lines[0]
len(pairs[0][1].split())<10
len(pairs)
len(pairs[1])
MAX_LENGTH = 10  # Maximum sentence length to consider



# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold

def filterPair(p):

    # Input sequences need to preserve the last word for EOS token

    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH



# Filter pairs using filterPair condition

def filterPairs(pairs):

    return [pair for pair in pairs if filterPair(pair)]

pairs= [pair for pair in pairs if len(pair)>1]

print("there are {} pairs/conversations in the dataset".format(len(pairs)))

pairs= filterPairs(pairs)

print("After filtering there are {} pairs/conversations=".format(len(pairs)))
#Load throth each pair of and add the question and reply sentence to vocabulary

for pair in pairs:

    voc.addSentence(pair[0])

    voc.addSentence(pair[1])

print("Counted words:", voc.num_words)

for pair in pairs[:10]:

    print(pair)

MIN_COUNT = 3    # Minimum word count threshold for trimming



def trimRareWords(voc, pairs, MIN_COUNT):

    # Trim words used under the MIN_COUNT from the voc

    voc.trim(MIN_COUNT)

    # Filter out pairs with trimmed words

    keep_pairs = []

    for pair in pairs:

        input_sentence = pair[0]

        output_sentence = pair[1]

        keep_input = True

        keep_output = True

        # Check input sentence

        for word in input_sentence.split(' '):

            if word not in voc.word2index:

                keep_input = False

                break

        # Check output sentence

        for word in output_sentence.split(' '):

            if word not in voc.word2index:

                keep_output = False

                break



        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence

        if keep_input and keep_output:

            keep_pairs.append(pair)



    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))

    return keep_pairs





# Trim voc and pairs

pairs = trimRareWords(voc, pairs, MIN_COUNT)
def indexesFromSentence(voc, sentence):

    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
pairs[1][0]
#Test the function

indexesFromSentence(voc, pairs[1][0])
#Define some samples for testing

inp=[]

out=[]

for pair in pairs[:10]:

    inp.append(pair[0])

    out.append(pair[1])

print(inp)

print(len(inp))

indexes=[indexesFromSentence(voc, sentence) for sentence in inp]

indexes
def zeroPadding(l, fillvalue=PAD_token):

    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

leng=[len(inp) for inp in indexes]

max(leng)
#Test the function

test_result=zeroPadding(indexes)

print(len(test_result))

test_result
#convert to binary

def binaryMatrix(l, value=PAD_token):

    m = []

    for i, seq in enumerate(l):

        m.append([])

        for token in seq:

            if token == PAD_token:

                m[i].append(0)

            else:

                m[i].append(1)

    return m
#test

binary_result=binaryMatrix(test_result)

binary_result
# Returns padded input sequence tensor and lengths

def inputVar(l, voc):

    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)

    padVar = torch.LongTensor(padList)

    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length

def outputVar(l, voc):

    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    max_target_len = max([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)

    mask = binaryMatrix(padList)

    mask = torch.ByteTensor(mask)

    padVar = torch.LongTensor(padList)

    return padVar, mask, max_target_len
# Returns all items for a given batch of pairs

def batch2TrainData(voc, pair_batch):

    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)

    input_batch, output_batch = [], []

    for pair in pair_batch:

        input_batch.append(pair[0])

        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch, voc)

    output, mask, max_target_len = outputVar(output_batch, voc)

    return inp, lengths, output, mask, max_target_len



# Example for validation

small_batch_size = 5

batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])

input_variable, lengths, target_variable, mask, max_target_len = batches



print("input_variable:", input_variable)

print("lengths:", lengths)

print("target_variable:", target_variable)

print("mask:", mask)

print("max_target_len:", max_target_len)
class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):

        super(EncoderRNN, self).__init__()

        self.n_layers = n_layers

        self.hidden_size = hidden_size

        self.embedding = embedding



        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'

        #   because our input size is a word embedding with number of features == hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,

                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)



    def forward(self, input_seq, input_lengths, hidden=None):

        # Convert word indexes to embeddings

        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # Forward pass through GRU

        outputs, hidden = self.gru(packed, hidden)

        # Unpack padding

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs

        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Return output and final hidden state

        return outputs, hidden
# Luong attention layer

class Attn(torch.nn.Module):

    def __init__(self, method, hidden_size):

        super(Attn, self).__init__()

        self.method = method

        if self.method not in ['dot', 'general', 'concat']:

            raise ValueError(self.method, "is not an appropriate attention method.")

        self.hidden_size = hidden_size

        if self.method == 'general':

            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':

            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)

            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))



    def dot_score(self, hidden, encoder_output):

        return torch.sum(hidden * encoder_output, dim=2)



    def general_score(self, hidden, encoder_output):

        energy = self.attn(encoder_output)

        return torch.sum(hidden * energy, dim=2)



    def concat_score(self, hidden, encoder_output):

        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()

        return torch.sum(self.v * energy, dim=2)



    def forward(self, hidden, encoder_outputs):

        # Calculate the attention weights (energies) based on the given method

        if self.method == 'general':

            attn_energies = self.general_score(hidden, encoder_outputs)

        elif self.method == 'concat':

            attn_energies = self.concat_score(hidden, encoder_outputs)

        elif self.method == 'dot':

            attn_energies = self.dot_score(hidden, encoder_outputs)



        # Transpose max_length and batch_size dimensions

        attn_energies = attn_energies.t()



        # Return the softmax normalized probability scores (with added dimension)

        return F.softmax(attn_energies, dim=1).unsqueeze(1)
import torch.nn.functional as F

a=torch.rand(5,7)

a
b=F.softmax(a,dim=1)

b
b[0].sum()
class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):

        super(LuongAttnDecoderRNN, self).__init__()



        # Keep for reference

        self.attn_model = attn_model

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.n_layers = n_layers

        self.dropout = dropout



        # Define layers

        self.embedding = embedding

        self.embedding_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))

        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)



        self.attn = Attn(attn_model, hidden_size)



    def forward(self, input_step, last_hidden, encoder_outputs):

        # Note: we run this one step (word) at a time

        # Get embedding of current input word

        embedded = self.embedding(input_step)

        embedded = self.embedding_dropout(embedded)

        # Forward through unidirectional GRU

        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention weights from the current GRU output

        attn_weights = self.attn(rnn_output, encoder_outputs)

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate weighted context vector and GRU output using Luong eq. 5

        rnn_output = rnn_output.squeeze(0)

        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)

        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word using Luong eq. 6

        output = self.out(concat_output)

        output = F.softmax(output, dim=1)

        # Return output and final hidden state

        return output, hidden
def maskNLLLoss(decoder_out, target, mask):

    nTotal = mask.sum() #how many element should be consider

    target= target.view(-1, 1)

    #decode_out shape: (batch_size,vocab_size),target_size=(batch_size,1)

    gathered_tensor=torch.gather(decoder_out, 1, target)

    # calculate the negative  log likelihood loss

    crossEntropy = -torch.log(gathered_tensor)

    #select the non_zero elements

    loss = crossEntropy.masked_select(mask)

    #calculate the mean of loss

    loss=loss.mean()

    loss = loss.to(device)

    return loss, nTotal.item()
#visulizeing what's happenning in one iteration , only run this for visulizing

small_batch_size = 5

batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])

input_variable, lengths, target_variable, mask, max_target_len = batches



print("input_variable:", input_variable.shape)

print("lengths:", lengths.shape)

print("target_variable:", target_variable.shape)

print("mask:", mask.shape)

print("max_target_len:", max_target_len)



# define the parameters

attn_model = 'dot'

hidden_size = 500

encoder_n_layers = 2

decoder_n_layers = 2

dropout = 0.1

embedding = nn.Embedding(voc.num_words, hidden_size)



print('Building encoder and decoder ...')

#Define the encoder and decoder 

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)

decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

# Use appropriate device

encoder = encoder.to(device)

decoder = decoder.to(device)

print('Models built and ready to go!')

# Ensure dropout layers are in train mode

encoder.train()

decoder.train()



# Initialize optimizers

print('Building optimizers ...')

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)

decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)

encoder_optimizer.zero_grad()

decoder_optimizer.zero_grad()



# Set device options

input_variable = input_variable.to(device)

lengths = lengths.to(device)

target_variable = target_variable.to(device)

mask = mask.to(device)



# Initialize variables

loss = 0

print_losses = []

n_totals = 0



# Forward pass through encoder

encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

print("encoder_outputs shape",encoder_outputs.shape)

print("last encoder_hidden shape",encoder_hidden.shape)



 # Create initial decoder input (start with SOS tokens for each sentence)

decoder_input = torch.LongTensor([[SOS_token for _ in range(small_batch_size)]])

decoder_input = decoder_input.to(device)

print("initilize decoder_input shape",decoder_input.shape)

print(decoder_input)



# Set initial decoder hidden state to the encoder's final hidden state

decoder_hidden = encoder_hidden[:decoder.n_layers]

print("initial decoder hidden state shape",decoder_hidden.shape)

print("\n")

print("---------------------------------------------------------")

print("Now let's look what's happinig in every timestep of the GPU!")

print("---------------------------------------------------------")

print("\n")





for t in range(max_target_len):

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        print("decoder_output shape",decoder_output.shape)

        print("encoder_hidden shape",encoder_hidden.shape)

        # Teacher forcing: next input is current target

        decoder_input = target_variable[t].view(1, -1)

        print("The target variable at the current timestep before reshaping",target_variable[t])

        print("The target variable at the current timestep shape before reshaping",target_variable[t].shape)

        print("The decoder_input shape(reshape target_variable)",decoder_input.shape)

        # Calculate and accumulate loss

        print("The mask at the current timestep : ",mask[t] )

        print("The mask at the current timestep shape: ",mask[t].shape )

        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

        print("Mask loss ",mask_loss)

        print("Total ",nTotal)

        loss += mask_loss

        print_losses.append(mask_loss.item() * nTotal)

        print(print_losses)

        n_totals += nTotal

        print(n_totals)

        encoder_optimizer.step()

        decoder_optimizer.step()

        returned_loss=sum(print_losses)/ n_totals

        print("returned_loss : ",returned_loss)

        print("\n")

        print("------------------- Done one timestep ----------------------------")

        print("\n")
random.random()
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,

          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):



    # Zero gradients

    encoder_optimizer.zero_grad()

    decoder_optimizer.zero_grad()



    # Set device options

    input_variable = input_variable.to(device)

    lengths = lengths.to(device)

    target_variable = target_variable.to(device)

    mask = mask.to(device)



    # Initialize variables

    loss = 0

    print_losses = []

    n_totals = 0



    # Forward pass through encoder

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)



    # Create initial decoder input (start with SOS tokens for each sentence)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])

    decoder_input = decoder_input.to(device)



    # Set initial decoder hidden state to the encoder's final hidden state

    decoder_hidden = encoder_hidden[:decoder.n_layers]



    # Determine if we are using teacher forcing this iteration

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False



    # Forward batch of sequences through decoder one time step at a time

    if use_teacher_forcing:

        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoder(

                decoder_input, decoder_hidden, encoder_outputs

            )

            # Teacher forcing: next input is current target

            decoder_input = target_variable[t].view(1, -1)

            # Calculate and accumulate loss

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

            loss += mask_loss

            print_losses.append(mask_loss.item() * nTotal)

            n_totals += nTotal

    else:

        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoder(

                decoder_input, decoder_hidden, encoder_outputs

            )

            # No teacher forcing: next input is decoder's own current output

            _, topi = decoder_output.topk(1)

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

            decoder_input = decoder_input.to(device)

            # Calculate and accumulate loss

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

            loss += mask_loss

            print_losses.append(mask_loss.item() * nTotal)

            n_totals += nTotal



    # Perform backpropatation

    loss.backward()



    # Clip gradients: gradients are modified in place

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)



    # Adjust model weights

    encoder_optimizer.step()

    decoder_optimizer.step()



    return sum(print_losses) / n_totals
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):



    # Load batches for each iteration

    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])

                      for _ in range(n_iteration)]



    # Initializations

    print('Initializing ...')

    start_iteration = 1

    print_loss = 0

    if loadFilename:

        start_iteration = checkpoint['iteration'] + 1



    # Training loop

    print("Training...")

    for iteration in range(start_iteration, n_iteration + 1):

        training_batch = training_batches[iteration - 1]

        # Extract fields from batch

        input_variable, lengths, target_variable, mask, max_target_len = training_batch



        # Run a training iteration with batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,

                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)

        print_loss += loss



        # Print progress

        if iteration % print_every == 0:

            print_loss_avg = print_loss / print_every

            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))

            print_loss = 0



        # Save checkpoint

        if (iteration % save_every == 0):

            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))

            if not os.path.exists(directory):

                os.makedirs(directory)

            torch.save({

                'iteration': iteration,

                'en': encoder.state_dict(),

                'de': decoder.state_dict(),

                'en_opt': encoder_optimizer.state_dict(),

                'de_opt': decoder_optimizer.state_dict(),

                'loss': loss,

                'voc_dict': voc.__dict__,

                'embedding': embedding.state_dict()

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))