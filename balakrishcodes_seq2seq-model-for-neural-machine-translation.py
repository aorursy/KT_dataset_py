!pip install torchtext==0.6.0 --quiet

import torch

import torch.nn as nn

import torch.optim as optim

from torchtext.datasets import Multi30k

from torchtext.data import Field, BucketIterator

import numpy as np

import pandas as pd

import spacy

import random

from torchtext.data.metrics import bleu_score

from pprint import pprint

from torch.utils.tensorboard import SummaryWriter



# Seeding for reproducible results everytime

'''

SEED = 777



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True'''
!python -m spacy download en --quiet

!python -m spacy download de --quiet
spacy_german = spacy.load("de")

spacy_english = spacy.load("en")
def tokenize_german(text):

    return [token.text for token in spacy_german.tokenizer(text)]



def tokenize_english(text):

    return [token.text for token in spacy_english.tokenizer(text)]



### Sample Run ###



sample_text = "I love machine learning"

print(tokenize_english(sample_text))
german = Field(tokenize=tokenize_german,

               lower=True,

               init_token="<sos>",

               eos_token="<eos>")



english = Field(tokenize=tokenize_english,

               lower=True,

               init_token="<sos>",

               eos_token="<eos>")



train_data, valid_data, test_data = Multi30k.splits(exts = (".de", ".en"),

                                                    fields=(german, english))



german.build_vocab(train_data, max_size=10000, min_freq=3)

english.build_vocab(train_data, max_size=10000, min_freq=3)
print(f"Unique tokens in source (de) vocabulary: {len(german.vocab)}")

print(f"Unique tokens in target (en) vocabulary: {len(english.vocab)}")
# dir(english.vocab)

e = list(english.vocab.__dict__.values())



'''

print(english.vocab.__dict__.keys())

print(list(english.vocab.__dict__.values()))

e = list(english.vocab.__dict__.values())

for i in e:

  print(i)'''
word_2_idx = dict(e[3])

idx_2_word = {}

for k,v in word_2_idx.items():

      idx_2_word[v] = k
print(f"Number of training examples: {len(train_data.examples)}")

print(f"Number of validation examples: {len(valid_data.examples)}")

print(f"Number of testing examples: {len(test_data.examples)}")



print(train_data[5].__dict__.keys())

pprint(train_data[5].__dict__.values())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32



train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 

                                                                      batch_size = BATCH_SIZE, 

                                                                      sort_within_batch=True,

                                                                      sort_key=lambda x: len(x.src),

                                                                      device = device)
count = 0

max_len_eng = []

max_len_ger = []

for data in train_data:

      max_len_ger.append(len(data.src))

      max_len_eng.append(len(data.trg))

      if count < 10 :

        print("German - ",*data.src, " Length - ", len(data.src))

        print("English - ",*data.trg, " Length - ", len(data.trg))

        print()

        count += 1



print("Maximum Length of English sentence {} and German sentence {} in the dataset".format(max(max_len_eng),max(max_len_ger)))

print("Minimum Length of English sentence {} and German sentence {} in the dataset".format(min(max_len_eng),min(max_len_ger)))
count = 0

for data in train_iterator:

  if count < 1 :

    print("Shapes", data.src.shape, data.trg.shape)

    print()

    print("German - ",*data.src, " Length - ", len(data.src))

    print()

    print("English - ",*data.trg, " Length - ", len(data.trg))

    temp_ger = data.src

    temp_eng = data.trg

    count += 1
temp_eng_idx = (temp_eng).cpu().detach().numpy()

temp_ger_idx = (temp_ger).cpu().detach().numpy()
df_eng_idx = pd.DataFrame(data = temp_eng_idx, columns = [str("S_")+str(x) for x in np.arange(1, 33)])

df_eng_idx.index.name = 'Time Steps'

df_eng_idx.index = df_eng_idx.index + 1 

# df_eng_idx.to_csv('/content/idx.csv')

df_eng_idx
df_eng_word = pd.DataFrame(columns = [str("S_")+str(x) for x in np.arange(1, 33)])

df_eng_word = df_eng_idx.replace(idx_2_word)

# df_eng_word.to_csv('/content/Words.csv')

df_eng_word
class EncoderLSTM(nn.Module):

  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):

    super(EncoderLSTM, self).__init__()



    # Size of the one hot vectors that will be the input to the encoder

    #self.input_size = input_size



    # Output size of the word embedding NN

    #self.embedding_size = embedding_size



    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.

    self.hidden_size = hidden_size



    # Number of layers in the lstm

    self.num_layers = num_layers



    # Regularization parameter

    self.dropout = nn.Dropout(p)

    self.tag = True



    # Shape --------------------> (5376, 300) [input size, embedding dims]

    self.embedding = nn.Embedding(input_size, embedding_size)

    

    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]

    self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)



  # Shape of x (26, 32) [Sequence_length, batch_size]

  def forward(self, x):



    # Shape -----------> (26, 32, 300) [Sequence_length , batch_size , embedding dims]

    embedding = self.dropout(self.embedding(x))

    

    # Shape --> outputs (26, 32, 1024) [Sequence_length , batch_size , hidden_size]

    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size]

    outputs, (hidden_state, cell_state) = self.LSTM(embedding)



    return hidden_state, cell_state



input_size_encoder = len(german.vocab)

encoder_embedding_size = 300

hidden_size = 1024

num_layers = 2

encoder_dropout = 0.5



encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,

                           hidden_size, num_layers, encoder_dropout).to(device)

print(encoder_lstm)
class DecoderLSTM(nn.Module):

  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, output_size):

    super(DecoderLSTM, self).__init__()



    # Size of the one hot vectors that will be the input to the encoder

    #self.input_size = input_size



    # Output size of the word embedding NN

    #self.embedding_size = embedding_size



    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.

    self.hidden_size = hidden_size



    # Number of layers in the lstm

    self.num_layers = num_layers



    # Size of the one hot vectors that will be the output to the encoder (English Vocab Size)

    self.output_size = output_size



    # Regularization parameter

    self.dropout = nn.Dropout(p)



    # Shape --------------------> (5376, 300) [input size, embedding dims]

    self.embedding = nn.Embedding(input_size, embedding_size)



    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]

    self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)



    # Shape -----------> (1024, 4556) [embedding dims, hidden size, num layers]

    self.fc = nn.Linear(hidden_size, output_size)



  # Shape of x (32) [batch_size]

  def forward(self, x, hidden_state, cell_state):



    # Shape of x (1, 32) [1, batch_size]

    x = x.unsqueeze(0)



    # Shape -----------> (1, 32, 300) [1, batch_size, embedding dims]

    embedding = self.dropout(self.embedding(x))



    # Shape --> outputs (1, 32, 1024) [1, batch_size , hidden_size]

    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size] (passing encoder's hs, cs - context vectors)

    outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))



    # Shape --> predictions (1, 32, 4556) [ 1, batch_size , output_size]

    predictions = self.fc(outputs)



    # Shape --> predictions (32, 4556) [batch_size , output_size]

    predictions = predictions.squeeze(0)



    return predictions, hidden_state, cell_state



input_size_decoder = len(english.vocab)

decoder_embedding_size = 300

hidden_size = 1024

num_layers = 2

decoder_dropout = 0.5

output_size = len(english.vocab)



decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,

                           hidden_size, num_layers, decoder_dropout, output_size).to(device)

print(decoder_lstm)
for batch in train_iterator:

  print(batch.src.shape)

  print(batch.trg.shape)

  break



x = batch.trg[1]

print(x)
class Seq2Seq(nn.Module):

  def __init__(self, Encoder_LSTM, Decoder_LSTM):

    super(Seq2Seq, self).__init__()

    self.Encoder_LSTM = Encoder_LSTM

    self.Decoder_LSTM = Decoder_LSTM



  def forward(self, source, target, tfr=0.5):

    # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]

    batch_size = source.shape[1]



    # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]

    target_len = target.shape[0]

    target_vocab_size = len(english.vocab)

    

    # Shape --> outputs (14, 32, 5766) 

    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)



    # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)

    hidden_state, cell_state = self.Encoder_LSTM(source)



    # Shape of x (32 elements)

    x = target[0] # Trigger token <SOS>



    for i in range(1, target_len):

      # Shape --> output (32, 5766) 

      output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)

      outputs[i] = output

      best_guess = output.argmax(1) # 0th dimension is batch size, 1st dimension is word embedding

      x = target[i] if random.random() < tfr else best_guess # Either pass the next word correctly from the dataset or use the earlier predicted word



    # Shape --> outputs (14, 32, 5766) 

    return outputs

# Hyperparameters



learning_rate = 0.001

writer = SummaryWriter(f"runs/loss_plot")

step = 0



model = Seq2Seq(encoder_lstm, decoder_lstm).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



pad_idx = english.vocab.stoi["<pad>"]

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
model
def translate_sentence(model, sentence, german, english, device, max_length=50):

    spacy_ger = spacy.load("de")



    if type(sentence) == str:

        tokens = [token.text.lower() for token in spacy_ger(sentence)]

    else:

        tokens = [token.lower() for token in sentence]

    tokens.insert(0, german.init_token)

    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)



    # Build encoder hidden, cell state

    with torch.no_grad():

        hidden, cell = model.Encoder_LSTM(sentence_tensor)



    outputs = [english.vocab.stoi["<sos>"]]



    for _ in range(max_length):

        previous_word = torch.LongTensor([outputs[-1]]).to(device)



        with torch.no_grad():

            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)

            best_guess = output.argmax(1).item()



        outputs.append(best_guess)



        # Model predicts it's the end of the sentence

        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:

            break



    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    return translated_sentence[1:]



def bleu(data, model, german, english, device):

    targets = []

    outputs = []



    for example in data:

        src = vars(example)["src"]

        trg = vars(example)["trg"]



        prediction = translate_sentence(model, src, german, english, device)

        prediction = prediction[:-1]  # remove <eos> token



        targets.append([trg])

        outputs.append(prediction)



    return bleu_score(outputs, targets)



def checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss):

    print('saving')

    print()

    state = {'model': model,'best_loss': best_loss,'epoch': epoch,'rng_state': torch.get_rng_state(), 'optimizer': optimizer.state_dict(),}

    torch.save(state, '/kaggle/working/checkpoint-NMT')

    torch.save(model.state_dict(),'/kaggle/working/checkpoint-NMT-SD')
epoch_loss = 0.0

num_epochs = 100

best_loss = 999999

best_epoch = -1

sentence1 = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster"

ts1  = []



for epoch in range(num_epochs):

  print("Epoch - {} / {}".format(epoch+1, num_epochs))

  model.eval()

  translated_sentence1 = translate_sentence(model, sentence1, german, english, device, max_length=50)

  print(f"Translated example sentence 1: \n {translated_sentence1}")

  ts1.append(translated_sentence1)



  model.train(True)

  for batch_idx, batch in enumerate(train_iterator):

    input = batch.src.to(device)

    target = batch.trg.to(device)



    # Pass the input and target for model's forward method

    output = model(input, target)

    output = output[1:].reshape(-1, output.shape[2])

    target = target[1:].reshape(-1)



    # Clear the accumulating gradients

    optimizer.zero_grad()



    # Calculate the loss value for every epoch

    loss = criterion(output, target)



    # Calculate the gradients for weights & biases using back-propagation

    loss.backward()



    # Clip the gradient value is it exceeds > 1

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)



    # Update the weights values using the gradients we calculated using bp 

    optimizer.step()

    step += 1

    epoch_loss += loss.item()

    writer.add_scalar("Training loss", loss, global_step=step)



  if epoch_loss < best_loss:

    best_loss = epoch_loss

    best_epoch = epoch

    checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss) 

    if ((epoch - best_epoch) >= 10):

      print("no improvement in 10 epochs, break")

      break

  print("Epoch_Loss - {}".format(loss.item()))

  print()

  

print(epoch_loss / len(train_iterator))



score = bleu(test_data[1:100], model, german, english, device)

print(f"Bleu score {score*100:.2f}")
progress  = []

import nltk

from nltk.tokenize.treebank import TreebankWordDetokenizer

for i, sen in enumerate(ts1):

  progress.append(TreebankWordDetokenizer().detokenize(sen))

print(progress)



progress_df = pd.DataFrame(data = progress, columns=['Predicted Sentence'])

progress_df.index.name = "Epochs"

progress_df.head()
model.eval()

test_sentences  = ["Zwei Männer gehen die Straße entlang", "Kinder spielen im Park.", "Diese Stadt verdient eine bessere Klasse von Verbrechern. Der Spaßvogel"]

actual_sentences  = ["Two men are walking down the street", "Children play in the park", "This city deserves a better class of criminals. The joker"]

pred_sentences = []



for idx, i in enumerate(test_sentences):

  model.eval()

  translated_sentence = translate_sentence(model, i, german, english, device, max_length=50)

  progress.append(TreebankWordDetokenizer().detokenize(translated_sentence))

  print("German : {}".format(i))

  print("Actual Sentence in English : {}".format(actual_sentences[idx]))

  print("Predicted Sentence in English : {}".format(progress[-1]))

  print()
