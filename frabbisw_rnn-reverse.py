import torch.nn as nn

import torch
import string

import random
all_letters = string.ascii_lowercase+"#"

n_letters = len(all_letters)
def random_string(r=10):

    x = ''.join([random.choice(all_letters) for i in range(r)])

    y = x[::-1]    

    return x,y
def letterToIndex(letter):

    return all_letters.find(letter)
def lineToTensor(line):

    alph = [torch.tensor([letterToIndex(a)]).long().cuda() for a in line]

    return alph
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size



        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)



    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)

        output = embedded

        output, hidden = self.gru(output, hidden)

        return output, hidden

    

    def init_hidden(self):

        return torch.zeros(1, 1, hidden_size).cuda()
class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):

        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size



        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)

        output = torch.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden

    

    def init_hidden(self):

        return torch.zeros(1, 1, hidden_size).cuda()
input_size=n_letters

output_size=n_letters

hidden_size=128
encoder=EncoderRNN(input_size,hidden_size).cuda()

decoder=DecoderRNN(hidden_size,output_size).cuda()
criterion = nn.NLLLoss()

optim_encoder = torch.optim.Adam(encoder.parameters(),lr=.0001)

optim_decoder = torch.optim.Adam(decoder.parameters(),lr=.0001)
SOS=torch.Tensor([letterToIndex('#')]).long().cuda()
def train(sx,sy):

    x=lineToTensor(sx)

    y=lineToTensor(sy)



    encoder_hidden=encoder.init_hidden()

    optim_encoder.zero_grad()

    optim_encoder.zero_grad()



    loss=0

    for i in range(len(x)):

        encoder_output, encoder_hidden = encoder(x[i],encoder_hidden)

    

    decoder_hidden = encoder_hidden

    decoder_input = SOS

    

    for i in range(len(y)):

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        decoder_input = y[i]

        loss += criterion(decoder_output,y[i])

    

    loss.backward()

    

    optim_decoder.step()

    optim_encoder.step()

    

    return loss
sx,sy = random_string(6)
def evaluate(sx,sy):

    predict_output = []

    x=lineToTensor(sx)

    y=lineToTensor(sy)



    for i in range(len(x)):

        encoder_hidden=encoder.init_hidden()

        encoder_output, encoder_hidden = encoder(x[i],encoder_hidden)



        decoder_hidden = encoder_hidden

        decoder_input = SOS



    for i in range(len(y)):

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        decoder_input = y[i]



        topv, topi = decoder_output.data.topk(1)

        predict_output.append(all_letters[topi.view(-1).cpu().numpy()[0]])

        #print(all_letters[topi.view(-1).cpu().numpy()[0]])



    predict_output = ''.join(predict_output)

    

    print("word:\t\t",sx,"\nreverse:\t",sy,"\npredicted:\t",predict_output)
r=50000

for i in range(r):

    sx,sy = random_string(6)

    #x=lineToTensor(sx)

    #y=lineToTensor(sy)



    loss = train(sx,sy)

    

    if i % 1000 == 0:

        print(i," of ",r,loss)

        evaluate(sx,sy)
sx,sy = random_string(6)

evaluate(sx,sy)