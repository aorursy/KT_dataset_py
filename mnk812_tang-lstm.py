!pip install torchnet -q
import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torchnet import meter
class Config(object):

    num_layers = 3  # LSTM层数

    lr = 1e-3

    weight_decay = 1e-4

    epochs = 3

    batch_size = 16

    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格

    max_gen_len = 125  # 生成诗歌最长长度

    embedding_dim = 128

    hidden_dim = 256

    start_words = '湖光秋月两相和'  # 唐诗的第一句
def prepareData():

    datas = np.load("../input/tangshi/tang.npz", allow_pickle=True)

    data = datas['data']

    ix2word = datas['ix2word'].item()

    word2ix = datas['word2ix'].item()

    

    data = torch.from_numpy(data)

    dataloader = DataLoader(data,

                         batch_size = Config.batch_size,

                         shuffle = True,

                         num_workers = 2)

    

    return dataloader, ix2word, word2ix
dataloader, ix2word, word2ix = prepareData()
class PoetryModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):

        super(PoetryModel, self).__init__()

        self.hidden_dim = hidden_dim

        # 词向量层，词表大小 * 向量维度

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 网络主要结构

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.num_layers)

        # 进行分类

        self.linear = nn.Linear(self.hidden_dim, vocab_size)



    def forward(self, input, hidden=None):

        seq_len, batch_size = input.size()

        if hidden is None:

            h_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()

            c_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()

        else:

            h_0, c_0 = hidden

        # 输入 序列长度 * batch(每个汉字是一个数字下标)，

        # 输出 序列长度 * batch * 向量维度

        embeds = self.embeddings(input)

        # 输出hidden的大小： 序列长度 * batch * hidden_dim

        output, hidden = self.lstm(embeds, (h_0, c_0))

        output = self.linear(output.view(seq_len * batch_size, -1))

        return output, hidden
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = PoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim)

optimizer = torch.optim.Adam(model.parameters(),lr=Config.lr)

criterion = nn.CrossEntropyLoss()

model.to(device)

loss_meter = meter.AverageValueMeter()
for epoch in range(Config.epochs):

    loss_meter.reset()

    for batch_idx,data_ in enumerate(dataloader):

        data_ = data_.long().transpose(1,0).contiguous()

        data_ = data_.to(device)

        input_,target = data_[:-1,:],data_[1:,:]

        model.zero_grad()

        output,_ = model(input_)

        loss = criterion(output,target.view(-1))

        

        if batch_idx % 900 == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                        epoch+1, batch_idx * len(data_[1]), len(dataloader.dataset),

                        100. * batch_idx / len(dataloader), loss.item()))

        loss.backward()

        optimizer.step()

        loss_meter.add(loss.item())

        

torch.save(model.state_dict(), 'model.pth')
def generate(start_words, ix2word, word2ix):



    model = PoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim)

    model.load_state_dict(torch.load('model.pth'))

    model.to(device)

    

    results = list(start_words)

    start_word_len = len(start_words)

    

    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()

    input = input.to(device)

    hidden = None



    for i in range(Config.max_gen_len):

        output, hidden = model(input, hidden)

        if i < start_word_len:

            w = results[i]

            input = input.data.new([word2ix[w]]).view(1, 1)

        else:

            top_index = output.data[0].topk(1)[1][0].item()

            w = ix2word[top_index]

            results.append(w)

            input = input.data.new([top_index]).view(1, 1)

        if w == '<EOP>':

            del results[-1]

            break

            

    return results
results = generate(Config.start_words, ix2word, word2ix)

print(results)
#藏头诗

def gen_acrostic(start_words, ix2word, word2ix):



    model = PoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim)

    model.load_state_dict(torch.load('model.pth'))

    model.to(device)

   

    results = []

    start_word_len = len(start_words)

    

    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long())

    input = input.to(device)

    hidden = None



    index = 0            # 指示已生成了多少句

    pre_word = '<START>' # 上一个词



    for i in range(Config.max_gen_len):

        output, hidden = model(input, hidden)

        top_index = output.data[0].topk(1)[1][0].item()

        w = ix2word[top_index]



        if (pre_word in {u'。', u'！', '<START>'}):

            if index == start_word_len:

                break

            else:

                w = start_words[index]

                index += 1

                input = (input.data.new([word2ix[w]])).view(1, 1)



        else:

            input = (input.data.new([word2ix[w]])).view(1, 1)

        results.append(w)

        pre_word = w

        

    return results
results_acrostic = gen_acrostic(Config.start_words, ix2word, word2ix)

print(results_acrostic)