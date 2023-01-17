!pip install pyro-ppl -q;
import time
import math
import pandas as pd
from torch.distributions import kl_divergence
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.fclv = nn.Linear(hidden_size, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)

    def forward(self, inputs):
        h1 = F.softplus(self.fc1(inputs))
        h2 = F.softplus(self.fc2(h1))
        mu = self.bnmu(self.fcmu(h2))
        lv = self.bnlv(self.fclv(h2))
        dist = LogNormal(mu, (0.5 * lv).exp())
        return dist


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.fc = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        return F.log_softmax(self.bn(self.fc(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics, dropout):
        super().__init__()
        self.encode = Encoder(vocab_size, hidden_size, num_topics, dropout)
        self.decode = Decoder(vocab_size, num_topics, dropout)

    def forward(self, inputs):
        posterior = self.encode(inputs)
        if self.training:
            t = posterior.rsample().to(inputs.device)
        else:
            t = posterior.mean.to(inputs.device)
        t = t / t.sum(1, keepdim=True)
        outputs = self.decode(t)
        return outputs, posterior
def recon_loss(targets, outputs):
    nll = - torch.sum(targets * outputs)
    return nll


def standard_prior_like(posterior):
    loc = torch.zeros_like(posterior.loc)
    scale = torch.ones_like(posterior.scale)
    prior = LogNormal(loc, scale)
    return prior


def get_loss(inputs, model, device):
    inputs = inputs.to(device)
    outputs, posterior = model(inputs)
    prior = standard_prior_like(posterior)
    nll = recon_loss(inputs, outputs)
    kld = torch.sum(kl_divergence(posterior, prior).to(device))
    return nll, kld


def train(data_source, model, optimizer, epoch, device):
    model.train()
    total_nll = 0.0
    total_kld = 0.0
    total_words = 0

    num_batches = math.ceil(data_source.shape[0] / batch_size)
    bar = range(num_batches)
    for i in bar:
        data = data_source[i * batch_size:(i + 1) * batch_size, :]
        nll, kld = get_loss(data, model, device)
        total_nll += nll.item() / data.shape[0]
        total_kld += kld.item() / data.shape[0]
        total_words += data.sum()
        optimizer.zero_grad()
        loss = nll + kld
        loss.backward()
        optimizer.step()

    # total_nll /= data_source.shape[0]
    # total_kld /= data_source.shape[0]
        
    ppl = math.exp(total_nll / total_words)
    return (total_nll, total_kld, ppl)
def main():
    print("Loading data")

    corpus = torch.load('../input/prodlda/doc_sum_20newsgroup.pt').float()
    vocab_size = corpus.shape[1]
    print("\ttraining data size: ", corpus.shape[0])
    print("\tvocabulary size: ", vocab_size)
    print("Constructing model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ProdLDA(
        vocab_size, hidden_size, num_topics, dropout).to(device)

    corpus = corpus.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_loss = None

    print("\nStart training")
    try:
        bar = trange(epochs)
        for epoch in bar:
            epoch_start_time = time.time()
            train_nll, train_kld, train_ppl = train(corpus, model, optimizer, epoch, device)
            bar.set_postfix(time=time.time()-epoch_start_time, loss=train_nll, kld=train_kld, ppl=train_ppl)

    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early')

    beta = model.decode.fc.weight.cpu().detach().T
    torch.save(beta, 'betas.pt')
    return beta
torch.cuda.is_available()
hidden_size = 100
num_topics = 20
dropout = 0.2
epochs = 50
batch_size = 32
lr = 1e-3
wd = 0

beta = main()
vocab = pd.read_csv('../input/prodlda/vocab_20newsgroup.csv')

for i in range(beta.shape[0]):
    sorted_, indices = torch.sort(beta[i], descending=True)
    df = pd.DataFrame(indices[:20].numpy(), columns=['index'])
    print(pd.merge(df, vocab[['index', 'word']], how='left', on='index')['word'].values)
    print()
