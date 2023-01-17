!pip install pyro-ppl -q
import math
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from tqdm import trange
class Encoder(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        theta_loc = self.bnmu(self.fcmu(h))
        theta_scale = self.bnlv(self.fclv(h))
        return theta_loc, theta_scale


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        return F.log_softmax(self.bn(self.beta(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.inference_net = Encoder(vocab_size, num_topics, hidden, dropout)
        self.recognition_net = Decoder(vocab_size, num_topics, dropout)
        self.device = device

    def model(self, doc_sum=None):
        # register PyTorch module `decoder` with Pyro
        pyro.module("recognition_net", self.recognition_net)
        with pyro.plate("documents", doc_sum.shape[0]):
            # setup hyperparameters
            theta_loc = doc_sum.new_zeros((doc_sum.shape[0], self.num_topics))
            theta_scale = doc_sum.new_ones((doc_sum.shape[0], self.num_topics))
            # sample from prior (value will be sampled by guide
            # when computing the ELBO)
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, (0.5 * theta_scale).exp()).to_event(1))
            theta = theta / theta.sum(1, keepdim=True)

            count_param = self.recognition_net(theta)
            pyro.sample(
                'obs',
                dist.Multinomial(doc_sum.shape[1], count_param).to_event(1),
                obs=doc_sum
            )

    def guide(self, doc_sum=None):
        # Use an amortized guide for local variables.
        pyro.module("inference_net", self.inference_net)
        with pyro.plate("documents", doc_sum.shape[0]):
            theta_loc, theta_scale = self.inference_net(doc_sum)
            pyro.sample(
                "theta", dist.LogNormal(theta_loc, (0.5 * theta_scale).exp()).to_event(1))
            
    def beta(self):
        return self.recognition_net.beta.weight.cpu().detach().T
def train(device, doc_sum, batch_size, learning_rate, num_epochs):
    # clear param store
    pyro.clear_param_store()

    prodLDA = ProdLDA(
        vocab_size=doc_sum.shape[1],
        num_topics=100,
        hidden=100,
        dropout=0.2,
        device=device
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=Trace_ELBO())
    num_batches = int(math.ceil(doc_sum.shape[0] / batch_size))

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        num_preds = 0

        # Iterate over data.
        for i in range(num_batches):
            batch_doc_sum = doc_sum[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_doc_sum)
            running_loss += loss / batch_doc_sum.size(0)

        epoch_loss = running_loss / doc_sum.shape[0]
        bar.set_postfix(epoch_loss='{:.2f}'.format(epoch_loss))

    return prodLDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
doc_sum = torch.load('../input/prodlda/doc_sum_ap.pt').float().to(device)
trained_model = train(device, doc_sum, 32, 1e-3, 60)

beta = trained_model.beta()
torch.save(beta, 'betas.pt')
vocab = pd.read_csv('../input/prodlda/vocab.csv')

for i in range(beta.shape[0]):
    sorted_, indices = torch.sort(beta[i], descending=True)
    df = pd.DataFrame(indices[:20].numpy(), columns=['index'])
    print(pd.merge(df, vocab[['index', 'word']], how='left', on='index')['word'].values)
    print()
