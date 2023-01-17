!pip install pyro-ppl -q
import math
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceMeanField_ELBO
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
        theta_scale = (0.5 * theta_scale).exp()  # Enforces positivity
        return theta_loc, theta_scale


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.inference_net = Encoder(vocab_size, num_topics, hidden, dropout)
        self.recognition_net = Decoder(vocab_size, num_topics, dropout)

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
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
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
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))  # (0.5 * ).exp()
            
    def beta(self):
        return self.recognition_net.beta.weight.cpu().detach().T
def train(device, num_topics, doc_sum, batch_size, learning_rate, num_epochs):
    # clear param store
    pyro.clear_param_store()

    prodLDA = ProdLDA(
        vocab_size=doc_sum.shape[1],
        num_topics=num_topics,
        hidden=100,
        dropout=0.2
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO(num_particles=1))
    num_batches = int(math.ceil(doc_sum.shape[0] / batch_size))

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        total_words = 0

        # Iterate over data.
        for i in range(num_batches):
            batch_doc_sum = doc_sum[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_doc_sum)
            running_loss += loss / batch_doc_sum.size(0)
            total_words += batch_doc_sum.sum()

        epoch_loss = running_loss #/ doc_sum.shape[0]
        ppl = math.exp(epoch_loss / total_words)
        bar.set_postfix(epoch_loss='{:.2e}'.format(epoch_loss), ppl=ppl)

    return prodLDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
seed = 101
torch.manual_seed(seed)
pyro.set_rng_seed(seed)

num_topics = 20  # 
doc_sum = torch.load('../input/prodlda/doc_sum_20newsgroup.pt').float().to(device)  # 
trained_model = train(device, num_topics, doc_sum, 32, 1e-3, 50)  # 50

beta = trained_model.beta()
torch.save(beta, 'betas.pt')
vocab = pd.read_csv('../input/prodlda/vocab_20newsgroup.csv')  # 

for i in range(beta.shape[0]):
    sorted_, indices = torch.sort(beta[i], descending=True)
    df = pd.DataFrame(indices[:20].numpy(), columns=['index'])
    print(pd.merge(df, vocab[['index', 'word']], how='left', on='index')['word'].values)
    print()
from wordcloud import WordCloud
import matplotlib.pyplot as plt
def plot_word_cloud(b, ax, v, n):
    sorted_, indices = torch.sort(b, descending=True)
    df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
    words = pd.merge(df, vocab[['index', 'word']], how='left', on='index')['word'].values.tolist()
    sizes = (sorted_[:100] * 1000).int().numpy().tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(background_color="white", width=800, height=500)  # max_words=1000
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title('Topic %d' % (n + 1))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    

fig, axs = plt.subplots(7, 3, figsize=(14, 24))
for n in range(beta.shape[0]):
    i, j = divmod(n, 3)
    plot_word_cloud(beta[n], axs[i, j], vocab, n)
axs[-1, -1].axis('off');

plt.show()
