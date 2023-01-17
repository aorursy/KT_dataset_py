!pip install pyro-ppl -q
import math
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO, Predictive, TraceEnum_ELBO, TraceMeanField_ELBO, config_enumerate
from tqdm import trange
#@config_enumerate
def model(data, vocab_size, num_docs, num_topics, doc_idx=None):
    # Globals.
    eta = data.new_ones(vocab_size)
    with pyro.plate("topics", num_topics):
        beta = pyro.sample("beta", dist.Dirichlet(eta))

    # Locals.
    with pyro.plate("documents", data.shape[1]):
        alpha = data.new_ones(num_topics)
        theta = pyro.sample("theta", dist.Dirichlet(alpha))

        with pyro.plate("words", data.shape[0]):
            zeta = pyro.sample("zeta", dist.Categorical(theta))
            pyro.sample("doc_words", dist.Categorical(beta[..., zeta, :]), obs=data)


#@config_enumerate
def guide(data, vocab_size, num_docs, num_topics, doc_idx=None):
    # Parameters
    lambda_ = pyro.param("lambda", data.new_ones(num_topics, vocab_size))
    gamma = pyro.param("gamma", data.new_ones(num_docs, num_topics))
    phi = pyro.param("phi", data.new_ones(num_docs, data.shape[0], num_topics),
                     constraint=constraints.positive)
    phi = phi / phi.sum(dim=2, keepdim=True)  # Enforces probability

    # Topics
    with pyro.plate("topics", num_topics):
        pyro.sample("beta", dist.Dirichlet(lambda_))

    # Documents
    with pyro.plate("documents", data.shape[1]):
        pyro.sample("theta", dist.Dirichlet(gamma[..., doc_idx, :]))

        # Words
        with pyro.plate("words", data.shape[0]):
            pyro.sample("zeta", dist.Categorical(phi[..., doc_idx, :, :].transpose(1, 0)))
def train(device, docs, vocab_size, num_topics, batch_size, learning_rate, num_epochs):
    # clear param store
    pyro.clear_param_store()

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO(num_particles=1))
    num_batches = int(math.ceil(docs.shape[0] / batch_size))

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0

        # Iterate over data.
        for i in range(num_batches):
            idx = torch.arange(i * batch_size, min((i + 1) * batch_size, len(docs)))
            batch_docs = docs[idx, :]
            loss = svi.step(batch_docs.T, vocab_size, docs.shape[0], num_topics, idx)
            running_loss += loss

        epoch_loss = running_loss / docs.shape[0]
        bar.set_postfix(epoch_loss='{:.2f}'.format(epoch_loss))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
vocab = pd.read_csv('../input/prodlda/vocab.csv')
vocab_size = len(vocab)
num_topics = 100
    
docs = torch.load('../input/prodlda/docs_ap.pt').float().to(device)
train(device, docs, vocab_size, num_topics, 32, 1e-3, 80)

%%time
predictive = Predictive(model, guide=guide, num_samples=100,
                        return_sites=["beta", 'obs'])

i = 0
batch_size = 32
idx = torch.arange(i * batch_size, min((i + 1) * batch_size, len(docs))).cpu()
batch_docs = docs[idx, :].cpu()
samples = predictive(batch_docs.T, vocab_size, docs.shape[0], num_topics, idx)
beta = samples['beta'].mean(dim=0).squeeze().detach().cpu()
beta.shape
vocab = pd.read_csv('../input/prodlda/vocab.csv')

for i in range(beta.shape[0]):
    sorted_, indices = torch.sort(beta[i], descending=True)
    df = pd.DataFrame(indices[:20].numpy(), columns=['index'])
    print(pd.merge(df, vocab[['index', 'word']], how='left', on='index')['word'].values)
    print()
vocab_size
