!pip install pyro-ppl
import numpy as np

import scipy.special as ssp

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.distributions.constraints as constraints



from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler



import pyro

import pyro.distributions as dist



from pyro.infer import SVI, Trace_ELBO

from pyro.optim import Adam, SGD



pyro.enable_validation(True)

torch.set_default_dtype(torch.double)
def generate_data(loc=0, scale=2, num_samples=100000, dim=2, alphas=(1, 1), beta=0, plt_data=True):

    """

    function to sample data points from gaussian dist given loc and scale. 

    1. According to given alphas and beta to draw a boundary to split the data set to pos and neg. 

    2. Through logistic regression equation to calculate the probs of each points p.

    3. Sample scores from uniform distribution, for each point, if p > s, label the data with 1 otherwise 0

    default: generate 2-dims data from N(loc, scale) 

    """

    def sigmoid(x):

        return 1/(1 + np.exp(-x))

    

    X = np.random.normal(loc=loc, scale=scale, size=[num_samples, dim])

    

    p = sigmoid(np.sum(X * alphas, axis=1) + beta)

    s = np.random.uniform(0, 1, size=[num_samples])

    

    y = p > s

    y = y.astype(float)

    

    plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], s=1, alpha=0.)

    plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], s=1, alpha=0.3)

    

    plt.show()

    

    return X, y.reshape(-1, 1)
generate_data()
def train(svi, train_loader, n_train):

    # initialize loss accumulator

    epoch_loss = 0.

    # do a training epoch over each mini-batch x returned

    # by the data loader

    for _, xs in enumerate(train_loader):

        # do ELBO gradient and accumulate loss

        epoch_loss += svi.step(*xs)



    # return epoch loss

    total_epoch_loss_train = epoch_loss / n_train

    return total_epoch_loss_train





def evaluate(svi, test_loader, n_test):

    # initialize loss accumulator

    test_loss = 0.

    # compute the loss over the entire test set

    for _, xs in enumerate(test_loader):

        # compute ELBO estimate and accumulate loss

        test_loss += svi.evaluate_loss(*xs)



    total_epoch_loss_test = test_loss / n_test

    return total_epoch_loss_test





def plot_llk(train_elbo, test_elbo, test_int):

    plt.figure(figsize=(8, 6))



    x = np.arange(len(train_elbo))



    plt.plot(x, train_elbo, marker='o', label='Train ELBO')

    plt.plot(x[::test_int], test_elbo, marker='o', label='Test ELBO')

    plt.xlabel('Training Epoch')

    plt.legend()

    plt.show()
class LogRegressionModel(nn.Module):

    def __init__(self, p):

        super(LogRegressionModel, self).__init__()

        

        self.p = p



        # hyperparameters for normal priors

        self.alpha_h_loc = torch.zeros(1, p)

        self.alpha_h_scale = 10.0 * torch.ones(1, p)

        self.beta_h_loc = torch.zeros(1)

        self.beta_h_scale = 10.0 * torch.ones(1)

        

        # initial values of variational parameters

        self.alpha_0 = np.zeros((1, p))

        self.alpha_0_scale = np.ones((1, p))

        self.beta_0 = np.zeros((1,))

        self.beta_0_scale = np.ones((1,))



    def model(self, x, y):

        # sample from prior

        a = pyro.sample(

            "weight", dist.Normal(self.alpha_h_loc, self.alpha_h_scale, validate_args=True).independent(1)

        )

        b = pyro.sample(

            "bias", dist.Normal(self.beta_h_loc, self.beta_h_scale, validate_args=True).independent(1)

        )



        with pyro.iarange("data", x.size(0)):

            model_logits = (torch.matmul(x, a.permute(1, 0)) + b).squeeze()

            

            pyro.sample(

                "obs", 

                dist.Bernoulli(logits=model_logits, validate_args=True),

                obs=y.squeeze()

            )

            

    def guide(self, x, y):

        # register variational parameters with pyro

        alpha_loc = pyro.param("alpha_loc", torch.tensor(self.alpha_0))

        alpha_scale = pyro.param("alpha_scale", torch.tensor(self.alpha_0_scale),

                                 constraint=constraints.positive)

        beta_loc = pyro.param("beta_loc", torch.tensor(self.beta_0))

        beta_scale = pyro.param("beta_scale", torch.tensor(self.beta_0_scale),

                                constraint=constraints.positive)



        pyro.sample(

            "weight", dist.Normal(alpha_loc, alpha_scale, validate_args=True).independent(1)

        )

        pyro.sample(

            "bias", dist.Normal(beta_loc, beta_scale, validate_args=True).independent(1)

        )
optim = Adam({'lr': 0.01})



num_points = 100000

data_dim = 2

num_subsample = 10



X, y = generate_data(dim=data_dim, num_samples=num_points)



example_indices = np.random.permutation(num_points)

n_train = int(0.9 * num_points)  # 90%/10% train/test split

n_test = num_points - n_train

test_iter = 50



X = torch.from_numpy(X)

y = torch.from_numpy(y)



pyro.clear_param_store()



num_epochs = 300

batch_size = 10



lr_model = LogRegressionModel(p=data_dim)



svi = SVI(lr_model.model, lr_model.guide, optim, loss=Trace_ELBO())





lr_dataset = torch.utils.data.TensorDataset(X, y)

data_loader_train = DataLoader(

    dataset=lr_dataset, batch_size=batch_size, pin_memory=False,

    sampler=SubsetRandomSampler(example_indices[:num_subsample]),

)

    

data_loader_test = DataLoader(

    dataset=lr_dataset, batch_size=batch_size, pin_memory=False,

    sampler=SubsetRandomSampler(example_indices[n_train:]),

)



train_elbo = []

test_elbo = []

for epoch in range(num_epochs):

    total_epoch_loss_train = train(svi, data_loader_train, n_train)

    train_elbo.append(-total_epoch_loss_train)



    if epoch % test_iter == 0:

        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        # report test diagnostics

        total_epoch_loss_test = evaluate(svi, data_loader_test, n_test)

        test_elbo.append(-total_epoch_loss_test)

        

print(f"Uniform sampling with {num_subsample} examples")

print(f"estimated loc for alpha is {pyro.param('alpha_loc')}")

print(f"estimated scale for alpha is {pyro.param('alpha_scale')}")

print(f"estimated loc for beta is {pyro.param('beta_loc')}")

print(f"estimated scale for beta is {pyro.param('beta_scale')}")
optim = Adam({'lr': 0.01})



num_points = 100000

data_dim = 2

num_subsample = 100



X, y = generate_data(dim=data_dim, num_samples=num_points)



example_indices = np.random.permutation(num_points)

n_train = int(0.9 * num_points)  # 90%/10% train/test split

n_test = num_points - n_train

test_iter = 50



X = torch.from_numpy(X)

y = torch.from_numpy(y)



pyro.clear_param_store()



num_epochs = 300

batch_size = 50



lr_model = LogRegressionModel(p=data_dim)



svi = SVI(lr_model.model, lr_model.guide, optim, loss=Trace_ELBO())





lr_dataset = torch.utils.data.TensorDataset(X, y)

data_loader_train = DataLoader(

    dataset=lr_dataset, batch_size=batch_size, pin_memory=False,

    sampler=SubsetRandomSampler(example_indices[:num_subsample]),

)

    

data_loader_test = DataLoader(

    dataset=lr_dataset, batch_size=batch_size, pin_memory=False,

    sampler=SubsetRandomSampler(example_indices[n_train:]),

)



train_elbo = []

test_elbo = []

for epoch in range(num_epochs):

    total_epoch_loss_train = train(svi, data_loader_train, n_train)

    train_elbo.append(-total_epoch_loss_train)



    if epoch % test_iter == 0:

        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        # report test diagnostics

        total_epoch_loss_test = evaluate(svi, data_loader_test, n_test)

        test_elbo.append(-total_epoch_loss_test)

        

print("Uniform sampling with 100 examples")

print(f"estimated loc for alpha is {pyro.param('alpha_loc')}")

print(f"estimated scale for alpha is {pyro.param('alpha_scale')}")

print(f"estimated loc for beta is {pyro.param('beta_loc')}")

print(f"estimated scale for beta is {pyro.param('beta_scale')}")
optim = Adam({'lr': 0.01})



num_points = 100000

data_dim = 2

num_subsample = 1000



X, y = generate_data(dim=data_dim, num_samples=num_points)



example_indices = np.random.permutation(num_points)

n_train = int(0.9 * num_points)  # 90%/10% train/test split

n_test = num_points - n_train

test_iter = 50



X = torch.from_numpy(X)

y = torch.from_numpy(y)



pyro.clear_param_store()



num_epochs = 300

batch_size = 50



lr_model = LogRegressionModel(p=data_dim)



svi = SVI(lr_model.model, lr_model.guide, optim, loss=Trace_ELBO())





lr_dataset = torch.utils.data.TensorDataset(X, y)

data_loader_train = DataLoader(

    dataset=lr_dataset, batch_size=batch_size, pin_memory=False,

    sampler=SubsetRandomSampler(example_indices[:num_subsample]),

)

    

data_loader_test = DataLoader(

    dataset=lr_dataset, batch_size=batch_size, pin_memory=False,

    sampler=SubsetRandomSampler(example_indices[n_train:]),

)



train_elbo = []

test_elbo = []

for epoch in range(num_epochs):

    total_epoch_loss_train = train(svi, data_loader_train, n_train)

    train_elbo.append(-total_epoch_loss_train)



    if epoch % test_iter == 0:

        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        # report test diagnostics

        total_epoch_loss_test = evaluate(svi, data_loader_test, n_test)

        test_elbo.append(-total_epoch_loss_test)

        

print("Uniform sampling with 1000 examples")

print(f"estimated loc for alpha is {pyro.param('alpha_loc')}")

print(f"estimated scale for alpha is {pyro.param('alpha_scale')}")

print(f"estimated loc for beta is {pyro.param('beta_loc')}")

print(f"estimated scale for beta is {pyro.param('beta_scale')}")
optim = Adam({'lr': 0.01})



num_points = 100000

data_dim = 2

num_subsample = 10000



X, y = generate_data(dim=data_dim, num_samples=num_points)



example_indices = np.random.permutation(num_points)

n_train = int(0.9 * num_points)  # 90%/10% train/test split

n_test = num_points - n_train

test_iter = 50



X = torch.from_numpy(X)

y = torch.from_numpy(y)



pyro.clear_param_store()



num_epochs = 300

batch_size = 50



lr_model = LogRegressionModel(p=data_dim)



svi = SVI(lr_model.model, lr_model.guide, optim, loss=Trace_ELBO())





lr_dataset = torch.utils.data.TensorDataset(X, y)

data_loader_train = DataLoader(

    dataset=lr_dataset, batch_size=batch_size, pin_memory=False,

    sampler=SubsetRandomSampler(example_indices[:num_subsample]),

)

    

data_loader_test = DataLoader(

    dataset=lr_dataset, batch_size=batch_size, pin_memory=False,

    sampler=SubsetRandomSampler(example_indices[n_train:]),

)



train_elbo = []

test_elbo = []

for epoch in range(num_epochs):

    total_epoch_loss_train = train(svi, data_loader_train, n_train)

    train_elbo.append(-total_epoch_loss_train)



    if epoch % test_iter == 0:

        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        # report test diagnostics

        total_epoch_loss_test = evaluate(svi, data_loader_test, n_test)

        test_elbo.append(-total_epoch_loss_test)

        

print(f"Uniform sampling with {num_subsample} examples")

print(f"estimated loc for alpha is {pyro.param('alpha_loc')}")

print(f"estimated scale for alpha is {pyro.param('alpha_scale')}")

print(f"estimated loc for beta is {pyro.param('beta_loc')}")

print(f"estimated scale for beta is {pyro.param('beta_scale')}")