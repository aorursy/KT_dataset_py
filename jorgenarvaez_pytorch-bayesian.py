import torch

import torch.nn as nn

import numpy as np

import random

import matplotlib.pyplot as plt
# Define torch device

torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#

# Functions for easy training

# 'train' function is based on trainer object

# trainer object need implement a run_batch method

#

# def run_batch(self, batch, training=True):

# # batch is a dictionary with data for training

# # Run training if training=True, else calcule criterions only

# # Return criterions dictionary, including loss

#     return {

#         'criterion_a': <a criterion>

#         'loss': <loss criterion>

#     }

#



def run_minibatches(trainer, batch, minibatch_size, training=True):

    data_size = batch[list(batch.keys())[0]].size(0)

    minibatch_N = data_size // minibatch_size

    minibatch_start = 0

    minibatch = {}

    criterions_sum = {}

    while minibatch_start + minibatch_size <= data_size:

        # Select minibatch and run trainer

        for key in batch:

            minibatch[key] = batch[key][minibatch_start:minibatch_start+minibatch_size]

        criterions = trainer.run_batch(minibatch, training=training)

        # Update sum of criterions

        for c in criterions:

            criterions_sum[c] = criterions[c] + (criterions_sum[c] if c in criterions_sum else 0)

        # Increment minibatch start

        minibatch_start += minibatch_size

    # Return mean of criterions

    return {c : criterions_sum[c]/minibatch_N for c in criterions_sum}



def run_round_and_save_history(i, trainer, loader, history, minibatch_size, training=True):

    batch = loader if type(loader) is dict else  loader(i)

    criterions = run_minibatches(trainer, batch, minibatch_size, training)

    for c in criterions:

        if c in history:

            history[c].append(criterions[c])

        else:

            history[c] = [criterions[c]]

    return criterions



# Run complete training

def train(trainer, max_rounds, minibatch_size, train_data_loader, valid_data_loader, 

          loss_criterion='loss', stop_function=None, selection='last', vervose=False, temp_path='/tmp/'):

    random_int = random.randint(0, 1000000)              # Random int for temporal saving

    # Verify selection choice

    if selection not in ['last', 'min-valid-loss']:

        raise Exception("Invalid Selection")

    # Init values 

    train_criterions_history = {}

    valid_criterions_history = {}

    min_valid_loss = np.inf

    for i in range(max_rounds):

        # Training

        run_round_and_save_history(i, trainer, train_data_loader, 

                                   train_criterions_history, minibatch_size, training=True)

        # Validation

        if valid_data_loader is not None:

            run_round_and_save_history(i, trainer, valid_data_loader, 

                                       valid_criterions_history, minibatch_size, training=False)

            valid_loss = valid_criterions_history[loss_criterion][-1]

        # Save best model if is necessary

        if selection == 'min-valid-loss' and valid_loss < min_valid_loss:

            min_valid_loss = valid_loss

            torch.save(trainer.model, temp_path + str(random_int) + '.pt')          

        # Check stop function if it was provided

        if (stop_function is not None) and (stop_function(train_criterions_history, valid_criterions_history)):

            break

    if selection == 'min-valid-loss':

        best_model = torch.load(temp_path + str(random_int) + '.pt')

    elif selection == 'last':

        best_model = trainer.model    

    return best_model, train_criterions_history, valid_criterions_history



# Univariate normal distribution model definition

# out:     mean,    std

# ranges: [-1, 1],  [0, 2]

class UnivariateNormalDistribution(nn.Module):

    

    def __init__(self):

        nn.Module.__init__(self)

        self.l1 = nn.Linear(1, 100, bias=True)

        self.a1 = nn.ReLU()

        self.l2 = nn.Linear(100, 50)

        self.a2 = nn.ReLU()

        self.l3 = nn.Linear(50, 2)

        self.a3 = nn.Tanh()

    

    def forward(self):

        x = torch.ones(size=(1, 1)).to(self.l1.weight.device)

        o1 = self.a1(self.l1(x))

        o2 = self.a2(self.l2(o1))

        o =  self.a3(self.l3(o2))

        return o[0,0], o[0,1] + 1.0

# Univariate normal distribution trainer definition

class UnivariateNormalDistributionTrainer():

    

    def __init__(self, model):

        self.model = model

        # Define optimizer

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    

    # Define normal pdf for calculate log-likelihood

    def normal_pdf(self, x, mean, std):

         return torch.exp(-0.5*((x-mean)/std)**2)/(std*np.sqrt(2*np.pi))

    

    # run_batch implementation

    def run_batch(self, batch, training=True):

        x = batch['X']                      # Extract data from dictionary

        mean, std = self.model()            # get distribution from model

        p = self.normal_pdf(x, mean, std)   # Calculate individual probability

        log_likelihood = torch.log(p).sum() # Calculate log_likelihood

        loss = -log_likelihood              # Define loss to minimice

        # Define loss criterion and others criterions to register history

        criterions = {

            'log-likelihood': log_likelihood,

            'loss': loss

        }

        # Run training if training is True

        if training:

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        return criterions

        
# Define 'Real' distribution, limit to model ranges

real_mean = -0.5

real_std = 1.2

# Generate samples

X = torch.normal(real_mean, real_std, size=(10000, 1)).to(torch_device) # 1000 sambles

data = {'X': X} # Define data dictionary

# define model

model = UnivariateNormalDistribution().to(torch_device)

trainer = UnivariateNormalDistributionTrainer(model)



# Start train

trained_model, train_history, _ = train(

    trainer, 

    max_rounds = 20, 

    minibatch_size = 128, 

    train_data_loader = data, 

    valid_data_loader = None

)



trained_mean, trained_std = trained_model()

trained_mean = trained_mean.data.item() # Extract value

trained_std = trained_std.data.item() # Extract value



print('real_mean:\t', real_mean)

print('trained_mean:\t', trained_mean)

print('real_std:\t', real_std)

print('trained_std:\t', trained_std)



plt.figure(figsize=[8, 4])

plt.plot(train_history['log-likelihood'])

plt.xlabel('Epoch')

plt.ylabel('log-likelihood')

plt.show()
# Conditional univariate normal distribution model definition

# out:     mean,    std

# ranges: [-1, 1],  [0, 2]

class ConditionalUnivariateNormalDistribution(nn.Module):

    

    def __init__(self):

        nn.Module.__init__(self)

        self.l1 = nn.Linear(4, 100, bias=False)

        self.a1 = nn.ReLU()

        self.l2 = nn.Linear(100, 50, bias=False)

        self.a2 = nn.ReLU()

        self.l3 = nn.Linear(50, 2, bias=False)

        self.a3 = nn.Tanh()

    

    def forward(self, x):

        o1 = self.a1(self.l1(x))

        o2 = self.a2(self.l2(o1))

        o =  self.a3(self.l3(o2))

        return o[:,0].view(-1,1), o[:,1].view(-1,1) + 1.0
# Conditional Univariate normal distribution trainer definition

class ConditionalUnivariateNormalDistributionTrainer():

    

    def __init__(self, model):

        self.model = model

        # Define optimizer

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    

    # Define normal pdf for calculate log-likelihood

    def normal_pdf(self, x, mean, std):

         return torch.exp(-0.5*((x-mean)/std)**2)/(std*np.sqrt(2*np.pi))

    

    # run_batch implementation

    def run_batch(self, batch, training=True):

        x = batch['X']                      # Extract data from dictionary

        y = batch['Y']                      # Extract data from dictionary

        mean, std = self.model(x)           # get distributions from model

        p = self.normal_pdf(y, mean, std)   # Calculate individual probability

        log_likelihood = torch.log(p).sum() # Calculate log_likelihood

        loss = -log_likelihood              # Define loss to minimice

        # Define loss criterion and others criterions to register history

        criterions = {

            'log-likelihood': log_likelihood,

            'loss': loss

        }

        # Run training if training is True

        if training:

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        return criterions

        
# Define 'Real' distribution, limit to model ranges

init_x = torch.rand(size=[5, 4])

init_mean = (-0.7 + init_x[:,0]*init_x[:,1]).view(-1,1)

init_std  = (0.5 + 1.5*(init_x[:,2] + init_x[:,3])/2.0).view(-1,1)



N_samples_per_dist = 4000



# Generate samples

X =       init_x.repeat([N_samples_per_dist, 1])

mean = init_mean.repeat([N_samples_per_dist, 1])

std =   init_std.repeat([N_samples_per_dist, 1])

Y = torch.normal(mean, std)



# Shuffle Data

r_index=torch.randperm(X.size(0))

X =       X[r_index]

mean = mean[r_index]

std =   std[r_index]

Y =       Y[r_index]



data = {'X': X, 'Y': Y} # Define data dictionary
# define model

model = ConditionalUnivariateNormalDistribution().to(torch_device)

trainer = ConditionalUnivariateNormalDistributionTrainer(model)



# Start train

trained_model, train_history, _ = train(

    trainer, 

    max_rounds = 40, 

    minibatch_size = 128, 

    train_data_loader = data, 

    valid_data_loader = None

)



trained_mean, trained_std = trained_model(init_x)



print("mean:      Real vs Trained")

print(torch.cat([init_mean, trained_mean], axis=1) )



print("\nstd:      Real vs Trained")

print(torch.cat([init_std, trained_std], axis=1) )



plt.figure(figsize=[8, 4])

plt.plot(train_history['log-likelihood'])

plt.xlabel('Epoch')

plt.ylabel('log-likelihood')

plt.show()
# Discrete distribution model definition

# out:     N_states discrete Distribution

# ranges:  0 -> 1

class DiscreteDistribution(nn.Module):

    

    def __init__(self, N_states):

        nn.Module.__init__(self)

        self.l1 = nn.Linear(1, 100, bias=False)

        self.a1 = nn.ReLU()

        self.l2 = nn.Linear(100, 100, bias=False)

        self.a2 = nn.ReLU()

        self.l3 = nn.Linear(100, N_states, bias=False)

        self.a3 = nn.Softmax(1)

    

    def forward(self):

        x = torch.ones(size=(1, 1)).to(self.l1.weight.device)

        o1 = self.a1(self.l1(x))

        o2 = self.a2(self.l2(o1))

        o =  self.a3(self.l3(o2))

        return o[0, :]

    

# Discrete distribution trainer definition

class DiscreteDistributionTrainer():

    

    def __init__(self, model):

        self.model = model

        # Define optimizer

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    

    # run_batch implementation

    def run_batch(self, batch, training=True):

        x = batch['X']                      # Extract data from dictionary

        discrete_dist = self.model()        # get distribution from model

        p = discrete_dist[x]                # Calculate individual probability

        log_likelihood = torch.log(p).sum() # Calculate log_likelihood

        loss = -log_likelihood              # Define loss to minimice

        # Define loss criterion and others criterions to register history

        criterions = {

            'log-likelihood': log_likelihood,

            'loss': loss

        }

        # Run training if training is True

        if training:

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        return criterions

# Define 'Real' distribution, limit to model ranges

real_P = [0.1, 0.5, 0.2, 0.2]



# Generate samples

N_samples = 10000

X_np = np.random.choice(range(len(real_P)), N_samples, p = real_P)

X = torch.Tensor(list(X_np)).type(torch.long).view(-1,1).to(torch_device)



data = {'X': X} # Define data dictionary



# define model

model = DiscreteDistribution(4).to(torch_device)

trainer = DiscreteDistributionTrainer(model)



# Start train

trained_model, train_history, _ = train(

    trainer, 

    max_rounds = 40, 

    minibatch_size = 128, 

    train_data_loader = data, 

    valid_data_loader = None

)



trained_P = trained_model()

trained_P = list(trained_P.cpu().detach().numpy())



print('real_P:\t\t',  "[{:.3f} {:.3f} {:.3f} {:.3f}]".format(*real_P))

print('trained_P:\t', "[{:.3f} {:.3f} {:.3f} {:.3f}]".format(*trained_P))



plt.figure(figsize=[8, 4])

plt.plot(train_history['log-likelihood'])

plt.xlabel('Epoch')

plt.ylabel('log-likelihood')

plt.show()