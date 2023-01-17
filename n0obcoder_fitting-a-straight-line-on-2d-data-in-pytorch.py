# Importing the bare necessities...

seed_for_reproducibility = 3333

import numpy as np

np.random.seed(seed_for_reproducibility)

import random

random.seed(seed_for_reproducibility)

from matplotlib import pyplot as plt

%matplotlib inline



import torch

torch.manual_seed(seed_for_reproducibility)
# Defining the range of 'x', the independent variable

x_range = [-2000, 2000]



# Defining the extent of noise, which would be added to both the dependent as well as the independent variable

deviation = 100



# For the data points to roughly fall on a staright line, we need to define the slope and the intercept of that line

# Let's intoduce some randomness in the slope and intercept selection process

m_synthetic = random.randint(-100, 100)/100. # m_synthetic is real number from the set(-1.0, -0.99, -0.98 ...., 0.98, 0.99, 1.0)

c_synthetic = random.randint(-10, 10)        # c_synthetic is an integer from the set(-10, -9, -8 ..., 8, 9, 10)



print('\nm_synthetic: {}'.format(m_synthetic))

print('c_synthetic: {}'.format(c_synthetic))



# Defining the number of data points to be generated

num_points = 100



# Let's generate the synthetic data points

x_list = []

y_list = []

for _ in range(num_points):

    

    # Selecting a random integer from the predefined range

    x = random.randint(x_range[0] , x_range[1])

    

    # Calculating the dependent valiable 'y' using the formula of a straight line y = mx + c

    y = m_synthetic*x + c_synthetic

    

    # Randomly choosing the deviation (noise) for both the dependent as well as the independent variable, so that the dataset becomes a little noisy

    deviation_x = random.randint(-deviation , deviation)

    deviation_y = random.randint(-deviation , deviation)

    

    # Finally, appending the noisy data points in the respective lists

    x_list.append(x + deviation_x)

    y_list.append(y + deviation_y)



x_min, y_min, x_max, y_max = min(x_list) , min(y_list) , max(x_list) , max(y_list)

print('\nFollowing are the extreme ends of the synthetic data points...')

print('x_min, x_max: {}, {}'.format(x_min, x_max))

print('y_min, y_max: {}, {}'.format(y_min, y_max))
# Let's now visualize how the synthetic data looks like 

plt.scatter(x_list, y_list , color = 'cyan')

plt.plot((x_min, x_max), (m_synthetic*x_min + c_synthetic, m_synthetic*x_max + c_synthetic), color = 'r')

plt.title('Synthetic Data with m = {} and c = {}'.format(m_synthetic, c_synthetic))

plt.xlabel("Independent Variable 'x'")

plt.ylabel("Dependent Variable 'y'")

plt.savefig('synthetic_m_and_c.jpg')
# Defining the model architecture.

class LinearRegressionModel(torch.nn.Module): 

    def __init__(self): 

        super(LinearRegressionModel, self).__init__() 

        self.linear = torch.nn.Linear(1, 1)  # this layer of the model has a single neuron, that takes in one scalar input and gives out one scalar output. 

  

    def forward(self, x): 

        y_pred = self.linear(x) 

        return y_pred 



# Creating the model

model = LinearRegressionModel()
# We can print the 'model' to see it's architecture

print(model)
for name, parameter in model.named_parameters():

    print('name           : {}'.format(name))

    print('parameter      : {}'.format(parameter.item()))

    print('learnable      : {}'.format(parameter.requires_grad))

    print('parameter.shape: {}'.format(parameter.shape))

    print('---------------------------------')
# Defining the Loss Function

# Mean Squared Error is the most common choice of Loss Function for Linear Regression models.

criterion = torch.nn.MSELoss()



# Defining the Optimizer, which would update all the trainable parameters of the model, making the model learn the data distribution better and hence fit the distribution better.

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005) 
# We also need to convert all the data into tensors before we could use them for training our model.

data_x = torch.tensor([[x] for x in x_list], dtype = torch.float)

data_y = torch.tensor([[y] for y in y_list], dtype = torch.float)
losses = []         # to keep track of the epoch lossese 

slope_list = []     # to keep track of the slope learnt by the model

intercept_list = [] # to keep track of the intercept learnt by the model



EPOCHS = 2500

print('\nTRAINING...')

for epoch in range(EPOCHS):

    # We need to clear the gradients of the optimizer before running the back-propagation in PyTorch

    optimizer.zero_grad() 

    

    # Feeding the input data in the model and getting out the predictions

    pred_y = model(data_x)



    # Calculating the loss using the model's predictions and the real y values

    loss = criterion(pred_y, data_y) 



    # Back-Propagation

    loss.backward() 

    

    # Updating all the trainable parameters

    optimizer.step()

    

    # Appending the loss.item() (a scalar value)

    losses.append(loss.item())

    

    # Appending the learnt slope and intercept   

    slope_list.append(model.linear.weight.item())

    intercept_list.append(model.linear.bias.item())

    

    # We print out the losses after every 2000 epochs

    if (epoch)%100 == 0:

        print('loss: ', loss.item())



# Let's see what are the learnt parameters after having trained the model for hundreds of epochs

m_learnt = model.linear.weight.item()

c_learnt = model.linear.bias.item()



print('\nCompare the learnt parameters with the original ones')

print('\nm_synthetic     VS     m_learnt')

print('     {}                   {}'.format(m_synthetic, m_learnt))

print('\nc_synthetic     VS     c_learnt')

print('     {}                   {}'.format(c_synthetic, c_learnt))



# Plotting the epoch losses

plt.plot(losses)

plt.title('Loss VS Epoch')

plt.xlabel('#Epoch')

plt.ylabel('Loss')

plt.savefig('losses.jpg')
plt.scatter(x_list, y_list , color = 'cyan')

plt.plot((x_min, x_max), (m_learnt*x_min + c_learnt, m_learnt*x_max + c_learnt), color = 'r')

plt.title('Synthetic Data Points, with m = {} and c = {}'.format(round(m_learnt, 2), round(c_learnt, 2)))

plt.xlabel("Independent Variable 'x'")

plt.ylabel("Dependent Variable 'y'")

plt.savefig('learnt_m_and_c.jpg')
plt.plot(slope_list)

plt.plot(intercept_list)

plt.title('Learnt Values of Slope and Intercept')

plt.legend(['slope', 'intercept'])

plt.xlabel('#Epochs')

plt.ylabel('Learnt Parameteres')

plt.savefig('learning_m_and_c.jpg')