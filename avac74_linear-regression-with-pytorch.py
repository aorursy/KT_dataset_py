# torchsummaryX is a pretty neat library that allows one to get a summary of the PyTorch modelin a similar way to

# the summary provided by Keras

!pip install torchsummaryX
import torch

import numpy as np



import matplotlib.pyplot as plt



from torchsummaryX import summary



from pylab import rcParams



rcParams['figure.figsize'] = 10, 8
x = np.arange(100)

y = x + np.random.randn(len(x)) * 10
plt.plot(x, y, 'o')
model = torch.nn.Sequential(

    torch.nn.Linear(1, 1),

)



# Make sure the data is represented as tensors

xt = torch.tensor(x.reshape(-1, 1), dtype=torch.float)

yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
summary(model, xt)
[x for x in model[0].parameters()]
# Let's run this for a few epochs

for i in range(20000):

    # Run the data through our model

    y_pred = model(xt)

    

    # Using MSE as the loss function, no big deal here

    loss = (y_pred - yt).pow(2.0).sum()

    

    # According to PyTorch's documentation, the gradient values are accumulated in the leaves of the

    # graph that represents the model. This means that between each interaction, before running the backward

    # pass, I need to zero the gradients

    model.zero_grad()

    

    # This does the backward pass, which basically computes the gradients for all parameters that can be learned

    # in our model (the ones that I mentioned above as flagged with requires_grad=True) 

    loss.backward()

    

    if i % 1000 == 0:

        print(i, loss.item())

    

    # Can improve here by using PyTorch's optimizer, instead of updating the parameters myself(??)

    with torch.no_grad():

        for p in model.parameters():

            p -= 0.0000001 * p.grad    # update the parameters using a learning rate to scale down the changes
y_pred = model(xt)
plt.plot(xt.detach().numpy(), y_pred.detach().numpy(), label='Regression')

plt.plot(x, y, 'or', label='Data Points')

plt.plot(x, x, 'k', label='Original Line')

plt.legend()

plt.show()
[x for x in model.parameters()]