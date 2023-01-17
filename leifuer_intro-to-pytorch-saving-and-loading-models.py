%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms



# import helper

# import fc_model
## fc_model.py



import torch

from torch import nn

import torch.nn.functional as F





class Network(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):

        ''' Builds a feedforward network with arbitrary hidden layers.

        

            Arguments

            ---------

            input_size: integer, size of the input layer

            output_size: integer, size of the output layer

            hidden_layers: list of integers, the sizes of the hidden layers

        

        '''

        super().__init__()

        # Input to a hidden layer

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        

        # Add a variable number of more hidden layers

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        

        self.output = nn.Linear(hidden_layers[-1], output_size)

        

        self.dropout = nn.Dropout(p=drop_p)

        

    def forward(self, x):

        ''' Forward pass through the network, returns the output logits '''

        

        for each in self.hidden_layers:

            x = F.relu(each(x))

            x = self.dropout(x)

        x = self.output(x)

        

        return F.log_softmax(x, dim=1)





def validation(model, testloader, criterion):

    accuracy = 0

    test_loss = 0

    for images, labels in testloader:



        images = images.resize_(images.size()[0], 784)



        output = model.forward(images)

        test_loss += criterion(output, labels).item()



        ## Calculating the accuracy 

        # Model's output is log-softmax, take exponential to get the probabilities

        ps = torch.exp(output)

        # Class with highest probability is our predicted class, compare with true label

        equality = (labels.data == ps.max(1)[1])

        # Accuracy is number of correct predictions divided by all predictions, just take the mean

        accuracy += equality.type_as(torch.FloatTensor()).mean()



    return test_loss, accuracy





def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):

    

    steps = 0

    running_loss = 0

    for e in range(epochs):

        # Model in training mode, dropout is on

        model.train()

        for images, labels in trainloader:

            steps += 1

            

            # Flatten images into a 784 long vector

            images.resize_(images.size()[0], 784)

            

            optimizer.zero_grad()

            

            output = model.forward(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            

            running_loss += loss.item()



            if steps % print_every == 0:

                # Model in inference mode, dropout is off

                model.eval()

                

                # Turn off gradients for validation, will speed up inference

                with torch.no_grad():

                    test_loss, accuracy = validation(model, testloader, criterion)

                

                print("Epoch: {}/{}.. ".format(e+1, epochs),

                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),

                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),

                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                

                running_loss = 0

                

                # Make sure dropout and grads are on for training

                model.train()
## helper.py



import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torch.autograd import Variable





def test_network(net, trainloader):



    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)



    dataiter = iter(trainloader)

    images, labels = dataiter.next()



    # Create Variables for the inputs and targets

    inputs = Variable(images)

    targets = Variable(images)



    # Clear the gradients from all Variables

    optimizer.zero_grad()



    # Forward pass, then backward pass, then update weights

    output = net.forward(inputs)

    loss = criterion(output, targets)

    loss.backward()

    optimizer.step()



    return True





def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax





def view_recon(img, recon):

    ''' Function for displaying an image (as a PyTorch Tensor) and its

        reconstruction also a PyTorch Tensor

    '''



    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    axes[0].imshow(img.numpy().squeeze())

    axes[1].imshow(recon.data.numpy().squeeze())

    for ax in axes:

        ax.axis('off')

        ax.set_adjustable('box-forced')



def view_classify(img, ps, version="MNIST"):

    ''' Function for viewing an image and it's predicted classes.

    '''

    ps = ps.data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    if version == "MNIST":

        ax2.set_yticklabels(np.arange(10))

    elif version == "Fashion":

        ax2.set_yticklabels(['T-shirt/top',

                            'Trouser',

                            'Pullover',

                            'Dress',

                            'Coat',

                            'Sandal',

                            'Shirt',

                            'Sneaker',

                            'Bag',

                            'Ankle Boot'], size='small');

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()
# Define a transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data

trainset = datasets.FashionMNIST('/F_MNIST_data/', download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)



# Download and load the test data

testset = datasets.FashionMNIST('/F_MNIST_data/', download=True, train=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
image, label = next(iter(trainloader))

imshow(image[0,:]);
# Create the network, define the criterion and optimizer



model = Network(784, 10, [512, 256, 128])

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, trainloader, testloader, criterion, optimizer, epochs=2)
print("Our model: \n\n", model, '\n')

print("The state dict keys: \n\n", model.state_dict().keys())
torch.save(model.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth')

print(state_dict.keys())
model.load_state_dict(state_dict)
# Try this

model = Network(784, 10, [400, 200, 100])

# This will throw an error because the tensor sizes are wrong!

model.load_state_dict(state_dict)
checkpoint = {'input_size': 784,

              'output_size': 10,

              'hidden_layers': [each.out_features for each in model.hidden_layers],

              'state_dict': model.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    model = Network(checkpoint['input_size'],

                             checkpoint['output_size'],

                             checkpoint['hidden_layers'])

    model.load_state_dict(checkpoint['state_dict'])

    

    return model
model = load_checkpoint('checkpoint.pth')

print(model)