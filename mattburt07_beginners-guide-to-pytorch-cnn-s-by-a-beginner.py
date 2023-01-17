import torch

import torch.nn as nn                         

# torch.nn contains the classes we will use to actually build the CNN

from torch.autograd import Variable           

# for recording the operations used to create tensors, for computing

# the gradients through backpropagation (deprecated)

import torch.utils.data as Data               

# for creating our dataset and our dataloader 

import torchvision                            

# contains some datasets (such as MNIST we use here) and some common 

# transforms

import matplotlib.pyplot as plt               

# matplotlib for plotting %matplotlib inline means you can plot in the

# notebook

%matplotlib inline
torch.manual_seed(1)
trainData = torchvision.datasets.MNIST(root='./mnist/',train = True,transform = torchvision.transforms.ToTensor(),download = True)

trainData.data.size()

    
plt.imshow(trainData.data[111].numpy(), cmap='gray')

plt.title('%d' % trainData.targets[111])

trainData.data[111].shape, trainData.data[111].numpy().shape

trainLoader = Data.DataLoader(dataset=trainData, batch_size=30, shuffle=True)
testData = torchvision.datasets.MNIST(root='./mnist/', train=False, transform = torchvision.transforms.ToTensor())

testData.data.size()
test_x = torch.unsqueeze(testData.data, dim=1).type(torch.FloatTensor)[:]/255 

# need to convert to FloatTensor.

test_x = Variable(test_x)

test_x.data.size()



# Can also use view

# test_x = testData.data.view((10000, -1, 28, 28))

# test_x.data.size()
test_y = testData.targets[:]

test_y.size()
class CNN(nn.Module):   # the class we create will inherit from nn.Module

    def __init__(self):

        super(CNN, self).__init__()                        

        # we always include this line of code. Lets us use the base class functions.

        

        self.conv1 = nn.Sequential(                        

        

        # the input shape will be (1, 28, 28)

        nn.Conv2d(1, 8, 3, 1, 1),                   

        # the output shape will be (8, 28, 28)

            

        # The Conv2d layer above takes 5 numbers in this case. (input_channels=1, 

        # output_channels=8, kernel_size=3, stride=1, padding=1)

        # So the input image is 1 channel x 28 x 28 pixels. Padding=1 means we add 

        # a row/column of zeros to every side to prevent the output

        # from losing a row/column as the kernel is 3x3. If the kernel was 5x5 we 

        # would need to increase padding to 2. 

        # Stride of 1 means we move the kernel over by one pixel at a time. 

        # If we wanted to half the output to 14 x 14 we could set stride to

        # 2. Lastly, 8 output channels means we pass 8 convolutional kernels 

        # over the image to end up with 8 x (1 x 28 x 28) -> (8 x 28 x 28)

           

        # ReLU and BatchNorm: I will try to add a better explanation in the future :-)

            

        nn.ReLU(),

            

        # A ReLU (Rectified Linear Unit) is a non-linear activation function that, in 

        # simple terms, just throws away the negatives. It can

        # be represented as f(x) = max(0, x).

            

        nn.BatchNorm2d(8),

            

        # BatchNorm (from what I understand) normalises the activations of the layer 

        # by transforming them to have 0 mean and unit variance

        # (i.e. 0 mean SD of 1). The issue is that each batch that passes through 

        # the network may not have the same distributions.

        # Normalising the activations is meant to help the network train faster by 

        # giving it more uniform activation distributions.

        # Alternatively to the above, BatchNorm makes models train faster as it smooths 

        # the optimisation landscape (that crazy graph people

        # show you when they talk about gradient descent). Instead of an irregular 

        # graph, with many local minima and bumps, BatchNorm 

        # gives us a smooth graph. So This means when you train, the gradients are 

        # more predictable and stable.

        )

        

        # I have 4 convolutional layers in this CNN. Not really sure how to choose

        # the structure but this one seems to work okay. If anyone has any ideasd

        # on how to decide on a structure let me know.

        

        self.conv2 = nn.Sequential(     # the input shape will bee (8, 28, 28)

        nn.Conv2d(8, 16, 3, 1, 1),      # the output shape will be (16, 28, 28)

        nn.ReLU(),

        nn.BatchNorm2d(16),             # needs to match output channels above

        )

        self.conv3 = nn.Sequential(     # the input shape will be (16, 28, 28)

        nn.Conv2d(16, 32, 3, 1, 1),     # the output shape will be (32, 28, 28)

        nn.ReLU(),

        nn.BatchNorm2d(32),                       

        nn.MaxPool2d(2)                 

        # the max pool output is (32, 14, 14). Max pooling takes the max value of a

        # square of cells and throws the rest away. This effectively means the width and

        # height of the shape is halved in this case as our max pool is 2 x 2.

        )

        self.conv4 = nn.Sequential( # the input shape will be (32, 14, 14)

        nn.Conv2d(32, 32, 3, 1, 1),    # the output shape will be (32, 14, 14)

        nn.ReLU(),

        nn.BatchNorm2d(32),

        )

        self.out = nn.Linear(32 * 14 * 14, 10)          

        # this takes the conv4 output from 6272 features down to 10 for our 10 digits



    # the forward function basically defines how our network works. It takes the CNN above

    # and each mini batch x, and passes it through the series of functions we defined above.

    # It then returns the output function (our predicted digit). If you prefer, you could

    # have your ReLU and BatchNorm here instead of above like x = ReLU(self.conv1(x)).



    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        # We have to reshape our conv4 output to fit it into the linear layer.

        output = self.out(x)

        return output, x
mnist_cnn = CNN()

print(mnist_cnn)
lossF = nn.CrossEntropyLoss()
epoch = 1
def TrainCNN(dataLoader, model, num_epochs, loss_function, lr):

    for i in range(num_epochs):

        # Enumerate took me awhile to get my head around. But it basically returns a counter

        # with each of the mini batches from my dataloader of the size I set as batch size. 

        # I.e. it will return -> 1, (images minibatch 1, labels minibatch 1),

        # -> 2, (images minibatch 2, labels minibatch 2) until we have gone through all of

        # the data in the dataloader.

        for step, (images,labels) in enumerate(dataLoader):

            images_x = Variable(images)

            labels_y = Variable(labels)



            output = model(images_x)[0]

            # output is the 10 predictions from our CNN.

            loss = loss_function(output, labels_y)

            # loss is the loss of our predictions vs actual.

            optimiser = torch.optim.Adam(model.parameters(), lr)

            # Creating the optimiser and feeding in the parameters and learning rate.

            optimiser.zero_grad()

            # This zeroes out the gradients as they would accumulate otherwise.

            loss.backward()

            # This computes the derivatives of the parameters of the model.

            optimiser.step()

            # Updates the parameters of the model by adding the learning rate multiplies

            # by the derivative of each parameter.



            if step % 500 == 0: # Only does this every 500 minibatches.

                test_output, last_layer = model(test_x) 

                # Gets our 10 predictions of each x in the nth minibatch

                pred_y = torch.max(test_output, 1)[1].data.squeeze()

                # Finds the maximum probability and records as prediction for all.

                accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))

                # Compares to the actual to work out how many percent are correct.

                print('Epoch: '+ str(i) + '| test accuracy: %.4f' % accuracy)

                # Prints an accuracy to 4 dp.

for i in range(1,4):

    lr = 1e-2/(i*10)

    # Decreases the lr by factor of 10 each iteration.

    TrainCNN(trainLoader, mnist_cnn,1,lossF, lr)

    