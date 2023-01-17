import torch

from torch.autograd import Variable



x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))

y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))



class Model(torch.nn.Module):

    

    def __init__(self):

        super(Model, self).__init__()

        self.linear = torch.nn.Linear(1,1)

        

    # In the forward function we accept a Variable of input data and we must return

    # a Variable of output data. We can use Modules defined in the constructor as

    # well as arbitrary operators on Variables.

    

    def forward(self, x):

        y_pred = self.linear(x)

        return y_pred

    

model = Model()



# Construct our loss function and an Optimizer. The call to model.parameters()

# in the SGD constructor will contain the learnable parameters of the two

# nn.Linear modules which are members of the model.



criterion = torch.nn.MSELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



for epoch in range(500):

    

    # Forward pass: Compute predicted y by passing x to the model

    

    y_pred = model(x_data)

    

    # lets see how y_pred reaches actual y_data while training goes on for num of epochs

    

    if epoch%50 == 0:

        print("\n{}  {}".format(x_data.data[0][0], y_pred.data[0][0]))

        print("{}  {}".format(x_data.data[1][0], y_pred.data[1][0]))

        print("{}  {}\n".format(x_data.data[2][0], y_pred.data[2][0]))

    

    # Compute and print loss

    

    loss = criterion(y_pred, y_data)

    if epoch%50 == 0:

        print(epoch, loss.item())

        

    # Zero gradients, perform a backward pass, and update the weights.

    

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



# test on new data



test = Variable(torch.Tensor([[4.0]]))

y_pred = model(test)

print("\nprediction for {} is {}".format(test.data[0][0], y_pred.data[0][0]))