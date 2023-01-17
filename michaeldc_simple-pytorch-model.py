import torch
inputs = torch.rand(1, 1, 128, 128)

outputs = torch.rand(1,23)
class Model(torch.nn.Module):

    def __init__(self):

        

        super().__init__()

        self.layer_one = torch.nn.Linear(128, 256)

        self.activation_one = torch.nn.ReLU()

        self.layer_two = torch.nn.Linear(256, 512)

        self.activation_two = torch.nn.ReLU()

        self.layer_three = torch.nn.Linear(512, 256)

        self.activation_three = torch.nn.ReLU()

        self.shape_outputs = torch.nn.Linear(128*256, 23)

        

    def forward(self, inputs):

        buffer = self.layer_one(inputs)

        buffer = self.activation_one(buffer)

        buffer = self.layer_two(buffer)

        buffer = self.activation_two(buffer)

        buffer = self.layer_three(buffer)

        buffer = self.activation_three(buffer)

        buffer = buffer.flatten(start_dim=1)

        return self.shape_outputs(buffer)
model = Model()

test_results = model(inputs)

test_results
loss_function = torch.nn.MSELoss()

alpha = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

EPOCHS = 1000



for i in range(EPOCHS):

    optimizer.zero_grad()

    results = model(inputs)

    loss = loss_function(results, outputs)

    loss.backward()

    optimizer.step()

    

    gradients = 0.0

    for parameter in model.parameters():

        gradients += parameter.grad.data.sum()

    if abs(gradients) <= 0.0001:

        print(gradients)

        print('gradient vanished at iteration {}'.format(i))

        break

    
model(inputs), outputs