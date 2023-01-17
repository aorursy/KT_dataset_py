import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

df = pd.read_csv('../data/sat_gpa.csv')

sat = df['SAT']
gpa = df['GPA']

plt.scatter(gpa, sat,c="g", alpha=0.5, label="SAT vs GPA")
plt.xlabel("GPA")
plt.ylabel("SAT")
plt.legend(loc='upper left')
plt.show()

def plot_losses(losses, min_y = None, max_y = None):
    plt.plot(losses, label="Loss (MSE)")
    plt.xlabel("iteration")
    plt.ylabel("MSE")
    if min_y is not None and max_y is not None:
        plt.ylim((min_y, max_y))
    plt.show()
import torch.nn

# convert data to tensors
x_data = torch.tensor(df['GPA'].values.reshape(-1,1), dtype=torch.float)
y_data = torch.tensor(df['SAT'].values.reshape(-1,1), dtype=torch.float)

# build model
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
losses = []
num_epochs = 10_000

for i, epoch in enumerate(range(num_epochs)):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(x_data.float())
    # Compute Loss
    loss = criterion(y_pred.float(), y_data.float())
    losses.append(loss)
    # Backward pass
    loss.backward()
    optimizer.step()
    
    
print(f"Final train loss: {loss.item():.3f}")
plot_losses(losses, min_y=6_000, max_y=20_000)
def normalize(data):
    return (data - data.mean()) / data.std()

def unnormalize(data, mean, std):
    return data * std + mean

def print_arr_stats(data):
    print(f"Tensor: {data.tolist()} Shape: {data.shape}\nMean: {data.mean().item()} Std: {data.std().item()}")

 # Let's test it out.
test_arr = torch.tensor([1, 2, 1, 2], dtype=torch.float)
test_arr_norm = normalize(test_arr)
test_arr_unnorm = unnormalize(test_arr_norm, test_arr.mean(), test_arr.std())

for arr in [test_arr, test_arr_norm, test_arr_unnorm]:
    print_arr_stats(arr)
    print("")
x_data_norm = normalize(x_data)
y_data_norm = normalize(y_data)

x_data_unnorm = unnormalize(x_data_norm, x_data.mean(), x_data.std())

# Print the first 3 to make sure it worked
list(zip(x_data, x_data_unnorm))[:3]
# Now let's train!

losses = []
num_epochs = 200
check_10_x = num_epochs/10

# Reset model and optimizer by recreating or we'll be fine tuning
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

model.train()

for i, epoch in enumerate(range(num_epochs)):
    
    optimizer.zero_grad()
    
    y_pred = model(x_data_norm)
    
    loss = criterion(y_pred, y_data_norm)
    
    losses.append(loss)
    loss.backward()
    
    optimizer.step()
    
    if i > 0 and i % check_10_x == 0:
        print(f"Train loss @ iter {i}: {loss.item():.3f}")
        
plot_losses(losses)
def train(lr, num_epochs=200):
    
    losses = []
    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
    
    model.train()

    for i, epoch in enumerate(range(num_epochs)):
        optimizer.zero_grad()   
        y_pred = model(x_data_norm)  
        loss = criterion(y_pred, y_data_norm)  
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return losses
lrs = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
loss_lrs = [train(lr, num_epochs=1000) for lr in lrs] 

# Now let's plot.
for lr, lr_losses in zip(lrs, loss_lrs):
    plt.plot(lr_losses, label=f"{lr:.0e}")
plt.legend()
plt.show()
lr_loss = list((lr, min(ll)) for (lr, ll) in zip(lrs, loss_lrs)) 
min_loss = min(lr_loss, key=lambda x: x[1])

print("Losses:")
print(lr_loss)
print(f"\nLowest loss: {min_loss[0]:.0e} = {min_loss[1]:.2f}")
def train(model, x_train, y_train, lr, num_epochs=200):
    
    losses = []
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
    
    model.train()

    for i, epoch in enumerate(range(num_epochs)):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return losses

def validate(model, x_valid, y_valid):
    
    losses = []
    model.eval()

    with torch.no_grad():
        y_pred = model(x_valid)
        loss = criterion(y_pred, y_valid)
        losses.append(loss.item())

    return losses
train_size = int(x_data_norm.shape[0] * 0.95) # 95% train, 5% valid

x_train = x_data_norm[:train_size, :]
y_train = y_data_norm[:train_size, :]

x_valid = x_data_norm[train_size:, :]
y_valid = y_data_norm[train_size:, :]

x_train.shape, y_train.shape, x_valid.shape, y_valid.shape
model = torch.nn.Linear(1, 1)
train_losses = train(model, x_train, y_train, 1e-2)
valid_losses = validate(model, x_valid, y_valid)

print(f"Train loss: {train_losses[-1]:.2f}")
print(f"Valid loss: {valid_losses[-1]:.2f}")
new_x = torch.Tensor([[1]])
y_pred = model(new_x)

plt.scatter(x_data_norm, y_data_norm, c="g", alpha=0.5, label="SAT vs GPA")
plt.scatter(new_x.detach().numpy(),y_pred.detach().numpy(),  c="r")
plt.xlabel("GPA")
plt.ylabel("SAT")
plt.legend(loc='upper left')
plt.show()

# different ways to print weights (results) 

for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)

for param in model.parameters():
    print(param.data)


# weight 
w= list(model.parameters())[0].data.numpy()[0]

# bias
b = list(model.parameters())[1].data.numpy()[0]

# or 
[w,b] = model.parameters()

predicted_norm = model(x_data_norm).data.numpy()

# prediction of y given x
new_x = torch.Tensor([[1]])
new_y = model(new_x)

# graph data points
plt.scatter(x_data_norm, y_data_norm, c="g", alpha=0.5, label="SAT vs GPA")

# graph regression line
plt.plot(x_data_norm, predicted_norm, c="orange", label="Regression")

# graph predicted new point
plt.scatter(new_y.detach().numpy(), new_x.detach().numpy(), c="r", label="Predicted ")


plt.xlabel("SAT")
plt.ylabel("GPA")
plt.legend(loc='upper left')

plt.show()
