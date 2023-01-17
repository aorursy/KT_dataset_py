! pip install --quiet colored
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from colored import fore, style
plt.style.use('fivethirtyeight')
data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data.head()
data.info()
# Look at the descriptive statistics of the data
data.describe()
# Drop useless column for further visualization and also check for any null values
data = data.drop(['Serial No.'], axis=1)
data.isna().sum()
# A pairplot visualizes how each variable depends on other variables (If you have no idea what that is, pick a stats book for god's sake)
sns.pairplot(data)
# This is a heatmap.
# It shows the correlation between different variables at play
fig = plt.figure(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True)
fig.show()
print("Correlation in a nutshell: ")
print(fore.GREEN+"More Correlation between 2 features => More closely they affect each other and vice-versa"+style.RESET)
feature_importance = dict(data.corr()['Chance of Admit '])
sort_orders = sorted(feature_importance.items(), key=lambda x: x[1])
sort_orders.pop()

print(fore.GREEN+f"Most Important feature for getting selected is: {sort_orders[-1][0]}"+style.RESET)
print(fore.RED+f"Least Important feature for getting selected is: {sort_orders[0][0]}"+style.RESET)
# Order of Most Important to Least Important Features
print("Following are the features from most important to least important (Darker Blue Shade = More Important) and (Lesser Blue Shade = Less Important)")
i=len(sort_orders)-1
colors = [fore.BLUE_VIOLET, fore.VIOLET, fore.BLUE, fore.GREEN, fore.YELLOW, fore.ORANGE_1, fore.RED][::-1]
while i>=0:
    print(colors[i]+f"{sort_orders[i][0]}"+style.RESET)
    i-=1
# Split the data
X = data.drop('Chance of Admit ', axis=1).values
y = data['Chance of Admit '].values
# Just taking 5% of the total data for validation
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.05)
# Define a linear regression model
class LinearRegressionTorch(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out
# And Hyperparamters
inp_dim = 7
op_dim = 1
learningRate = 0.001
epochs = 15000
# See if CUDA is available
model = LinearRegressionTorch(inp_dim, op_dim)
try:
    model.cuda()
except AssertionError:
    print("GPU isn't enabled")
# Define the loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
# Reshape the labels
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
verbose=True
all_loss = []
for epoch in range(epochs+1):
    
    # Convert the data into torch tensors
    inputs = Variable(torch.from_numpy(x_train)).float()
    label = Variable(torch.from_numpy(y_train)).float()
    
    # Clear the existing Gradients from the optimizer
    optimizer.zero_grad()
    
    # Pass the data thorugh model and get one set of predictions
    output = model(inputs)
    
    # Calculate the loss from the obtained predictions and the ground truth values
    loss = torch.sqrt(criterion(output, label))
    
    # Calculate gradients by doing one step of back propagation
    loss.backward()
    
    # Apply those gradients to the weights by doing one step of optimizer
    optimizer.step()
    
    # Add the current loss to the list of all loses (used later for prediction)
    all_loss.append(loss)
    
    # For monitoring and debugging
    if verbose and epoch % 1000 == 0:
        print(f"Epoch: {epoch}  |  Loss: {loss}")
# Test the model and compute Validation Accuracy
VAL_inp = Variable(torch.from_numpy(x_val)).float()

y_pred = model(VAL_inp).detach().numpy()

# Calculate R^2 Accuracy (used for regression where discrete values are absent)
rss = sum((y_val - y_pred)**2)       # Residual Sum of Squares
tss = sum((y_val - y_val.mean())**2) # Total Sum of Squares

r2_accuracy = (1 - rss / tss)
print(f"Validation Accuracy of the model is: {r2_accuracy.squeeze() * 100} %")
plt.plot(all_loss)
# Let's now use sklearn's Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)
# Measure the model accuracy (the same R^2 Accuracy) using score method
model.score(x_val, y_val)
# Save the sklearn model
joblib.dump(model, "sklearn_model_college_adm.sav")
