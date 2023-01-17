import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
def banner(msg, _verbose=1):
    if not _verbose:
        return
    print("-"*80)
    print(msg.upper())
    print("-"*80)
df = pd.read_csv('../input/Housing.csv', index_col=0)
def convert_to_binary(string):
    return int('yes' in string)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(convert_to_binary)
data = df.values

scaler = StandardScaler()
data = scaler.fit_transform(data)
X = data[:, 1:]
y = data[:, 0]

print("X: ", X.shape)
print("y: ", y.shape)
def get_torch_variable(x):
    return torch.from_numpy(x).double()
X_train, X_valid, y_train, y_valid = map(get_torch_variable, train_test_split(X, y, test_size=0.2))
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_valid: ", X_valid.shape)
print("y_valid: ", y_valid.shape)
class LinearRegression:
    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        
        self.Theta = torch.randn((X_train.shape[1]+1)).type(type(X_train)).double()
        
    def _add_bias(self, tensor):
        bias = torch.ones((tensor.shape[0], 1)).double()
        return torch.cat((bias, tensor), 1)
        
    def _forward(self, tensor):
        return torch.matmul(
            self._add_bias(tensor),
            self.Theta
        ).view(-1)
    
    def forward(self, train=True):
        if train:
            return self._forward(self.X_train)
        else:
            return self._forward(self.X_valid)
    
    def _cost(self, X, y):
        y_hat = self._forward(X)
        mse = torch.sum(torch.pow(y_hat - y, 2))/2/X.shape[0]
        return mse
    
    def cost(self, train=True):
        if train:
            return self._cost(self.X_train, self.y_train)
        else:
            return self._cost(self.X_valid, self.y_valid)
        
    def batch_update_vectorized(self):
        m, _ = self.X_train.size()
        return torch.matmul(
                self._add_bias(self.X_train).transpose(0, 1),
                (self.forward() - self.y_train)
            ) / m
    
    def batch_update_iterative(self):
        m, _ = self.X_train.size()
        update_theta = None
        X = self._add_bias(self.X_train)
        for i in range(m):
            if type(update_theta) == torch.DoubleTensor:
                update_theta += (self._forward(self.X_train[i].view(1, -1)) - self.y_train[i]) * X[i]
            else:
                update_theta = (self._forward(self.X_train[i].view(1, -1)) - self.y_train[i]) * X[i]
        return update_theta/m
        
    
    def batch_train(self, tolerance=0.01, alpha=0.01):
        converged = False
        prev_cost = self.cost()
        init_cost = prev_cost
        num_epochs = 0
        while not converged:
            self.Theta = self.Theta - alpha * self.batch_update_vectorized()
            cost = self.cost()
            if (prev_cost - cost) < tolerance:
                converged = True
            prev_cost = cost
            num_epochs += 1
        banner("Batch")
        print("\tepochs: ", num_epochs)
        print("\tcost before optim: ", init_cost)
        print("\tcost after optim: ", cost)
        print("\ttolerance: ", tolerance)
        print("\talpha: ", alpha)
            
    def stochastic_train(self, tolerance=0.01, alpha=0.01):
        converged = False
        m, _ = self.X_train.size()
        X = self._add_bias(self.X_train)
        init_cost = self.cost()
        num_epochs=0
        while not converged:
            prev_cost = self.cost()
            for i in range(m):
                self.Theta = self.Theta - alpha * (self._forward(self.X_train[i].view(1, -1)) - self.y_train[i]) * X[i]
            cost = self.cost()
            if prev_cost-cost < tolerance:
                converged=True
            num_epochs += 1
        banner("Stochastic")
        print("\tepochs: ", num_epochs)
        print("\tcost before optim: ", init_cost)
        print("\tcost after optim: ", cost)
        print("\ttolerance: ", tolerance)
        print("\talpha: ", alpha)
        
    def mini_batch_train(self, tolerance=0.01, alpha=0.01, batch_size=8):
        converged = False
        m, _ = self.X_train.size()
        X = self._add_bias(self.X_train)
        init_cost = self.cost()
        num_epochs=0
        while not converged:
            prev_cost = self.cost()
            for i in range(0, m, batch_size):
                self.Theta = self.Theta - alpha / batch_size * torch.matmul(
                    X[i:i+batch_size].transpose(0, 1),
                    self._forward(self.X_train[i: i+batch_size]) - self.y_train[i: i+batch_size]
                )
            cost = self.cost()
            if prev_cost-cost < tolerance:
                converged=True
            num_epochs += 1
        banner("Stochastic")
        print("\tepochs: ", num_epochs)
        print("\tcost before optim: ", init_cost)
        print("\tcost after optim: ", cost)
        print("\ttolerance: ", tolerance)
        print("\talpha: ", alpha)
%%time
l = LinearRegression(X_train, y_train, X_valid, y_valid)
l.mini_batch_train()
%%time
l = LinearRegression(X_train, y_train, X_valid, y_valid)
l.stochastic_train()
%%time
l = LinearRegression(X_train, y_train, X_valid, y_valid)
l.batch_train()