import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
df['OutcomeCat'] = np.where(df.Outcome == 1,"Diabetic","Not Diabetic")
sns.pairplot(df,hue='OutcomeCat')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Outcome','OutcomeCat'],axis=1),df.Outcome,test_size  = 0.2, random_state=0)
import torch
import torch.nn as nn
import torch.nn.functional as F
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)
class Model(nn.Module):
    def __init__(self,input_features = 8, hidden1 = 32, hidden2 = 32, hidden3 = 64,out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.f_connected3 = nn.Linear(hidden2,hidden3)
        self.out = nn.Linear(hidden3,out_features)
    
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = F.relu(self.f_connected3(x))
        x = self.out(x)
        return x
torch.manual_seed(20)
model = Model()
model.parameters
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)
epochs = 500
final_losses = []
for i in range(epochs):
    i += 1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred,y_train)
    final_losses.append(loss)
    if i%10 == 0:
        print(f"Epoch {i} : {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
predictions = []
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)
cm
sns.heatmap(cm,annot=True)
plt.xlabel('Actual')
plt.ylabel('Predicted')