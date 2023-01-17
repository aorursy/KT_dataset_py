# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import scipy.io
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
mnist = scipy.io.loadmat('../input/mnist-original/mnist-original.mat')
X_init = mnist["data"] / 255.
Y_init = mnist["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X_init.T, Y_init.T, test_size=0.2, random_state=42)
#Converting into torch tensors
X_train = torch.from_numpy(X_train).float()
X_train = torch.reshape(X_train, (-1, 1, 28, 28))
Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
X_test = torch.from_numpy(X_test).float()
X_test = torch.reshape(X_test, (-1, 1, 28, 28))
Y_test = torch.from_numpy(Y_test).type(torch.LongTensor)
X_train.shape
#Defining the safe norm and squash as described in the paper
def safe_norm(s, axis=-1, epsilon=1e-7, keepdim=False):
    squared_norm = torch.sum(torch.square(s), axis=axis, keepdim=keepdim)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    return safe_norm

def squash(s, axis=-1, epsilon=1e-7):
    squared_norm = torch.sum(torch.square(s), axis=axis, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector
#Performs the first half of routing by agreement
def get_capsule_prediction(raw_weights, caps2_predicted):
    
    #the sum of the predictions of each capsule for all the next ones is 1
    routing_weights = torch.nn.functional.softmax(raw_weights, dim=2)
    
    #applying the weights
    weighted_predictions = routing_weights * caps2_predicted
    
    #summing up all the predictions for each secondary capsule
    weighted_sum = torch.sum(weighted_predictions, dim=1, keepdim=True)
    
    return squash(weighted_sum, axis=-2)
    
#Measuring the similarity between global and local predictions and adjusting the weights
def routing_by_agreement(raw_weights, caps2_predicted):
    
    #getting global predictions
    caps2_output = get_capsule_prediction(raw_weights, caps2_predicted)
    
    #repeating global prediction for number primary caps
    caps2_output_round_tiled = caps2_output.expand(-1, caps1_n_caps, -1, -1, -1)
    
    #scalar product to measure agreement between local and global
    caps2_predicted_transposed = torch.transpose(caps2_predicted, -1, -2)
    agreement = torch.matmul(caps2_predicted_transposed, caps2_output_round_tiled)
    
    return raw_weights + agreement
#HYPERPARAMETERS
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8
caps2_n_caps = torch.tensor(10)
caps2_n_dims = 16
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28
alpha = 0.0005
n_epochs = 5
batch_size = 64
restore_checkpoint = True
checkpoint_path = "/kaggle/working/my_capsule_network"
class CapsNetModel(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        #Initial 2 conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=9, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        
        #Initialising transformation weights and routing weights
        self.pose_w_init = torch.normal(0, 0.1, \
            size = (1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims)).cuda()
        self.raw_routing_w = torch.zeros((batch_size, caps1_n_caps, caps2_n_caps, 1, 1)).cuda()
        
        #Decoder layers
        self.dense1 = nn.Linear(caps2_n_caps * caps2_n_dims, n_hidden1)
        self.dense1_act = nn.ReLU()
        self.dense2 = nn.Linear(n_hidden1, n_hidden2)
        self.dense2_act = nn.ReLU()
        self.dense3 = nn.Linear(n_hidden2, n_output)
        self.dense3_act = nn.ReLU()
    
    def forward(self, X, Y, mask_with_labels):
        #initial convolutional layers
        conv1 = self.conv1(X)
        relu1 = self.relu1(conv1)
        conv2 = self.conv2(relu1)
        relu2 = self.relu2(conv2)
        
        #organising in capsules
        caps1_raw = torch.reshape(relu2, (-1, caps1_n_caps, caps1_n_dims))
        caps1_output = squash(caps1_raw)
        
        #expanding the weights to the whole batch
        batch_size = X.shape[0]
        W_tiled = self.pose_w_init.expand(batch_size, -1, -1, -1, -1)
        
        #expanding the output and adding a dimension for digit capsule
        caps1_output_unsqueezed = torch.unsqueeze(torch.unsqueeze(caps1_output, -1), 2)
        caps1_output_tiled = caps1_output_unsqueezed.expand(-1, -1, caps2_n_caps, -1, -1)
        
        #getting predictions
        caps2_predicted = torch.matmul(W_tiled, caps1_output_tiled)
        
        #routing by agreement
        new_weights = routing_by_agreement(self.raw_routing_w, caps2_predicted)
        caps2_output = get_capsule_prediction(new_weights, caps2_predicted)
        
        #marginal loss
        y_proba = safe_norm(caps2_output, axis=-2)
        y_proba_argmax = torch.argmax(y_proba, axis=2)
        y_pred = torch.squeeze(y_proba_argmax)   
        
        #only true labels during training
        reconstruction_targets = torch.where(mask_with_labels, Y, y_pred)

        #constructing the mask and adapting it to the dimentionality of the secondary layer predictions
        reconstruction_mask = torch.nn.functional.one_hot(reconstruction_targets, caps2_n_caps)
        reconstruction_mask_reshaped = torch.reshape(reconstruction_mask, (-1, 1, caps2_n_caps, 1, 1))

        #APPLYING THE MASK
        caps2_output_masked = caps2_output * reconstruction_mask_reshaped
        decoder_input = torch.reshape(caps2_output_masked,(-1, caps2_n_caps * caps2_n_dims))
        
        #dense layers
        dense1_out = self.dense1(decoder_input)
        dense1_act = self.dense1_act(dense1_out)
        
        dense2_out = self.dense2(dense1_act)
        dense2_act = self.dense2_act(dense2_out)
        
        dense3_out = self.dense3(dense2_act)
        dense3_act = self.dense3_act(dense3_out)
        
        return caps2_output, dense3_act
def capsule_loss(caps2_output, dense3_act, X, Y):
    #measure the confidency of the prediction
    caps2_output_norm = safe_norm(caps2_output, axis=-2, keepdim=True)
        
    #see if something has confidency over .9
    present_error_raw = torch.square(torch.max(torch.tensor(0.).cuda(), m_plus - caps2_output_norm))
    present_error = torch.reshape(present_error_raw, shape = (-1, 10))
        
    #see if something has confidency less than .1
    absent_error_raw = torch.square(torch.max(torch.tensor(0.).cuda(), caps2_output_norm - m_minus))
    absent_error = torch.reshape(absent_error_raw, shape = (-1, 10))
        
    T = torch.nn.functional.one_hot(Y, 10)
    #composite error
    L = T * present_error + lambda_ * (1.0 - T) * absent_error
    margin_loss = torch.mean(torch.sum(L, axis=1))
    
    #reconstruction loss
    X_flat = torch.reshape(X, (-1, n_output))
    squared_difference = torch.square(X_flat - dense3_act)
    reconstruction_loss = torch.mean(squared_difference)
    
    return margin_loss + alpha * reconstruction_loss
#TRAINING LOOP
n_iterations_per_epoch = X_train.shape[0] // batch_size

model = CapsNetModel().cuda()
optimiser = torch.optim.Adam(model.parameters())

for epoch in range(n_epochs):
    for iteration in range(1, n_iterations_per_epoch + 1):
        X_batch = X_train[(iteration-1)*batch_size:iteration*batch_size].cuda()
        Y_batch = torch.squeeze(Y_train[(iteration-1)*batch_size:iteration*batch_size]).cuda()
        
        caps2_output, dense3_output = model(X_batch, Y_batch, torch.tensor(True).cuda())
        loss = capsule_loss(caps2_output, dense3_output, X_batch, Y_batch)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print("epoch: " + str(epoch) + " iteration: " + str(iteration) + " loss: " + str(loss))

#Saving the model
checkpoint = {'model': CapsNetModel(),
              'state_dict': model.state_dict(),
              'optimizer' : optimiser.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
#Evaluating against test set
def eval(model):
    
    n_iterations_per_validation = X_test.shape[0] // batch_size
    acc_vals=[]
    
    #Making predictions and measuring accuracy
    for iteration in range(1, n_iterations_per_validation + 1):
        X_batch = X_test[(iteration-1)*batch_size:iteration*batch_size].cuda()
        Y_batch = torch.squeeze(Y_test[(iteration-1)*batch_size:iteration*batch_size]).cuda()
        caps2_output, dense3_output = model(X_batch, Y_batch, torch.tensor(False).cuda())
        y_proba = safe_norm(caps2_output, axis=-2)
        y_proba_argmax = torch.argmax(y_proba, axis=2)
        y_pred = torch.squeeze(y_proba_argmax)  
        accuracy = torch.mean(torch.eq(Y_batch, y_pred).type(torch.FloatTensor))
        acc_vals.append(accuracy)
        
    return torch.mean(torch.tensor(acc_vals))
print("ACCURACY: " + str(eval(model)*100) + "%")
#Retrieve model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model

print(eval(model))
#Show reconstructions
n_samples = 5

#Selecting a batch
sample_images = X_test[:batch_size]
sample_labels = torch.squeeze(Y_test[:batch_size])

#Getting results from the network
caps2_output, dense3_output = model(sample_images.cuda(), sample_labels.cuda(), torch.tensor(True).cuda())
reconstructions = dense3_output.detach().cpu().numpy()
sample_images = sample_images.reshape(-1, 28, 28)
reconstructions = reconstructions.reshape([-1, 28, 28])
y_proba = safe_norm(caps2_output, axis=-2)
y_proba_argmax = torch.argmax(y_proba, axis=2)
y_pred = torch.squeeze(y_proba_argmax) 
y_pred = y_pred.detach().cpu().numpy()

#plotting original with labels
plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(Y_test[index]))
    plt.axis("off")

plt.show()

#plotting predicted with labels
plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")
    
plt.show()