# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Input a tensor and convert a tensor to the corresponding picture
#for visual convenience
def tensor2picture(input_tensor):
    from torchvision import transforms
    im = transforms.ToPILImage()(input_tensor)
    display(im)
#Look up your version, which sometimes really bothers.
print(torch.__version__)
print(np.__version__)
#Data loading and processing to tensors.
#Adjust frac to change training sample number
all_data_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv');
test_set_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv');
train_set_df = all_data_df.sample(frac=0.6, axis=0);
selected_rows = list(train_set_df.index.values);
validation_set_df = all_data_df.drop(selected_rows, axis=0)
data_num = train_set_df.shape[0];
print(f"There are {data_num} samples to be trained.")
#Subtract labels from the data frame and convert the input data to
#tensors 
train_label_idx = train_set_df['label'].values
validation_label_idx = validation_set_df['label'].values
train_set = torch.from_numpy(train_set_df.drop('label', axis=1).values).float();
validation_set = torch.from_numpy(validation_set_df.drop('label', axis=1).values).float();
test_set = torch.from_numpy(test_set_df.values).float();
train_label = torch.from_numpy(train_label_idx).long();
validation_label = torch.from_numpy(validation_label_idx).long();
#Function to normalize input tensor
def train_data_normalization(input_data):
    mean = input_data.mean();
    std = input_data.std();
    return mean, std, (input_data - mean)/std;
#run this cell if you want to have normalized training data
#Make sure all dataset are normalized in the same way.
mean, std, train_set = train_data_normalization(train_set);
validation_set = (validation_set - mean)/std;
test_set = (test_set - mean)/std
#Model class, feel free to modify input layers and their hidden units just
#by input a list of number representing hidden units number for each layer
class perceptron(nn.Module):
    def __init__(self, hidden_layers, input_size):
        super().__init__();
        self.model = nn.Sequential();
        self.input_size = input_size;
        prev = input_size;
        curr = hidden_layers[0];
        for i in range(len(hidden_layers)-1):
            self.model.add_module('Linear'+str(i+1), nn.Linear(prev, curr));
            self.model.add_module('ReLU'+str(i+1), nn.ReLU());
            prev = curr;
            curr = hidden_layers[i+1];
        self.model.add_module('Output', nn.Linear(hidden_layers[-2], hidden_layers[-1]));
        print(self.model)
    
    def forward(self, input_data):
        assert len(input_data.size()) == 2
        assert input_data.size(1) == self.input_size
        input_data = input_data.view(-1, self.input_size)
        return self.model(input_data)
#Ignore this cell for further development of CNN
class conv_nn(nn.Module):
    def __init__(self, out_chan_1=3, out_chan_2=6, kernel_size=5, pool_size=2):
        super().__init__();
        pointer = torch.rand(8,1,28,28)
        self.model = nn.Sequential();
        self.pool = nn.MaxPool2d(pool_size, pool_size);
        self.model.add_module('conv1', nn.Conv2d(1, out_chan_1, kernel_size));
        self.model.add_module('Pooling1', self.pool);
        self.model.add_module('conv2', nn.Conv2d(out_chan_1, out_chan_2, kernel_size));
        self.model.add_module('Pooling2', self.pool);
        self.linear_layer = nn.Sequential();
        self.linear_layer.add_module('Linear1', nn.Linear(out_chan_2*((31-3*kernel_size)//4)**2, 64));
        self.linear_layer.add_module('ReLU', nn.ReLU());
        self.linear_layer.add_module('Linear2', nn.Linear(64, 16));
        self.linear_layer.add_module('ReLU', nn.ReLU());
        self.linear_layer.add_module('Linear3', nn.Linear(16, 10));
        print(self.model)
    
    def forward(self, input_data):
        if len(input_data.size()) != 4:
            input_data = input_data.view(input_data.size(0), 1, 28, 28);
        input_data = self.model(input_data);
        input_data = input_data.view(input_data.size(0), -1);
        return self.linear_layer(input_data);     
#Major training process for training the model
#model is the model to be used, currently we have perceptrons only
#criterion now is hard code to be nn.CrossEntropyLoss for multi-classification
#batch_size: Adjust number of inputs in a batch, default to be 16
#epoch: Number of epochs to train the model, default to be 1
#verbose: True for output loss and training process when training, False for no output
#plot_loss: True for plotting the loss curve throughout training, False for no plot showing
def train(model, criterion, optimizer, input_data, input_label, epoch=1, batch_size=16, verbose=True, plot_loss=True):
    model.train()
    axis1 = [];
    axis2 = [];
    plt.xlabel('iteration num');
    plt.ylabel('iter_loss');
    import time;
    start = time.time()
    for EPOCH in range(epoch):
        data_num = input_data.size(0);
        iterations = data_num//batch_size;
        if data_num%batch_size:
            iterations += 1;
        for i in range(iterations):
            optimizer.zero_grad()
            start = i*batch_size;
            end = min((i+1)*batch_size, data_num);
            batch_input = input_data[start: end];
            batch_label = input_label[start: end];
            y_pred = model(batch_input);
            loss = criterion(y_pred, batch_label);
            if plot_loss:
                axis1.append(EPOCH*iterations + i);
                axis2.append(loss.item());
            if i%1000 == 0 and verbose:
                print(f'Epoch{EPOCH}: {i}/{iterations} iterations, train loss = {round(loss.item(), 7)}, training time = {round((time.time() - start)/60, 2)}');
            loss.backward();
            optimizer.step();
    if plot_loss:
        plt.plot(axis1, axis2);
        plt.show();
    
#Evaluate the prediction accuracy on given dataset and model
#show_num: Number of random samples from evaluation dataset to be 
#          shown, both picture, prediction, and the label
def evaluate(model, criterion, eval_input, eval_label, show_num=3):
    model.eval();
    y_pred = model(eval_input);
    predict = torch.argmax(y_pred, dim=1);
    for i in range(show_num):
        chosen = torch.randint(0, eval_input.size(0), (1, 1)).item()
        img = eval_input[chosen:chosen+1].view(28, 28);
        tensor2picture(img);
        print(f'The prediction is {predict[chosen]}, and the true label is {eval_label[chosen]}')
    result = torch.sum((predict == eval_label)).item()/predict.size(0)
    print(f'Validation accuracy = {round(result*100.0, 7)}%')
    loss = criterion(y_pred, eval_label);
    print(f'Test loss after training is: {round(loss.item(), 7)}');
    return predict
#Cell to run the whole process including training and validation set evaluation
#All hyperparameters should be adjusted here.
#Input: perceptron for using MLP, CNN for using convolutional neural network
def run_epoch(network='perceptron'):
    #Tuning your hyperparameters here
    if network == 'perceptron':
        model = perceptron([128, 64, 16, 10], 784);
    elif network == 'CNN':
        model = conv_nn(out_chan_1=6, out_chan_2=16, kernel_size=7, pool_size=2);#To be finished
#         train_set = train_set.view(train_set.size(0), 1, 28, 28);
#         validation_set = validation_set.view(validation_set.size(0), 1, 28, 28);
#         test_set = test_set.view(test_set.size(0), 1, 28, 28);
    else:
        print('Undefined neural net, please input correct name!');
        return
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005);
    train(model, criterion, optimizer, train_set, train_label, batch_size=8, epoch=32)
    evaluate(model, criterion, validation_set, validation_label)
    return model, criterion
model, criterion = run_epoch(network='CNN');
evaluate(model, criterion, validation_set, validation_label, show_num=10)
def visualize_test(model, test_input, show_num=3, random=True):
    model.eval();
    predict = torch.argmax(model(test_input), dim=1);
    if random:
        for i in range(show_num):
            chosen = torch.randint(0, test_input.size(0), (1,1)).item();
            img = test_input[chosen:chosen+1].view(28, 28);
            tensor2picture(img);
            print(f'The predicted label is {predict[chosen]}')
    else:
        for i in range(show_num):
            img = test_input[i:i+1].view(28, 28);
            tensor2picture(img);
            print(f'The predicted label is {predict[i]}')
visualize_test(model, test_set, show_num=10, random=False)
#Test the test data given by kaggle and generate the formatted result for
#submission.
def test(model, test_input, output_file):
    model.eval();
    y_pred = model(test_input);
    predict = torch.argmax(y_pred, dim=1);
    idx = [i+1 for i in range(predict.size(0))]
    result = {'ImageId':idx, 'Label':predict}
    result = pd.DataFrame(result)
    result.to_csv(output_file, index=False)        
test(model, test_set, 'cnn_6_16_7.csv')
torch.save(model, 'cnn_6_16_7.pt')

