import numpy as np
import pandas as pd

from tqdm import tqdm
import json
import os
from os.path import join as path_join


def load_data(path):
    tasks = pd.Series()
    for file_path in os.listdir(path):
        task_file = path_join(path, file_path)

        with open(task_file, 'r') as f:
            task = json.load(f)

        tasks[file_path[:-5]] = task
    return tasks
train_tasks = load_data('../input/abstraction-and-reasoning-challenge/training/')
evaluation_tasks = load_data('../input/abstraction-and-reasoning-challenge/evaluation/')
test_tasks = load_data('../input/abstraction-and-reasoning-challenge/test/')

train_tasks.head()
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import Conv2d
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

class CAModel(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()      
        self.simple_conv = nn.Sequential(
            nn.Conv2d(num_states, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(32, num_states, kernel_size=1)
        self.conv2 = nn.Conv2d(32, num_states+1, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x, steps = 1):
        for step in range(steps):
            x = self.simple_conv(x)
            y1 = self.conv1(x)
            y2 = self.conv2(x)
            x = self.soft(y1)
        return y1, y2
import matplotlib.pyplot as plt
from matplotlib import colors


cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#530C25'])
norm = colors.Normalize(vmin=0, vmax=10)
    
def plot_pictures(pictures, labels):
    fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures),32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()
    

def plot_sample(sample, predict=None):
    if predict is None:
        plot_pictures([sample['input'], sample['output']], ['Input', 'Output'])
    else:
        plot_pictures([sample['input'], sample['output'], predict], ['Input', 'Output', 'Predict'])
def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp==i)
    return img

def target_maker(samin, samout):
    target = np.full((len(samin), len(samin[0])), 0, dtype=np.uint8)
    for i in range(len(samin)):
        for j in range(len(samin[0])):
            if samin[i][j]==samout[i][j]:
                target[i,j] = 10  #index 10 is for retained info. see pred_maker.
            else:
                target[i,j] = samout[i][j]
    return target

def pred_maker(inp, pred, pred_grade):
    inp = np.asarray(inp)
    final_pred = np.full((inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    grade_avg = np.mean(pred_grade)
    threshold = 0.66
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i,j] == 10 or pred_grade[i, j]/grade_avg < threshold: #index=10
                final_pred[i,j] = inp[i,j]
            else:
                final_pred[i,j] = pred[i,j]
    return final_pred
    
class TaskSolver:        
    def train(self, task_train, n_epoch=100, max_steps=2):
        """basic pytorch train loop"""
        self.net = CAModel(10)
        criterion = CrossEntropyLoss()
        optimizer = Adam(self.net.parameters(), lr = 0.01) #(0.1 / (num_steps * 2)))
            
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            for sample in task_train:
                #Our target is label - input. This makes it easier.
                target = target_maker(sample['input'],sample['output'])
                inputs = FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0)
                labels1 = LongTensor(sample['output']).unsqueeze(dim=0)
                labels2 = LongTensor(target).unsqueeze(dim=0)
                outputs1 = self.net(inputs)[0]
                outputs2 = self.net(inputs)[1]
                loss = criterion(outputs1, labels1)
                loss.backward()
                loss = criterion(outputs2, labels2)
                loss.backward()
                optimizer.step()

        #output_img = outputs.squeeze(dim=0).detach().cpu().numpy().argmax(0)
        #plot_pictures([sample['input'], sample['output'],target, output_img ], ['Input', 'label', 'target', 'Predict'])
            
        #output_img = outputs.detach().numpy()
        #output_img = np.squeeze(output_img)
        #output_img = np.argmax(output_img, axis=0)
        #plot_pictures([sample['input'],target,output_img],['input','target','predict'])
            
        return self
            
    def predict(self, task_test):
        predictions = []
        pred_grades = []
        with torch.no_grad():
            for sample in task_test:
                inputs = FloatTensor(inp2img(sample['input'])).unsqueeze(dim=0)
                outputs = self.net(inputs)[0]
                #outputs = self.net(inputs)[1]
                pred =  outputs.squeeze(dim=0).cpu().numpy().argmax(0)
                
                #pred_grade = np.full((pred.shape[0], pred.shape[1]), 0, dtype=np.float64)
                #for i in range(pred.shape[0]):
                #    for j in range(pred.shape[1]):
                #        k = pred[i][j]
                #        pred_grade[i][j] = (torch.nn.Softmax(dim=1)(outputs)).squeeze(dim=0).cpu().numpy()[k][i][j]
                
                #final_pred = pred_maker(sample['input'],pred, pred_grade)
                final_pred = pred
                predictions.append(final_pred)
                
                #predictions.append(pred)
                
        return predictions
def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])

def calk_score(task_test, predict):
    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]
def evaluate(tasks):
    ts = TaskSolver()
    result = []
    predictions = []
    for task in tqdm(tasks):
        if input_output_shape_is_same(task):
            ts.train(task['train'])
            pred = ts.predict(task['test'])
            score = calk_score(task['test'], pred)
        else:
            pred = [el['input'] for el in task['test']]
            score = [0]*len(task['test'])
        
        predictions.append(pred)
        result.append(score)
       
    return result, predictions
train_result, train_predictions = evaluate(train_tasks)
train_solved = [any(score) for score in train_result]

total = sum([len(score) for score in train_result])
print(f"solved : {sum(train_solved)} from {total} ({sum(train_solved)/total})")
evaluation_result, evaluation_predictions = evaluate(evaluation_tasks)
evaluation_solved = [any(score) for score in evaluation_result]

total = sum([len(score) for score in evaluation_result])
print(f"solved : {sum(evaluation_solved)} from {total} ({sum(evaluation_solved)/total})")
for task, prediction, solved in tqdm(zip(train_tasks, train_predictions, train_solved)):
    if solved:
        for i in range(len(task['train'])):
            plot_sample(task['train'][i])
            
        for i in range(len(task['test'])):
            plot_sample(task['test'][i], prediction[i])
for task, prediction, solved in tqdm(zip(evaluation_tasks, evaluation_predictions, evaluation_solved)):
    if solved:
        for i in range(len(task['train'])):
            plot_sample(task['train'][i])
            
        for i in range(len(task['test'])):
            plot_sample(task['test'][i], prediction[i])
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

def make_pediction(tasks):
    ts = TaskSolver()
    result = pd.Series()
    for idx, task in tqdm(test_tasks.iteritems()):
        if input_output_shape_is_same(task):
            ts.train(task['train'])
            pred = ts.predict(task['test'])
        else:
            pred = [el['input'] for el in task['test']]
        
        for i, p in enumerate(pred):
            result[f'{idx}_{i}'] = flattener(np.array(p).tolist())
       
    return result
submission = make_pediction(test_tasks)
submission.head()
submission = submission.reset_index()
submission.columns = ['output_id', 'output']
submission.to_csv('submission.csv', index=False)
submission
for task, prediction in tqdm(zip(train_tasks, train_predictions)):
    if input_output_shape_is_same(task):
        for i in range(len(task['test'])):
            plot_sample(task['test'][i], prediction[i])