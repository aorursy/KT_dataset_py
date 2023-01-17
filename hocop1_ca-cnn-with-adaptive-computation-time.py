import os

import json

import numpy as np

from pathlib import Path



import torch

import torch.nn as nn

import torch.nn.functional as F



from tqdm import tqdm

import matplotlib.pyplot as plt

from matplotlib import colors

from matplotlib import animation, rc

from IPython.display import HTML



rc('animation', html='jshtml')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge')



train_path = data_path / 'training'

eval_path = data_path / 'evaluation'

test_path = data_path / 'test'



train_tasks = { task.stem: json.load(task.open()) for task in train_path.iterdir() }

eval_tasks = { task.stem: json.load(task.open()) for task in eval_path.iterdir() }

test_tasks = { task.stem: json.load(task.open()) for task in test_path.iterdir() }
cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

    

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



def input_output_shape_is_same(task):

    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])





def calk_score(task_test, predict):

    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]
task = train_tasks["db3e9e38"]["train"]

for sample in task:

    plot_sample(sample)
HIDDEN_SIZE = 128

MAX_STEPS = 10

THRESHOLD = 0.99

REMAINDERS_PEN = 0.0



# Fix random seeds for reproducibility

torch.manual_seed(42)

np.random.seed(42)



class CAModel(nn.Module):

    def __init__(self, num_states):

        super(CAModel, self).__init__()

        self.embedding = nn.Sequential(

            nn.Conv2d(num_states, HIDDEN_SIZE, kernel_size=1),

            nn.InstanceNorm2d(HIDDEN_SIZE),

        )

        self.transition = nn.Sequential(

            nn.InstanceNorm2d(HIDDEN_SIZE),

            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=1, padding=0),

        )

        self.projection_out = nn.Conv2d(HIDDEN_SIZE, num_states, kernel_size=1)

        self.projection_halt = nn.Conv2d(HIDDEN_SIZE, 1, kernel_size=1)

        

    def forward(self, x, max_steps=None):

        x = self.embedding(x)

        # Initialize values

        halting_probability = torch.zeros([1, 1, x.shape[2], x.shape[3]], device=x.device)

        remainders = torch.zeros([1, 1, x.shape[2], x.shape[3]], device=x.device)

        n_updates = torch.zeros([1, 1, x.shape[2], x.shape[3]], device=x.device)

        # Cycle

        max_steps = max_steps or MAX_STEPS

        for i in range(max_steps):

            p = torch.sigmoid(self.projection_halt(x) - 1)

            # Formulas from https://arxiv.org/pdf/1807.03819.pdf APPENDIX C

            still_running = (halting_probability <= THRESHOLD).to(torch.float)

            new_halted = ((halting_probability + p * still_running) > THRESHOLD).to(torch.float) * still_running

            still_running = ((halting_probability + p * still_running) <= THRESHOLD).to(torch.float) * still_running

            halting_probability += p * still_running

            remainders += new_halted * (1 - halting_probability)

            halting_probability += new_halted * remainders

            n_updates += still_running + new_halted

            update_weights = p * still_running + new_halted * remainders

            # Apply transformation to the state

            transformed_state = self.transition(x)

            # Interpolate transformed and previous states for non-halted inputs

            x = transformed_state * update_weights + x * (1 - update_weights)

            # Halt

            if still_running.sum() == 0:

                break

        

        x = self.projection_out(x)

        self._remainders = remainders

        self._n_updates = n_updates

        return x
def solve_task(task, max_steps=10):

    model = CAModel(10).to(device)

    num_epochs = 100

    criterion = nn.CrossEntropyLoss()

    losses = np.zeros(num_epochs)

    n_updates = np.zeros(num_epochs)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    for e in range(num_epochs):

        optimizer.zero_grad()

        loss = 0.0

        

        for sample in task:

            # predict output from input

            x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)

            y = torch.tensor(sample["output"]).long().unsqueeze(0).to(device)

            y_pred = model(x)

            loss += criterion(y_pred, y) + (model._remainders * REMAINDERS_PEN).mean(0).sum()

            n_updates[e] += model._n_updates.detach().cpu().mean().numpy() / len(task)

        

        loss.backward()

        optimizer.step()

        losses[e] = loss.item()

    return model, losses, n_updates



@torch.no_grad()

def predict(model, task):

    predictions = []

    for sample in task:

        x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)

        pred = model(x).argmax(1).squeeze().cpu().numpy()

        predictions.append(pred)

    return predictions

    

task = train_tasks["db3e9e38"]["train"]

model, losses, n_updates = solve_task(task)
plt.plot(losses)
plt.plot(n_updates)
predictions = predict(model, task)

for i in range(len(task)):

    plot_sample(task[i], predictions[i])
test = train_tasks["db3e9e38"]["test"]

predictions = predict(model, test)

for i in range(len(test)):

    plot_sample(test[i], predictions[i])
def animate_solution(model, sample):

    x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)



    @torch.no_grad()

    def animate(i):

        pred = model(x, i)

        im.set_data(pred.argmax(1).squeeze().cpu().numpy())



    fig, ax = plt.subplots()

    im = ax.imshow(x.argmax(1).squeeze().cpu().numpy(), cmap=cmap, norm=norm)

    return animation.FuncAnimation(fig, animate, frames=100, interval=120)

    

anim = animate_solution(model, train_tasks["db3e9e38"]["test"][0])

HTML(anim.to_jshtml())
def evaluate(tasks, is_test=False):

    result = []

    predictions = {}

    for idx, task in tqdm(tasks.items()):

        if input_output_shape_is_same(task):

            model, _, _ = solve_task(task["train"])

            pred = predict(model, task["test"])

            if not is_test:

                score = calk_score(task["test"], pred)

            else:

                score = [0] * len(task["test"])

        else:

            pred = [el["input"] for el in task["test"]]

            score = [0] * len(task["test"])



        predictions[idx] = pred

        result.append(score)

    return result, predictions
train_result, train_predictions = evaluate(train_tasks)

train_solved = [any(score) for score in train_result]



total = sum([len(score) for score in train_result])

print(f"solved : {sum(train_solved)} from {total} ({sum(train_solved)/total})")
eval_result, eval_predictions = evaluate(eval_tasks)

eval_solved = [any(score) for score in eval_result]



total_eval = sum([len(score) for score in eval_result])

print(f"Eval: solved : {sum(eval_solved)} from {total_eval} ({sum(eval_solved)/total})")
test_result, test_predictions = evaluate(test_tasks, is_test=True)
for task, prediction, solved in tqdm(zip(train_tasks.values(), train_predictions.values(), train_solved)):

    if solved:

        for i in range(len(task['train'])):

            plot_sample(task['train'][i])

            

        for i in range(len(task['test'])):

            plot_sample(task['test'][i], prediction[i])
import pandas as pd

submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')

display(submission.head())
def flattener(pred):

    str_pred = str([list(row) for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
for output_id in submission.index:

    task_id = output_id.split('_')[0]

    pair_id = int(output_id.split('_')[1])

    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:

        task = json.load(read_file)

    # skipping over the training examples, since this will be naive predictions

    # we will use the test input grid as the base, and make some modifications

    #data = task['test'][pair_id]['input'] # test pair input

    data = test_predictions[task_id][pair_id]

    # for the first guess, predict that output is unchanged

    pred_1 = flattener(data)

    # for the second guess, change all 0s to 5s

    data = [[5 if i==0 else i for i in j] for j in data]

    pred_2 = flattener(data)

    # for the last gues, change everything to 0

    data = [[0 for i in j] for j in data]

    pred_3 = flattener(data)

    # concatenate and add to the submission output

    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 

    submission.loc[output_id, 'output'] = pred



submission.to_csv('submission.csv')

submission.head()