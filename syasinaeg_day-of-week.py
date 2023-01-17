import numpy as np

import csv

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from random import shuffle
INPUT_DAYS = 28

OUPUT_DAYS = 7

MIN_DAYS = INPUT_DAYS + OUPUT_DAYS



BATCH_SIZE = 512



num_of_days = 1099

last_day = num_of_days - 1
data = []

with open("train.csv", "r") as file:

    read = csv.reader(file)

    head = next(read, None)

    for row in read:

        data.append((int(row[0]) - 1, np.array([(int(i) - 1) for i in row[1].strip().split(" ")], dtype=np.short)))
datamatrix = np.zeros((len(data), num_of_days), dtype=np.byte)

for p, visits in data:

    datamatrix[p, visits] = 1
histograms = np.empty((len(data), 7))

for i in range(7):

    indices = list(range(i, last_day, 7))

    histograms[:, i] = np.sum(datamatrix[:,indices], axis=1)

histograms /= histograms.sum(axis=1, keepdims=True)
visits_per_day = np.sum(datamatrix, axis=0)

visits_per_day = visits_per_day / np.sum(visits_per_day) * num_of_days / 7
class Network(nn.Module):

    

    def __init__(self, input_dim, output_dim, fc_sizes):

        super().__init__()

        self.input_dim, self.output_dim = input_dim, output_dim

        fc_params = [(inp, out) for (inp, out) in zip([self.input_dim] + fc_sizes, fc_sizes + [self.output_dim])]

        self.linears = nn.ModuleList([nn.Linear(inp, out) for (inp, out) in fc_params])

        

    def forward(self, x):

        for l in self.linears[:-1]:

            x = F.relu(l(x))

        x = self.linears[-1](x)

        return x
def make_x(person, first_day):

    return np.concatenate([ datamatrix[person, first_day:first_day + INPUT_DAYS], 

                            histograms[person], 

                            visits_per_day[first_day:first_day + INPUT_DAYS] ])
def make(person, first_day):

    x = np.concatenate([ datamatrix[person, first_day:first_day + INPUT_DAYS], 

                         histograms[person], 

                         visits_per_day[first_day:first_day + INPUT_DAYS] ])

    return x, (np.argmax(datamatrix[person, (first_day + INPUT_DAYS):]) + first_day) % 7
def generate_first_day(visits):

    from random import randint

    size = len(visits)

    return 0 if (size <= MIN_DAYS) else (randint(0, size - MIN_DAYS) // 7) * 7        
def make_batch(data_slice):

    data_slice = [(p, v) for (p, v) in data_slice if (len(v) >= MIN_DAYS)]

    batch = [make(p, generate_first_day(v)) for (p, v) in data_slice]

    x = torch.FloatTensor([inp for (inp, _) in batch])

    y = torch.LongTensor([out for (_, out) in batch])

    return x, y
def batch_generator(data, batch_size):

    for start_index in range(0, len(data), batch_size):

        yield make_batch(data[start_index:start_index + batch_size])
model = Network(INPUT_DAYS * 2 + 7, OUPUT_DAYS, [64, 48, 32, 16])

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
for epoch in range(50):

    output_rate = 500

    shuffle(data)

    running_loss = 0.0

    for batch_number, (inputs, labels) in enumerate(batch_generator(data, BATCH_SIZE)):

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        if ((batch_number % output_rate) == (output_rate - 1)):

            print("[{:3d}, {:5d}] loss: {:.5f}".format(epoch + 1, batch_number + 1, running_loss / output_rate))

            running_loss = 0.0



    for g in optimizer.param_groups:

        g["lr"] *= 0.95
def slice_generator(data, length):

    for start_index in range(0, len(data), length):

        yield data[start_index:start_index + length]



answers = []



for b in slice_generator(data, BATCH_SIZE):

    persons = [p for (p, v) in b]

    with torch.no_grad():

        x = torch.stack([torch.from_numpy(make_x(p, last_day - INPUT_DAYS + 1)).float() for p in persons])

        outputs = model(x)

        answers.extend(zip(persons, torch.argmax(outputs, dim=1).tolist()))
with open("solution.csv", "w") as file:

    file.write("id,nextvisit\n")

    for person, visit in answers:

        file.write("{}, {}\n".format(person + 1, ((visit) % 7) + 1))