# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import count

from random import shuffle

import time

from copy import deepcopy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def dateToInt(date):

    days = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31]

    month, day = tuple(int(i) for i in date.split('-')[1:])

    return sum(days[:month]) + day - 22

dateToInt("2020-01-22")

dateToInt("2020-04-04")
def findDuplicates(l):

    duplicates = list()

    idxs = list()

    seen = dict()

    for i, element in enumerate(l):

        if element in seen:

            idx = seen[element]

            duplicates.append(i)

            idxs.append(idx)

        seen.update({element: i})

    return duplicates, idxs

findDuplicates([1, 2, 3, 4, 1, 2, 3, 4])
# Installs adabound

import sys

from subprocess import call

call([sys.executable, '-m', 'pip', 'install', 'adabound'])
testname = "/kaggle/input/covid19-global-forecasting-week-4/test.csv"

ptest = pd.read_csv(testname)

testnames = [p[1] if type(p[1]) is str else p[2] for p in ptest.to_numpy()]

ptest
trainname = "/kaggle/input/covid19-global-forecasting-week-4/train.csv"

ptrain = pd.read_csv(trainname)

nptrain = ptrain.to_numpy()

names = set()

provinces = set()

pdatas = dict()

for data in nptrain:

    name = data[2]

    names.add(name)

for name in names:

    pdatas.update({name: ptrain[ptrain["Country_Region"] == name].to_numpy()})

for name, data in list(pdatas.items()):

    for d in data:

        state = d[1]

        if type(state) is float or state in provinces:

            continue

        try:

            if name not in ("Canada",) and name not in testnames:

                del pdatas[name]

                names.remove(name)

        except:

            pass

        names.add(state)

        provinces.add(state)

        pdatas.update({state: ptrain[ptrain["Province_State"] == state].to_numpy()})

counter = 0

stuff = [0 for i in range(72)]

for i in pdatas["Illinois"]:

    stuff[counter % 72] += i[-2]

    counter += 1

# pdatas["California"]

oldxbycountry = dict()

oldybycountry = dict()

xbycountry = dict()

ybycountry = dict()

shufflexbycountry = dict()

shuffleybycountry = dict()



for name in names:

    data = pdatas[name]

    countryx = [dateToInt(p[3]) for p in data]

    countryy = [p[4:] for p in data]

    oldxbycountry.update({name: countryx})

    oldybycountry.update({name: countryy})

l = len(oldxbycountry["Italy"])

for name in names:

    scheme = list(range(l))

    shuffle(scheme)

    newx = list(0 for i in range(l))

    newy = list([0, 0] for i in range(l))

    shufflex = list(0 for i in range(l))

    shuffley = list([0,0] for i in range(l))

    for i, x, y in zip(count(), oldxbycountry[name], oldybycountry[name]):

        newx[i%l] = x

        newy[i%l][0] += y[0]

        newy[i%l][1] += y[1]

        shufflex[scheme[i%l]] = x

        shuffley[scheme[i%l]][0] += y[0]

        shuffley[scheme[i%l]][1] += y[1]

    xbycountry.update({name: np.array(newx)})

    ybycountry.update({name: np.array(newy)})

    shufflexbycountry.update({name: np.array(shufflex)})

    shuffleybycountry.update({name: np.array(shuffley)})

len(ybycountry["California"])

xbycountry["California"]
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import adabound
def train(model, criterion, optimizer, x, y, limit):

    beg = time.time()

    try:

        for epoch in count():

            model.train()

            optimizer.zero_grad()

            y_pred = model(x)

            loss = criterion(y_pred, y)

            if epoch % 5000 == 0:

                print("Epoch:", epoch + 0*120000, "\tLoss:", loss.item())

            loss.backward()

            optimizer.step()

            seconds = time.time() - beg

            if seconds > limit:

                break

    except KeyboardInterrupt:

        pass

    return model, criterion, optimizer

class Model(nn.Module):

    def __init__(self, deg=1):

        super(Model, self).__init__()

        self.fc1 = nn.Linear(deg, 2)

        self.sigmoid = nn.Sigmoid()

        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):

        return self.fc2(self.sigmoid(self.fc1(x/70)))



loosex = torch.FloatTensor([[pow(i, p) for p in (2,1)] for i in shufflexbycountry["Canada"]])

loosey = torch.FloatTensor([[float(i[0]/1e4), float(i[1]/2e2)] for i in shuffleybycountry["Canada"]])



m = Model(2)

loosemodel = train(

    m,

    nn.MSELoss(),

    adabound.AdaBound(m.parameters(), lr=1e-3, final_lr=.1),

    loosex,

    loosey,

    10

)[0]

models = dict()

cnames = names

# cnames = ("Anguilla",)



for cname in cnames:

    if cname == "Canada":

        continue

    cases = max(i[0] for i in ybycountry[cname])

    small = cases < 30000

    rllysmall = cases < 5000

    xsmall = cases < 100

    print(small, rllysmall, xsmall)

    

    casenorm = 100 if xsmall else cases + .0001

    fatnorm = max(i[1] for i in ybycountry[cname]) + .001



    powers = (2,1) if small else (1,)



    beg = time.time()

    

#     train_limit = 100 if xsmall else 45 if rllysmall else 60 if small else 100

    train_limit = 100 if rllysmall else 30 if small else 60

#     train_limit = 3



    model = deepcopy(loosemodel) if rllysmall else Model(2 if small else 1)

    criterion = nn.MSELoss()

    sgd = lambda model: torch.optim.SGD(model.parameters(), lr=.1)

#     ada = lambda model: adabound.AdaBound(model.parameters(), lr=1e-2, final_lr=.1)

    adam = lambda model: torch.optim.Adam(model.parameters())



    x = torch.FloatTensor([[pow(i, p) for p in powers] for i in shufflexbycountry[cname]])

    y = torch.FloatTensor([[float(i[0]/casenorm), float(i[1]/fatnorm)] for i in shuffleybycountry[cname]])



    # Around 120000 epochs

    newmodel = train(

        model,

        criterion,

        adam(model),

        x,

        y,

        train_limit

    )[0]

    

    models.update({cname: (newmodel, powers, casenorm, fatnorm)})



    seconds = time.time() - beg

    print(cname, "trained in", seconds // 60, "minutes", seconds % 60, "seconds")

# Fatality

for cname in cnames:

    if cname == "Canada":

        continue

    model, powers, casenorm, fatnorm = deepcopy(models[cname])



    import matplotlib.pyplot as plt



    raw = model(torch.Tensor([[pow(xval, i) for i in powers] for xval in list(xbycountry[cname])])).detach()

    predictions = np.array([j.numpy()[1]*fatnorm for j in raw])

    plt.plot(np.array(list(range(150))), np.array([j.numpy()[1]*fatnorm for j in model(torch.Tensor([[pow(xval, i) for i in powers] for xval in list(range(150))])).detach()]), label="future")

    plt.plot(xbycountry[cname], [i[1] for i in ybycountry[cname]], label="actual")

    plt.plot(xbycountry[cname], predictions, label="model")

    # plt.plot(np.array([140 for i in range(200000)]), np.array(list(range(200000))), label="day 70")

    # plt.plot(np.array(list(range(200))), np.array([390000 for i in range(200)]), label="growth slows down")

    plt.xlabel = "Days since 20-01-22"

    plt.ylabel = "Cases"

    plt.legend()

    plt.show()

    # print(f"Will max out at {model(torch.Tensor([[102, 102**2]])).detach().numpy()[0][1]*8e4} infected around May 26 2020")

    print(f"{cname} average accuracy: {100*(1 - np.mean(abs(predictions - [i[1] for i in ybycountry[cname]]))/fatnorm):.3}%")

    del plt
# Cases



for cname in cnames:

    if "Canada" == cname:

        continue

    model, powers, casenorm, fatnorm = deepcopy(models[cname])



    

    import matplotlib.pyplot as plt

    

    predictions = np.array([j.numpy()[0]*casenorm for j in model(torch.Tensor([[pow(xval, i) for i in powers] for xval in list(xbycountry[cname])])).detach()])

    plt.plot(np.array(list(range(150))), np.array([j.numpy()[0]*casenorm for j in model(torch.Tensor([[pow(xval, i) for i in powers] for xval in list(range(150))])).detach()]), label="future")

    plt.plot(xbycountry[cname], [i[0] for i in ybycountry[cname]], label="actual")

    plt.plot(xbycountry[cname], predictions, label="model")

#     plt.plot(np.array([73 for i in range(int(casenorm) * 2)]), np.array(list(range(int(casenorm) * 2))), label="day 70")

    # plt.plot(np.array(list(range(200))), np.array([390000 for i in range(200)]), label="growth slows down")

    plt.xlabel = "Days Since 2020-01-22"

    plt.ylabel = "Cases"

    plt.legend()

    plt.show()

    # print(f"Will max out at {model(torch.Tensor([[[102]]])).item()*8e4} infected around May 26 2020")

    print(f"{cname} average accuracy: {100*(1-np.mean(abs(predictions - [i[0] for i in ybycountry[cname]]))/casenorm):.3}%")

    del plt
!rm /kaggle/working/submission.csv

stuff = []

with open("/kaggle/working/submission.csv", 'a+') as fout:

    fout.write("ForecastId,ConfirmedCases,Fatalities\n")

    for i, p in enumerate(ptest.to_numpy()):

        cname = p[1] if type(p[1]) is str else p[2]

        model, powers, casenorm, fatnorm = deepcopy(models[cname])

        date = dateToInt(p[3])

        predictions = np.array([(j.numpy()[0]*casenorm, j.numpy()[1]*fatnorm) for j in model(torch.Tensor([[pow(xval, i) for i in powers] for xval in (date,)])).detach()])

        print(i, cname, date, tuple(predictions[0]))

        print(i + 1, int(round(predictions[0][0])), int(round(predictions[0][1])), sep=',', file=fout)

        stuff.append((i+1, predictions[0]))

print(len(stuff))