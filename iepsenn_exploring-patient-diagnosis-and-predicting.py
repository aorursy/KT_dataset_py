# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')

df.head()
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image



labels = ["age", "resting blood pressure", "serum cholestoral", "maximum heart rate achieved"]

fig = ff.create_distplot([df.age, df.trestbps, df.chol, df.thalach], labels, bin_size=0.5)

fig['layout'].update(title="dataset distribution")

iplot(fig, filename="Distributions")
df.columns

#labels = ["age", "resting blood pressure", "serum cholestoral", "maximum heart rate achieved"]

#fig = ff.create_distplot([df.age, df.trestbps, df.chol, df.thalach], labels, bin_size=0.5)

t0 = go.Box(y=df.age, name="Ages")

t1 = go.Box(y=df.trestbps, name="Bld. pressure")

t2 = go.Box(y=df.chol, name="Cholestoral")

t3 = go.Box(y=df.thalach, name="Max heart rate")

data=[t0,t1,t2,t3]



fig = {

    'data' : data, 

    'layout' : {

        'title' : "dataset stats",

        'yaxis' : {

            'zeroline' : False

        }

    }

}





iplot(fig)
heartDisease = df[df.target == 1] 

noHeartDisease = df[df.target == 0]



t0 = {

    'type' : 'violin',

    'x' : max(heartDisease.age),

    'y' : heartDisease.age.values,

    'name' : 'age distribution of patients with heart disease',

    'box' : {

        'visible' : True

    },

    "meanline" : {

        'visible' : True

    }

}



t1 = {

    'type' : 'violin',

    'x' : max(noHeartDisease.age),

    'y' : noHeartDisease.age.values,

    'name' : 'age distribution of patients without heart disease',

    'box' : {

        'visible' : True

    },

    "meanline" : {

        'visible' : True

    }

}



data = [t0,t1]



fig = {

    'data' : data, 

    'layout' : {

        'title' : 'age distribution',

        'yaxis' : {

            'zeroline' : False

        }

    }

}



iplot(fig, filename="ages", validate=False)
older = heartDisease[["age", "trestbps", "chol", "thalach"]].sort_values(by='age', ascending=False).iloc[0,:]

younger = heartDisease[["age", "trestbps", "chol", "thalach"]].sort_values(by='age', ascending=True).iloc[0,:]



w_older = noHeartDisease[["age", "trestbps", "chol", "thalach"]].sort_values(by='age', ascending=False).iloc[0,:]

w_younger = noHeartDisease[["age", "trestbps", "chol", "thalach"]].sort_values(by='age', ascending=True).iloc[0,:]



print("with heart disease:\n---older\n{}\n---younger\n{}\n\n".format(older, younger))

print("without heart disease:\n---older\n{}\n---younger\n{}".format(w_older, w_younger))


#heartDisease = df[df.target == 1] 

labels = ["age", "resting blood pressure", "cholestoral", "maximum heart rate achieved"]



#to_radar = df[["age", "trestbps", "chol", "thalach"]]

older_with_heart_disease = go.Scatterpolar(

    r = older,

    theta = labels,

    fill = "toself",

    name = "older with heart disease"

)



older_without_heart_disease = go.Scatterpolar(

    r = w_older,

    theta = labels,

    fill = "toself",

    name = "older without heart disease"

)



data = [older_with_heart_disease, older_without_heart_disease]



layout = go.Layout(

    polar = dict(

        radialaxis = dict(

            visible = True,

            range = [0, 300]

        )

    ),

    showlegend = True,

    title = "older people"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="Stats")


#heartDisease = df[df.target == 1] 

labels = ["age", "resting blood pressure", "cholestoral", "maximum heart rate achieved"]



#to_radar = df[["age", "trestbps", "chol", "thalach"]]

younger_with_heart_disease = go.Scatterpolar(

    r = younger,

    theta = labels,

    fill = "toself",

    name = "younger with heart disease"

)



younger_without_heart_disease = go.Scatterpolar(

    r = w_younger,

    theta = labels,

    fill = "toself",

    name = "younger without heart disease"

)



data = [younger_with_heart_disease, younger_without_heart_disease]



layout = go.Layout(

    polar = dict(

        radialaxis = dict(

            visible = True,

            range = [0, 300]

        )

    ),

    showlegend = True,

    title = "younger people"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="Stats")
#age

#sex

#cpchest -> pain type

#trestbpsresting -> blood pressure

#cholserum -> cholestoral

#fbs -> (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

#restecgresting -> electrocardiographic results

#thalachmaximum -> heart rate achieved

#exangexercise -> induced angina (1 = yes; 0 = no)

#oldpeakST -> depression induced by exercise relative to rest

#slopethe -> slope of the peak exercise ST segment

#ca -> number of major vessels (0-3) colored by flourosopy

#thal -> 3 = normal; 6 = fixed defect; 7 = reversable defect

#target -> 1 or 0

#for c in heartDisease.columns

titles =    ["pain type",

            "blood pressure",

            "cholestoral",

            "fasting blood sugar > 120 mg/dl",

            "electrocardiographic results",

            "heart rate achieved",

            "induced angina",

            "depression induced by exercise relative to rest",

            "slope of the peak exercise ST segment",

            "number of major vessels",

            "3 = normal; 6 = fixed defect; 7 = reversable defect"]



print("Features Overview")

for c, title in zip(heartDisease.columns[2:-1], titles):

    #plt.title("RMS")

    #x = np.arange(0,epochs)

    #plt.plot(x,train_loss_RMS,label = "train")

    #plt.plot(x,test_loss_RMS,label = "test")

    #plt.xlabel("epochs")

    #plt.ylabel("loss")

    #plt.legend()

    #plt.show()

    plt.title(str(title))

    plt.hist(heartDisease[c].value_counts(), alpha=0.75, label="with")

    plt.hist(noHeartDisease[c].value_counts(), alpha=0.75, label="without")

    plt.legend()

    plt.show()
train = df.sample(220)

test = df.sample(100)



x = train.iloc[:,:-1].values

y = train.iloc[:,-1:].values



x_train = x[:-100]

y_train = y[:-100]



x = test.iloc[:,:-1].values

y = test.iloc[:,-1:].values



x_test = x[-100:]

y_test = y[-100:]



print("train shapes:\nx {}\ny {}\n".format(x_train.shape, y_train.shape))

print("test shapes:\nx {}\ny {}\n".format(x_test.shape, y_test.shape))
import torch

import torchvision



x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)

x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)



print(x_train.type())

print(y_train.type())

print(x_test.type())

print(y_test.type())
class Net(torch.nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = torch.nn.Linear(13, 128)

        self.fc2 = torch.nn.Linear(128, 64)

        self.fc3 = torch.nn.Linear(64, 1)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):

        x = self.activation(self.fc1(x))

        x = self.activation(self.fc2(x))

        x = self.activation(self.fc3(x))

        return x
model = Net()

print(model)
# Loss and Optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = torch.nn.BCELoss()

print(optimizer)

print(loss_fn)
for param in model.parameters():

    param.requires_grad = True
def step(x, threshold=0.5):

    if x >= threshold:

        return 1

    else:

        return 0
# training the model

n_epochs = 100



for epoch in range(n_epochs):

    loss_per_epoch = 0

    preds = []

    model.train()

    for patient, diagnose in zip(x_train, y_train):

        optimizer.zero_grad()

        

        pred = model(patient)

        #print(pred)

        loss = loss_fn(pred, diagnose)

        #print(loss)

        loss.backward(retain_graph=True)

        optimizer.step()

        

        with torch.no_grad():

            loss_per_epoch += loss.item()

            preds.append(step(pred.item()))

            

    model.eval()

    acc = sum([1 if a==b else 0 for a,b in zip(preds, y_train)]) / len(x_train)

    loss_per_epoch /= len(x_train)

    print("Epoch {} Loss {} Acc on Train {}".format(epoch, loss_per_epoch, acc))
preds_test = []

for x, y in zip(x_test, y_test):

    pred = model(x)

    preds_test.append(step(pred.item()))

    

acc = [1 if a==b else 0 for a,b in zip(preds_test, y_test)]

print("Accuracy: {}".format(sum(acc)/len(y_train)))