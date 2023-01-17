import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.utils import shuffle # to shuffle the training and validation data

import abc

from abc import ABC # abstract base class (to define interfaces in python)

import numpy as np



voting_mode = "Democracy"

#voting_mode = "Aristocracy"



validation_ratio = 0.2 # 0<x<1
data = pd.read_csv("../input/titanic/train.csv")

data = shuffle(data)

val_off = int(len(data)*(1-validation_ratio))

train_data = data[0:val_off]

val_data = data[val_off:]

val_target = val_data["Survived"].tolist()

val_data = val_data.drop(["Survived"], axis=1)



print("The labeled data has been randomly distributed to %i datapoints for training, and %i datapoints for validation. A total of %i/891 with %i duplicates."%(len(train_data), len(val_data), len(train_data)+len(val_data), len(pd.merge(train_data, val_data, how='inner'))))
test_data = pd.read_csv("../input/titanic/test.csv")



print("Testing data loaded. %i/418." % len(test_data))



print("Loading complete.")
votes = []

validation_scores = []

components = []
class ComponentI(ABC):

    

    @abc.abstractmethod

    def _name(self):

        pass

    

    # Gets labeled data

    # Returns nothing

    @abc.abstractmethod

    def train(self, df):

        pass

    

    # Gets unlabeled data

    # Returns [int] with prediction

    @abc.abstractmethod

    def test(self, df):

        pass
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import cuda

from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
def sex_to_float(sex):

    ret = 1.

    if sex == "male":

        ret = 0.

    return ret



def float_to_2d(x):

    #return np.array(x).astype("double").reshape(1)

    ret = (1.,0.)

    if x == 1:

        ret = (0., 1.)

    return np.array(ret).astype("double")



class TitanicDataset(Dataset):

    def __init__(self, df, features, start=0, end="none", indexed=False, labeled=False):

        data_frame = df[features]

        labels_frame = "unknown"

        

        if labeled:

            labels_frame = df[["Survived"]]

            

                

        if end == "none":

            data_frame = data_frame[start:]

            labels_frame = labels_frame[start:]

        else:

            data_frame = data_frame[start:end]

            labels_frame = labels_frame[start:end]

        

        self.inputs = []

        self.labels = []

        

        for idx in range(len(data_frame)):

            inputs = data_frame.iloc[idx,0:]

            inputs["Sex"] = sex_to_float(inputs["Sex"])

            inputs = np.array(inputs)

            inputs = np.nan_to_num(inputs.astype("double"))

            if labeled == False:

                label = []

            else:

                label = float_to_2d(labels_frame.iloc[idx]["Survived"])  

            

            self.inputs.append(inputs)

            self.labels.append(label)

        

        self.inputs = torch.tensor(self.inputs)

        self.labels = torch.tensor(self.labels)

        

        self.inputs = self.inputs.cuda()

        self.labels = self.labels.cuda()



            

        if indexed:

            self.PassengerId = torch.tensor(pd.read_csv(path)[["PassengerId"]]["PassengerId"])

        

        

        

    def __len__(self):

        return len(self.inputs)

    

    def __getitem__(self, idx):

    

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        return self.inputs[idx], self.labels[idx]

            
class Net1(nn.Module):

    def __init__(self, input_count):

        super(Net1, self).__init__()

        self.fc1 = nn.Linear(input_count, 200)

        self.fc2 = nn.Linear(200, 100)

        self.fc3 = nn.Linear(100, 2)

        self.deepthink = nn.Linear(100,100)

        self.fc4 = nn.Linear(2,2)



    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.deepthink(x))

        x = F.relu(self.fc3(x))            

        x = F.softmax(self.fc4(x), dim=1)

        return x

    





class PytorchSimpleNNComponent(ComponentI):

    def __init__(self, apx=""):

        self.name = "Pytorch Simple NN" + apx

        self.epochs = 50

        self.features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

        self.net = Net1(len(self.features))

        self.net = self.net.double()

        self.net.cuda()

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)

        

    def set_param(self, param, obj):

        if param == "net":

            self.net = obj

            self.net.cuda()

        elif param == "epochs":

            self.epochs = obj

        elif param == "criterion":

            self.criterion = obj

        elif param == "optimizer":

            self.optimizer = obj

        

        else:

            raise Exception("Invalid Argument", "Unknown parameter %s"%param)

    

    def _name(self):

        return self.name

    

    def train(self, df):

        tds = TitanicDataset(df, self.features, labeled=True)

        train_loader = torch.utils.data.DataLoader(tds, batch_size=50, shuffle=True)



        for epoch in range(self.epochs):  # loop over the dataset multiple times

    

            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data

                inputs = inputs.cuda()

                labels = labels.cuda()

                # zero the parameter gradients

                self.optimizer.zero_grad()

                # forward + backward + optimize

                outputs = self.net(inputs)

                #print("o",outputs)

                #print("l", labels)

                loss = self.criterion(outputs, labels)

                #print(loss)

                loss.backward()

                self.optimizer.step()



                # print statistics

                running_loss += loss.item()

                if i % 10 == 9 and epoch % 10 == 9:    # print every 100 mini-batches

                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))

                    running_loss = 0.0

    

    def test(self, df):

        tds = TitanicDataset(df, self.features)

        test_loader = torch.utils.data.DataLoader(tds, batch_size=4, shuffle=False)

        ret = [] 

        

        for data in test_loader:

            inputs, _ = data

            result = [torch.argmax(y) for y in self.net(inputs)]

            for i in range(len(result)):

                ret.append(int(result[i].item()))

        

        return ret

    

for i in range(50):

    basic = PytorchSimpleNNComponent(apx=str(i))

    components.append(basic)
class GenderComponent(ComponentI):

    def __init__(self):

        pass

    def _name(self):

        return "Genderer"

    

    def train(self, df):

        pass

    

    def test(self, df):

        ret = []

        for _, row in df.iterrows():

            if row["Sex"] == "male":

                ret.append(0)

            else:

                ret.append(1)

        return ret

    

gc = GenderComponent()

components.append(gc)
class BadComponent(ComponentI):

    def __init__(self):

        pass

    def _name(self):

        return "Bad"

    

    def train(self, df):

        pass

    

    def test(self, df):

        ret = []

        for _, row in df.iterrows():

            ret.append(1)

        return ret

    

# bc = BadComponent()

# components.append(bc)
for c in components:

    if not isinstance(c, ComponentI):

        raise Exception("Error", "Component %s is not of interface ComponentI." % c._name()) 

        

print("All components are ok.")
print("Begin Training:")

for c in components:

    print("\tTraining: %s"%c._name())

    c.train(train_data)

    print("\t\tDone.")

print("Training complete")
print("Begin Validation:")

for c in components:

    res = c.test(val_data)

    

    # compute the score as how many percent are correct

    correct = 0

    for idx, x in enumerate(res):

        if x == val_target[idx]:

            correct += 1

    score = correct / len(res)

    

    validation_scores.append(score)

    print("\t[%f]: %s"%(score, c._name()))

    

print("Validation complete.")

    
for c in components:

    votes.append(c.test(test_data))
if not (voting_mode == "Democracy" or voting_mode == "Aristocracy"):

    raise Exception("Unknown voting mode.","The voting mode must be Democracy or Aristocracy.")



if voting_mode == "Democracy" and len(components)%2 == 0:

    idx = np.argmin(validation_scores)

    votes.pop(idx)

    print("Throwing out %s." % components[idx]._name())

    
result = []



# go through all passengers in test_data

for idx in range(len(test_data)):

    # counting the votes

    dead = 0

    alive = 0

    

    # with weigth one on every vote in Democracy mode

    weight = 1

    

    # go through all components asking for their opinion on this passenger

    for c_idx, vote in enumerate(votes):

        

        if voting_mode == "Aristocracy":

            weight = validation_scores[c_idx]

        

        if vote[idx] == 0:

            dead += weight

        else:

            alive += weight

    

    if dead > alive:

        result.append(0)

    elif alive > dead:

        result.append(1)

    else:

        # if the votes are even (only possible but unlikely in Aristocracy), decide randomly.

        result.append(np.random.choice(1,1))

        

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': result})

output.to_csv('voting_submission.csv', index=False)

print("Your submission was successfully saved!")
