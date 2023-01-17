import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch # pytorch for deep learning

import torch.nn as nn # For neural networks

import torch.nn.functional as F # for activation functions like relu

import torch.optim as optim # calling the optimizer that update the weights

from torch.utils.data import DataLoader, Dataset # for load and manage datasets

from sklearn.feature_extraction.text import CountVectorizer # for Bag Of Words represntation

import os # for interact with the operating system 
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
# get the data

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/twitgen_train_201906011956.csv")

train_data.head()
train_data.info()

# 34146 entries and no null values in all columns
train_data = train_data[["text","male"]]

train_data.head(3)

# we got the text and male columns only
train_data["male"] = train_data["male"].astype(int)

train_data.head()
# split the data into train_x and train_y

train_x = train_data["text"]; train_y = train_data["male"]
class Sequences(Dataset):

    def __init__(self, data_x,data_y):

        # Init the bag of words

        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)

        # transform the data to BoW

        self.sequences = self.vectorizer.fit_transform(data_x.to_list())

        self.labels = data_y.to_list()

        self.token2idx = self.vectorizer.vocabulary_

        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        

    def __getitem__(self, i):

        return self.sequences[i, :].toarray(), self.labels[i]

    

    def __len__(self):

        return self.sequences.shape[0]
dataset = Sequences(train_x, train_y)

train_loader = DataLoader(dataset, batch_size=4200)

# print The shape of BoW

print(dataset[4][0].shape)
class BagOfWordsClassifier(nn.Module):

    def __init__(self, vocab_size, hidden1, hidden2, hidden3):

        super(BagOfWordsClassifier, self).__init__()

        self.fc1 = nn.Linear(vocab_size, hidden1)

        self.fc2 = nn.Linear(hidden1, hidden2)

        self.fc3 = nn.Linear(hidden2, hidden3)

        self.fc4 = nn.Linear(hidden3, 1) # 1 because we just have one predict male or not

    

    

    def forward(self, inputs):

        x = F.relu(self.fc1(inputs.squeeze(1).float()))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        return self.fc4(x)
# pass the vocab size and nodes numbers of each hidden layer

model = BagOfWordsClassifier(len(dataset.token2idx), 256,128, 64)

model
# criterion for tells us how wrong the model

criterion = nn.BCEWithLogitsLoss()

# optimizer to update the wieghts and learn 

optimizer = optim.Adam( model.parameters(), lr=0.001) 
# check if the device has GPU or not

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
if train_on_gpu:

    # if the device has GPU run the model on GPU

    model.cuda()

    # start training

model.train()

train_losses = []

for epoch in range(10): # 10 epochs feel free to change 

    losses = []

    total = 0

    for inputs, target in train_loader:

        # clean the old gradient

        model.zero_grad()

        if train_on_gpu: # if the device has GPU pass inputs and targets to it

            inputs, target = inputs.cuda() , target.cuda()

        # forward the data inputs

        output = model(inputs)

        # calculate how wrong the model

        loss = criterion(output.squeeze(), target.float())

        # go back then ....

        loss.backward()

        # update the weights

        optimizer.step()



        # calculate the train loss

        losses.append(loss.item())

        total += 1

    # after finish one epoch sum the loss of each batch and divide on the total

    epoch_loss = sum(losses) / total

    train_losses.append(epoch_loss)



    print(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')
def predict_gender(text):

    # get the model back into cpu

    model.cpu()

    # evaluate the model

    model.eval()

    # we don't need the grad in test so we stop it by no_grad

    with torch.no_grad():

        # convert the test or the new tweet into BoW 

        test_vector = torch.LongTensor(dataset.vectorizer.transform([text]).toarray())

        # now pass the text into the model to predict

        output = model(test_vector)

        # our problem is binary so we'll use sigmoid to get the prediction

        prediction = torch.sigmoid(output).item()



        if prediction > 0.5:

            return "male"

        else:

            return "female"
test_data = pd.read_csv("/kaggle/input/twitgen_test_201906011956.csv")

test_data.head()
# convert True values to male and False values to True 

test_data["male"] [test_data["male"] == True] = "male"

test_data["male"] [test_data["male"] == False] = "female"
test_x = test_data["text"]

test_y = test_data["male"]
def predict_from_test(text, label):

    if predict_gender(text) == label:

        print(predict_gender(text) , " Correct predict")

    else:

        print(predict_gender(text) , " Wrong predict")
# time to test

from random import randint

num = randint(0,6000)

print(test_x[num])

predict_from_test(test_x[num],test_y[num])
t2 = "How u get such a discount on espresso shots????"

predict_gender(t2)
t3 = "I had the same thing in my trends.For a moment I thought it is a new feature where I can post to other's “Twitter moments”"

predict_gender(t3)