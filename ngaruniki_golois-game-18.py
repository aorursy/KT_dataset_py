#data preprosessing

import numpy as np

import torch

from torchvision import datasets

from torch.utils.data import Dataset

from PIL import Image



def get_dataset(name):

    return get_game_data()



def to_one_hot(labels_np):

    labels_np = labels_np.squeeze()

    uniq = np.sort(np.unique(labels_np)).tolist()

    result = np.zeros((len(labels_np), len(uniq)))



    #for index, label in enumerate(uniq):

        #result[labels_np == label, index] = 1

    return result



def load_X():



    # loaded_data = np.genfromtxt('test_data.csv', delimiter=' ')

    loaded_data = np.load('../input/goloisdata/DeepLearningProject/input_data.npy')



    print("X values LOADED")

    splited_data = np.split(loaded_data, 100000)



    reshaped = []

    for sd in splited_data:

        reshaped.append(np.reshape(sd, (1, 8, 19, 19)))



    X = np.concatenate(reshaped)



    print("X values reshaped and ready")

    return X



def load_Y():

    value = np.load('../input/goloisdata/DeepLearningProject/value.npy')

    

    policy = np.load('../input/goloisdata/DeepLearningProject/policy.npy')

    Y = np.column_stack((policy, value))

    #Y = value

    print("y values LOADED and READY")

    return Y



def get_game_data():

    X = load_X()

    y = load_Y()

    y = to_one_hot(y).astype(np.long)



    # train-test: 0.8/0.2, replace with your index selection

    indices = np.arange(len(X))

    np.random.seed(42)

    test_indices = np.random.choice(indices, size=int(len(X) * 0.2))

    train_mask = np.ones(len(X), dtype=np.bool)

    train_mask[test_indices] = 0

    test_mask = np.zeros(len(X), dtype=np.bool)

    test_mask[test_indices] = 1



    X_train = torch.from_numpy(X[train_mask])

    y_train = torch.from_numpy(y[train_mask])



    X_test = torch.from_numpy(X[test_mask])

    y_test = torch.from_numpy(y[test_mask])



    return X_train, y_train, X_test, y_test



def get_handler(name):

    if name == 'game_data':

        return DataHandler



class DataHandler(Dataset):

    def __init__(self, X, Y, transform=None):

        self.X = X

        self.Y = Y

        self.transform = transform



    def __getitem__(self, index):

        x, y = self.X[index], self.Y[index]

        if self.transform is not None:

            x = Image.fromarray(x.numpy())

            x = self.transform(x)

        return x, y, index



    def __len__(self):

        return len(self.X)
#network Design

import torch.nn as nn

import torch.nn.functional as F



def get_net(name):

    return Network

    

class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(8, 2, kernel_size=5, padding=2) # initially out X batch has shape [batch_size, 1, 9, 9]

        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        self.conv3_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(64, 400)

        self.fc2 = nn.Linear(400, 50)

        self.fc3 = nn.Linear(50, 2)



    def forward(self, x):

        x = F.relu(self.conv1(x.float()))

        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))

        x = x.view(-1, 64)

        x = F.relu(self.fc1(x))

        e1 = F.relu(self.fc2(x))

        x = F.dropout(e1, training=self.training)

        x = self.fc3(x)

        return x, e1



    def get_embedding_dim(self):

        return 50
import numpy as np

import torch

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

import sys



def erase_line():

    CURSOR_UP_ONE = '\x1b[1A'

    ERASE_LINE = '\x1b[2K'

    sys.stdout.write(CURSOR_UP_ONE)

    sys.stdout.write(ERASE_LINE)



def accuracy_quick(predictions, target):

    # calling code set mode = 'train' or 'eval'

    Y = torch.max(target, 1)[1]

    oupt = predictions

    (max_vals, arg_maxs) = torch.max(oupt.data, dim=1) 

    # arg_maxs is tensor of indices [0, 1, 0, 2, 1, 1 . . ]

    num_correct = torch.sum(Y==arg_maxs)

    acc = (1.0* num_correct * 100.0 / len(Y))

    return acc.item()  # percentage based



class Strategy:

    def __init__(self, X, Y, idxs_lb, net, handler, args):

        self.X = X

        self.Y = Y

        self.idxs_lb = idxs_lb

        self.net = net

        self.handler = handler

        self.args = args

        self.n_pool = len(Y)

        use_cuda = torch.cuda.is_available()

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.train_losses = []

        self.val_losses = []

        self.train_acc = []

        self.val_acc = []

        self.valX = None

        self.valY = None

    def query(self, n):

        pass

    #test data set

    def set_test_data(self, x, y):

        self.valX = x 

        self.valY = y

    def update(self, idxs_lb):

        self.idxs_lb = idxs_lb



    def _train(self, epoch, loader_tr, optimizer):

        self.clf.train()

        total_number = len(loader_tr)

        update_rate = 5

    

        # Initiate accuracies and losses per epoch

        final_train_accuracy = 0

        final_train_loss = 0

        final_val_loss = 0

        final_val_acc = 0



        for batch_idx, (x, y, idxs) in enumerate(loader_tr):

            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()

            out, e1 = self.clf(x)

            # print("out shape: "+ repr(out.shape))



            # print("target shape: " + repr((torch.max(y, 1)[1]).shape))

            loss = F.cross_entropy(out, torch.max(y, 1)[1])

            loss.backward()

            optimizer.step()

            if batch_idx % update_rate==0:

                if(batch_idx > 0):

                    erase_line()

                progress = batch_idx / total_number * 100

                acc = accuracy_quick(out, y)

                final_train_accuracy = acc

                final_train_loss = loss

                print('Training\t Progress:\t %f %%\tLoss: %f\t Training Accuracy %0.2f %%' %(progress, loss, acc))



        #Validation Load the test data and compute loss and accuracy

        loader_te = DataLoader(self.handler(self.valX, self.valY, transform=self.args['transform']),

                            shuffle=False, **self.args['loader_te_args'])



        self.clf.eval()

        loss = 0

        n = 0

        predictions = torch.zeros(len(self.valY), dtype=self.valY.dtype)

        with torch.no_grad():

            for x, y, idxs in loader_te:

                x, y = x.to(self.device), y.to(self.device)

                out, e1 = self.clf(x)

                loss += F.cross_entropy(out, torch.max(y, 1)[1])

                n += 1

                pred = out.max(1)[1]



                predictions[idxs] = pred.cpu().type(torch.LongTensor)



        final_val_acc = 100.0 * (torch.max(self.valY, 1)[1]==predictions).sum().item() / len(self.valY)

        final_val_loss =  1.0 * loss/n

        print('\nValidation\n=========\nProgress:\t 100 %%\nValidation Loss: %f\nValidation Accuracy %0.2f %%\n' %( final_val_loss, final_val_acc))

        return final_train_loss, final_val_loss, final_train_accuracy, final_val_acc

             

    def train(self):



        n_epoch = self.args['n_epoch']

        self.clf = self.net().to(self.device)

        optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])



        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),

                            shuffle=True, **self.args['loader_tr_args'])

        

        for epoch in range(1, n_epoch+1):

            print("="*100 + '\n')

            print('Epoch %d of %d' %(epoch, n_epoch))

            train_loss, val_loss, train_acc, val_acc = self._train(epoch, loader_tr, optimizer)



    def predict(self, X, Y):

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),

                            shuffle=False, **self.args['loader_te_args'])



        self.clf.eval()

        P = torch.zeros(len(Y), dtype=Y.dtype)

        with torch.no_grad():

            for x, y, idxs in loader_te:

                x, y = x.to(self.device), y.to(self.device)

                out, e1 = self.clf(x)



                pred = out.max(1)[1]

                P[idxs] = pred.cpu().type(torch.LongTensor)



        return P



    def predict_prob(self, X, Y):

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),

                            shuffle=False, **self.args['loader_te_args'])



        self.clf.eval()

        probs = torch.zeros([len(Y), len(np.unique(Y))])

        with torch.no_grad():

            for x, y, idxs in loader_te:

                x, y = x.to(self.device), y.to(self.device)

                out, e1 = self.clf(x)

                prob = F.softmax(out, dim=1)

                probs[idxs] = prob.cpu()

        

        return probs
import numpy as np

#from dataset import get_dataset, get_handler

#from model import get_net

from torchvision import transforms

import torch

#from train import Strategy



class RandomSampling(Strategy):

    def __init__(self, X, Y, idxs_lb, net, handler, args):

        super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):

        return np.random.choice(np.where(self.idxs_lb==0)[0], n)



print("Running...")

# parameters

NUM_INIT_LB = 800

NUM_QUERY = 1000

NUM_ROUND = 0

DATA_NAME = 'game_data'



args_pool = {'game_data':

                {'n_epoch': 5, 'transform': None, # transforms.Compose([transforms.ToTensor()])

                 'loader_tr_args':{'batch_size': 64},

                 'loader_te_args':{'batch_size': 64},

                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},

           }

args = args_pool[DATA_NAME]



# load dataset

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)

X_tr = X_tr[:40000]

Y_tr = Y_tr[:40000]



# start experiment

n_pool = len(Y_tr)

n_test = len(Y_te)

print('number of labeled pool: {}'.format(NUM_INIT_LB))

print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))

print('number of testing pool: {}'.format(n_test))



# generate initial labeled pool

idxs_lb = np.zeros(n_pool, dtype=bool)

idxs_tmp = np.arange(n_pool)

np.random.shuffle(idxs_tmp)

idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True



# load network

net = get_net(DATA_NAME)

handler = get_handler(DATA_NAME)

strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)



# print info

# round 0 accuracy

strategy.set_test_data(x=X_te, y=Y_te)

strategy.train()

P = strategy.predict(X_te, Y_te)

acc = np.zeros(NUM_ROUND+1)

acc[0] = 1.0 * (torch.max(Y_te, 1)[1]==P).sum().item() / len(Y_te)

print('Round 0\ntesting accuracy {}'.format(acc[0]))

exit()

for rd in range(1, NUM_ROUND+1):

    print('='*100+'\n' + '='*100) 

    print('Round {}'.format(rd))



    # query

    q_idxs = strategy.query(NUM_QUERY)

    idxs_lb[q_idxs] = True



    # update

    strategy.update(idxs_lb)

    strategy.train()

     

    # round accuracy

    P = strategy.predict(X_te, Y_te)

    acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)

    print('testing accuracy {}'.format(acc[rd]))


