import json

import torch

from torch import nn

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import torch.nn.functional as F

import catalyst.dl as dl

import catalyst.dl.utils as utils

import numpy as np, pandas as pd
def one_hot(categories, string):

    encoding = np.zeros((len(string), len(categories)))

    for idx, char in enumerate(string):

        encoding[idx, categories.index(char)] = 1

    return encoding



def featurize(entity):

    sequence = one_hot(list('ACGU'), entity['sequence'])

    structure = one_hot(list('.()'), entity['structure'])

    loop_type = one_hot(list('BEHIMSX'), entity['predicted_loop_type'])

    features = np.hstack([sequence, structure, loop_type])

    return features 



def char_encode(index, features, feature_size):

    half_size = (feature_size - 1) // 2

    

    if index - half_size < 0:

        char_features = features[:index+half_size+1]

        padding = np.zeros((int(half_size - index), char_features.shape[1]))

        char_features = np.vstack([padding, char_features])

    elif index + half_size + 1 > len(features):

        char_features = features[index-half_size:]

        padding = np.zeros((int(half_size - (len(features) - index))+1, char_features.shape[1]))

        char_features = np.vstack([char_features, padding])

    else:

        char_features = features[index-half_size:index+half_size+1]

    

    return char_features
def augment(X: np.array):

    

    X = np.vstack((X, np.flip(X, axis=1)))

    

    return X



class VaxDataset(Dataset):

    def __init__(self, path, test=False):

        self.path = path

        self.test = test

        self.features = []

        self.targets = []

        self.ids = []

        self.load_data()

    

    def load_data(self):

        with open(self.path, 'r') as text:

            for line in text:

                records = json.loads(line)

                features = featurize(records)

                

                for char_i in range(records['seq_scored']):

                    char_features = char_encode(char_i, features, 21)

                    self.features.append(augment(char_features))

                    self.ids.append('%s_%d' % (records['id'], char_i))

                        

                if not self.test:

                    

                    targets = np.stack([records['reactivity'], records['deg_Mg_pH10'], records['deg_Mg_50C']], axis=1)

                    self.targets.extend([targets[char_i] for char_i in range(records['seq_scored'])])

                    

    def __len__(self):

        return len(self.features)

    

    def targets(self):

        return self.targets

    

    def __getitem__(self, index):

        if self.test:

            return self.features[index], self.ids[index]

        else:

            return self.features[index], self.targets[index], self.ids[index]
class Flatten(nn.Module):

    def forward(self, x):

        batch_size = x.shape[0]

        return x.view(batch_size, -1)

 

class WaveBlock(nn.Module):



    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):

        super(WaveBlock, self).__init__()

        self.num_rates = dilation_rates

        self.convs = nn.ModuleList()

        self.filter_convs = nn.ModuleList()

        self.gate_convs = nn.ModuleList()



        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))

        dilation_rates = [2 ** i for i in range(dilation_rates)]

        for dilation_rate in dilation_rates:

            self.filter_convs.append(

                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))

            self.gate_convs.append(

                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))

            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))



    def forward(self, x):

        x = self.convs[0](x)

        res = x

        for i in range(self.num_rates):

            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))

            x = self.convs[i + 1](x)

            res = res + x

        return res





"""Modified version of the SED work by Hidehisa Arai."""

def init_layer(layer):

    nn.init.xavier_uniform_(layer.weight)



    if hasattr(layer, "bias"):

        if layer.bias is not None:

            layer.bias.data.fill_(0.)

            



def init_bn(bn):

    bn.bias.data.fill_(0.)

    bn.weight.data.fill_(1.0)

    

class AttBlock(nn.Module):

    def __init__(self,

                 in_features: int,

                 out_features: int,

                 activation="linear",

                 temperature=1.0):

        super().__init__()



        self.activation = activation

        self.temperature = temperature

        self.att = nn.Conv1d(

            in_channels=in_features,

            out_channels=out_features,

            kernel_size=1,

            stride=1,

            padding=0,

            bias=True)

        self.cla = nn.Conv1d(

            in_channels=in_features,

            out_channels=out_features,

            kernel_size=1,

            stride=1,

            padding=0,

            bias=True)



        self.bn_att = nn.BatchNorm1d(out_features)

        self.init_weights()



    def init_weights(self):

        init_layer(self.att)

        init_layer(self.cla)

        init_bn(self.bn_att)



    def forward(self, x):

        # x: (n_samples, n_in, n_time)

        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)

        cla = self.nonlinear_transform(self.cla(x))

        x = torch.sum(norm_att * cla, dim=2)

        return x



    def nonlinear_transform(self, x):

        if self.activation == 'linear':

            return x

        elif self.activation == 'sigmoid':

            return torch.sigmoid(x)



class VaxModel(nn.Module):

    def __init__(self):

        super(VaxModel, self).__init__()

        self.layers = nn.Sequential(

            nn.Dropout(0.2),

            nn.Conv1d(14, 32, 1, 1),

            WaveBlock(32, 64, 1, 1),

            nn.PReLU(),

            nn.BatchNorm1d(64),

            nn.Upsample(scale_factor=2, mode='linear'),

            nn.Dropout(0.2),

            nn.Conv1d(64, 1, 1, 1),

        )

        self.rnn1 = nn.LSTM(84, 64)

      

        

        self.finalprelu = nn.PReLU()

        self.finaldrop = nn.Dropout(0.2),

        self.attn =  AttBlock(16,32),

        self.final = nn.Sequential(

        nn.PReLU(),

        nn.Dropout(0.2),

        nn.Linear(32, 3)

        )

    

    def forward(self, features):

        

        features = self.layers(features)

        features = features.permute(1, 0, 2)

        features = self.rnn1(features)

        features = self.finalprelu(features[0])

        if features.size() == torch.Size([1, 16, 64]):

            self.attn =  AttBlock(16,32).cuda().float()

            features = self.attn(features)

        else:

            self.attn = AttBlock(3,32).cuda().float()

            features = self.attn(features)

        final = self.final(features)

        return final
model = VaxModel().cuda()

optimizer = torch.optim.SGD(model.parameters(), 0.005, momentum=0.9)

criterion = nn.MSELoss()
train_dataset = VaxDataset('../input/stanford-covid-vaccine/train.json')

train_dataloader = DataLoader(train_dataset, 16, shuffle=True, num_workers=4, pin_memory=True)
class CustomRunner(dl.Runner):



    def predict_batch(self, batch):

        # model inference step

        return self.model(batch[0].to(self.device).permute(0, 2, 1).float()), batch[1]



    def _handle_batch(self, batch):

        # model train/valid step

        x, y = batch[0], batch[1]

        x = x.cuda().permute(0,2,1).float()

        y = y.cuda().float().unsqueeze(0)[:, 0, :]

        y_hat = self.model(x)



        loss = criterion(y_hat, y)

        score = mcrmse_loss(y_hat, y)

        self.batch_metrics.update(

            {"loss": loss, 'metric': score}

        )



        if self.is_train_loader:

            loss.backward()

            self.optimizer.step()

            self.optimizer.zero_grad()

device = utils.get_device()
def mcrmse_loss(y_true, y_pred, N=3):

    """

    Calculates competition eval metric

    """

    y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()

    assert len(y_true) == len(y_pred)

    n = len(y_true)

    return np.sum(np.sqrt(np.sum((y_true - y_pred)**2, axis=0)/n)) / N
test_dataset = VaxDataset('../input/stanford-covid-vaccine/test.json', test=True)

test_dataloader = DataLoader(test_dataset, 16, num_workers=4, drop_last=False, pin_memory=True)

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-3,max_lr=1e-2,step_size_up=2000)



loaders = {

    'train': train_dataloader,

}

runner = CustomRunner(device=device)

# model training

runner.train(

    model=model,

    optimizer=optimizer,

    loaders=loaders,

    logdir="../working",

    num_epochs=6,

    scheduler=scheduler,

    verbose=False,

    load_best_on_end=True,

    

)
utils.plot_metrics(

    logdir="../working", 

    # specify which metrics we want to plot

    metrics=["loss", "metric"]

)
sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv', index_col='id_seqpos')



for predictions, ids in runner.predict_loader(loader=test_dataloader):

    sub.loc[ids, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = predictions.detach().cpu().numpy()
sub.head()
sub.to_csv('submission.csv')