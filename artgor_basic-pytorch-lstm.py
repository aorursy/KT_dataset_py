import numpy as np

import pandas as pd

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

import time

from sklearn.metrics import accuracy_score

import time

import altair as alt

from altair.vega import v3

from IPython.display import HTML

import json

from sklearn.preprocessing import LabelEncoder



import torch.utils.data

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split, GroupKFold, GroupShuffleSplit
# Preparing altair. I use code from this great kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey



vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>",

)))

train = pd.read_csv('../input/X_train.csv')

test = pd.read_csv('../input/X_test.csv')
for col in train.columns:

    if 'orient' in col:

        scaler = StandardScaler()

        train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))

        test[col] = scaler.transform(test[col].values.reshape(-1, 1))
train = np.array([x.values[:,3:].T for group,x in train.groupby('series_id')], dtype='float32')

test = np.array([x.values[:,3:].T for group,x in test.groupby('series_id')], dtype='float32')

train.shape
y = pd.read_csv('../input/y_train.csv')

le = LabelEncoder()

y = le.fit_transform(y['surface'])
class LSTMClassifier(nn.Module):



    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):

        super(LSTMClassifier, self).__init__()

        self.in_dim = in_dim

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.bidirectional = bidirectional

        self.num_dir = 2 if bidirectional else 1

        self.num_layers = num_layers

        self.dropout = dropout



        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout/2, bidirectional=self.bidirectional,

                            batch_first=True)

        self.gru = nn.GRU(self.hidden_dim * 2, self.hidden_dim, bidirectional=self.bidirectional, batch_first=True)





        self.fc = nn.Sequential(

            nn.Linear(4096, int(hidden_dim)),

            nn.SELU(True),

            nn.Dropout(p=dropout),

            nn.Linear(int(hidden_dim), num_classes),

        )



    def forward(self, x):



        x = x.permute(2, 0, 1)

        lstm_out, _ = self.lstm(x)

        gru_out, _ = self.gru(lstm_out)

        avg_pool_l = torch.mean(lstm_out.permute(1, 0, 2), 1)

        max_pool_l, _ = torch.max(lstm_out.permute(1, 0, 2), 1)

        

        avg_pool_g = torch.mean(gru_out.permute(1, 0, 2), 1)

        max_pool_g, _ = torch.max(gru_out.permute(1, 0, 2), 1)



        x = torch.cat((avg_pool_g, max_pool_g, avg_pool_l, max_pool_l), 1)

        y = self.fc(x)

        

        return y
def plot_training_process(df: pd.DataFrame(), epoch_col: str, value_columns: list, y_axis_name: str, title: str):

    

    # code mostly based on: https://altair-viz.github.io/gallery/multiline_tooltip.html

    plot_df = df.melt(id_vars=epoch_col, value_vars=value_columns, var_name='group', value_name=y_axis_name)

    plot_df[y_axis_name] = plot_df[y_axis_name].round(4)

    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=[epoch_col], empty='none')

    line = alt.Chart().mark_line(interpolate='basis').encode(

        x=f'{epoch_col}:Q',

        y=f'{y_axis_name}:Q',

        color='group:N',

    ).properties(

        title=title

    )



    # Transparent selectors across the chart. This is what tells us

    # the x-value of the cursor

    selectors = alt.Chart().mark_point().encode(

        x=f'{epoch_col}:Q',

        opacity=alt.value(0),

    ).add_selection(

        nearest

    )



    # Draw points on the line, and highlight based on selection

    points = line.mark_point().encode(

        opacity=alt.condition(nearest, alt.value(1), alt.value(0))

    )



    # Draw text labels near the points, and highlight based on selection

    text = line.mark_text(align='left', dx=5, dy=-5).encode(

        text=alt.condition(nearest, f'{y_axis_name}:Q', alt.value(' '))

    )



    # Draw a rule at the location of the selection

    rules = alt.Chart().mark_rule(color='gray').encode(

        x=f'{epoch_col}:Q',

    ).transform_filter(

        nearest

    )



    # Put the five layers into a chart and bind the data

    return alt.layer(line, selectors, points, rules, text,

              data=plot_df, width=600, height=300).interactive()
def train_net(train_loader, val_loader, patience, model, criterion, optimizer, scheduler, verbose, plot_training):

    valid_loss_min = np.Inf

    patience = patience

    # current number of epochs, where validation loss didn't increase

    p = 0

    # whether training should be stopped

    stop = False



    epochs = 20000

    training_logs = []



    for e in range(1, epochs + 1):

        # print(time.ctime(), 'Epoch:', e)

        train_loss = []

        train_acc = []

        model.train()

        for batch_i, (data, target) in enumerate(train_loader):

            data, target = data.cuda(), target.cuda()



            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            train_loss.append(loss.item())



            a = target.data.cpu().numpy()

            b = output.detach().cpu().numpy().argmax(1)

            train_acc.append(accuracy_score(a, b))



            loss.backward()

            optimizer.step()



        val_loss = []

        val_acc = []

        for batch_i, (data, target) in enumerate(val_loader):

            data, target = data.cuda(), target.cuda()

            output = model(data)



            loss = criterion(output, target)

            val_loss.append(loss.item()) 

            a = target.data.cpu().numpy()

            b = output.detach().cpu().numpy().argmax(1)

            val_acc.append(accuracy_score(a, b))



        if e % 100 == 0 and verbose:

            print(f'Epoch {e}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}, train acc: {np.mean(train_acc):.4f}, valid acc: {np.mean(val_acc):.4f}')



        training_logs.append([e, np.mean(train_loss), np.mean(val_loss), np.mean(train_acc), np.mean(val_acc)])



        scheduler.step(np.mean(val_loss))



        valid_loss = np.mean(val_loss)

        if valid_loss <= valid_loss_min:

            # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))

            torch.save(model.state_dict(), 'model.pt')

            valid_loss_min = valid_loss

            p = 0



        # check if validation loss didn't improve

        if valid_loss > valid_loss_min:

            p += 1

            # print(f'{p} epochs of increasing val loss')

            if p > patience:

                print('Stopping training')

                stop = True

                break        



        if stop:

            break



    checkpoint = torch.load('model.pt')      

    model.load_state_dict(checkpoint)

    

    if plot_training:

        training_logs = pd.DataFrame(training_logs, columns=['Epoch', 'Train loss', 'Valid loss', 'Train accuracy', 'Validation accuracy'])

        loss_plot = plot_training_process(df=training_logs, epoch_col='Epoch', value_columns=['Train loss', 'Valid loss'], y_axis_name='loss', title='Loss progress')

        acc_plot = plot_training_process(df=training_logs, epoch_col='Epoch', value_columns=['Train accuracy', 'Validation accuracy'], y_axis_name='accuracy', title='Accuracy progress')

        render(loss_plot & acc_plot)

    

    return model
def initialize_model():

    torch.manual_seed(42)

    model = LSTMClassifier(10, 512, 3, 0.1, True, 9, 64)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=.5)

    model.cuda()

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, verbose=True)

    return model, criterion, optimizer, scheduler
def train_net_folds(X, X_test, y, folds, plot_training, batch_size, patience, verbose, le):



    oof = np.zeros((len(X), 9))

    prediction = np.zeros((len(X_test), 9))

    scores = []

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X[train_index], X[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        train_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))

        val_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_valid), torch.LongTensor(y_valid))



        train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)

        val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size)

        

        model, criterion, optimizer, scheduler = initialize_model()

        model = train_net(train_loader, val_loader, patience, model, criterion, optimizer, scheduler, verbose, plot_training)

        

        y_pred_valid = []

        for batch_i, (data, target) in enumerate(val_loader):

            data, target = data.cuda(), target.cuda()

            p = model(data)

            pred = p.cpu().detach().numpy()

            y_pred_valid.extend(pred)

            

        y_pred = []

        for i, data in enumerate(test):

            p = model(torch.FloatTensor(data).unsqueeze(0).cuda())

            y_pred.append(p.cpu().detach().numpy().flatten())

            

        oof[valid_index] = np.array(y_pred_valid)

        scores.append(accuracy_score(y_valid, np.array(y_pred_valid).argmax(1)))



        prediction += y_pred



    prediction /= n_fold

    

    prediction = np.array(prediction).argmax(1)

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    print('--' * 50)

    

    return oof, prediction
n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

oof, prediction = train_net_folds(train, test, y, folds, True, 64, 40, True, le)
sub = pd.DataFrame(le.inverse_transform(prediction), columns=['surface'])

sub.to_csv('submission.csv', index_label='series_id')

sub.head()