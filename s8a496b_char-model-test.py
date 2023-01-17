%load_ext tensorboard.notebook

# %reload_ext tensorboard.notebook

%tensorboard --logdir "/kaggle/working/logs"
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import sys
# reducing dataframe : https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
# df_train = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

df_train = import_data('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
df_train.head()
df_train.comment_text.apply(len).plot.hist(bins = 50)
df_train.comment_text.apply(lambda x : len(x.split())).plot.hist(bins = 50)
# 특문, 영어 대문자, 영어 소문자



char_list = [chr(char) for char in range(32, 127)]

char_list = ["<UNK>", "<PAD>"] + char_list



char2id = {char : idx for idx, char in enumerate(char_list)}
def making_char_vector(input_text, max_char = 64):

    if len(input_text) < max_char:

        return([1] * (max_char - len(input_text)) + [char2id.get(char,0) for char in input_text.lower()[:max_char]])

    else:

        return([char2id.get(char,0) for char in input_text.lower()[:max_char]])



def printing_char_vector(char_vector):

    return("".join([char_list[char] for char in char_vector if char != 1]))
printing_char_vector(making_char_vector(df_train.comment_text[0], 100))
# making dataset

from torch.utils.data import Dataset



class MakingDataset(Dataset):

    def __init__(self, input_dataframe, max_char, is_it_test = False):

        self.is_it_test = is_it_test

        self.text_array = np.array([making_char_vector(text, max_char) for text in input_dataframe.comment_text])

        if not is_it_test:

            self.toxicity_array = np.array([int(toxicity >= 0.5) for toxicity in input_dataframe.target]) 

        

    def __len__(self):

        return len(self.text_array)



    def __getitem__(self, data_index):

        if not self.is_it_test:

            return self.text_array[data_index], self.toxicity_array[data_index]

        else:

            return self.text_array[data_index]

# making model



import torch.nn as nn

import torch.nn.functional as F





class CharNet(nn.Module):

    def __init__(self, max_char, emb_dim, nb_cls):

        super(CharNet, self).__init__()

        self.char_emb = nn.Embedding(len(char2id), emb_dim)

        self.weight_char_emb = nn.Parameter(torch.randn(size = (max_char,)))

        self.fc1 = nn.Linear(emb_dim, nb_cls)





    def forward(self, input_x):

        emb_x = self.char_emb(input_x)

        w_avg_x = torch.einsum('bme, m-> be', [emb_x, self.weight_char_emb])

        score = self.fc1(w_avg_x).squeeze(1)

        return score



batch_size = 256

max_char = 100



train_set = MakingDataset(input_dataframe = df_train.iloc[:int(df_train.shape[0] * 0.8)], max_char = max_char)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)



val_set = MakingDataset(input_dataframe = df_train.iloc[int(df_train.shape[0] * 0.8):], max_char = max_char)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
net = CharNet(max_char = max_char, emb_dim = 200, nb_cls = 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
lossfn = nn.CrossEntropyLoss(weight=torch.tensor([0.08,0.92]).to(device))

optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
max_epoch = 10



tr_index_list =[]

tr_loss_list = []

tr_acc_list= []

val_loss_list= []

val_acc_list = []



for idx_epoch in range(max_epoch):

    net.train()

    running_loss = 0.0

    running_acc = 0.0

    

    for idx_iter, data in enumerate(train_loader):

        text, labels = data

        text = text.to(device)

        labels = labels.to(device)

        

        optimizer.zero_grad()

        

        outputs = net(text)

        loss = lossfn(outputs, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        predicted = torch.max(outputs.data, 1)[1]

        correct = (predicted == labels).sum().item()

        acc = correct / batch_size

            

        running_acc += acc

        

        if idx_iter % 10 == 9:    # print every 2000 mini-batches

            

            sys.stdout.write(

                "\r" + '[%d, %5d] loss: %.5f acc : %.4f' %

                  (idx_epoch + 1, idx_iter + 1, running_loss / 10, running_acc / 10))

            

            tr_index_list.append(idx_iter + 1)

            tr_loss_list.append(running_loss / 10)

            tr_acc_list.append(running_acc / 10)

            

            running_loss = 0.0

            running_acc = 0.0

            

    net.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for idx, data in enumerate(val_loader):

            text, labels = data

            text = text.to(device)

            labels = labels.to(device)

            

            outputs = net(text)

            loss = lossfn(outputs, labels)

            running_loss += loss

            

            predicted = torch.max(outputs.data, 1)[1]

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

            

        running_loss /= (idx + 1)

        acc = correct / total

        

        val_loss_list.append(running_loss)

        val_acc_list.append(acc)

        

        print(f"\n[Val] {idx_epoch+1}-Epoch Loss : {running_loss:.4f}, Acc : {acc:.4f}")

        
# Making df_val without shuffle

df_val = df_train.iloc[int(df_train.shape[0] * 0.8):].reset_index(drop = True)



val_set = MakingDataset(input_dataframe = df_val, max_char = max_char)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)



# Prediction

net.eval()

output_list = []

with torch.no_grad():

    for idx, data in enumerate(val_loader):

        text, labels = data

        text = text.to(device)

        labels = labels.to(device)



        outputs = net(text)

        outputs = nn.Softmax(dim = 1)(outputs)

        output_list = np.append(output_list, outputs[:,1].detach().cpu().numpy().flatten())



# Add prediction to df

MODEL_NAME = 'char_v1'

df_val[MODEL_NAME] = output_list



# Edit df to be evaluated

def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    

def convert_dataframe_to_bool(df):

    bool_df = df.copy()

    for col in ['target'] + identity_columns:

        convert_to_bool(bool_df, col)

    return bool_df



df_val = convert_dataframe_to_bool(df_val)
SUBGROUP_AUC = 'subgroup_auc'

BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative

BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive



def compute_auc(y_true, y_pred):

    try:

        return metrics.roc_auc_score(y_true, y_pred)

    except ValueError:

        return np.nan



def compute_subgroup_auc(df, subgroup, label, model_name):

    subgroup_examples = df[df[subgroup]]

    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])



def compute_bpsn_auc(df, subgroup, label, model_name):

    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""

    subgroup_negative_examples = df[df[subgroup] & ~df[label]]

    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]

    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

    return compute_auc(examples[label], examples[model_name])



def compute_bnsp_auc(df, subgroup, label, model_name):

    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""

    subgroup_positive_examples = df[df[subgroup] & df[label]]

    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]

    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

    return compute_auc(examples[label], examples[model_name])



def compute_bias_metrics_for_model(dataset,

                                   subgroups,

                                   model,

                                   label_col,

                                   include_asegs=False):

    """Computes per-subgroup metrics for all subgroups and one model."""

    records = []

    for subgroup in subgroups:

        record = {

            'subgroup': subgroup,

            'subgroup_size': len(dataset[dataset[subgroup]])

        }

        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)

        print(f'{subgroup}__Done_subgroup_AUC')

        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)

        print(f'{subgroup}__Done_BPSN_AUC')

        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)

        print(f'{subgroup}__Done_BNSP_AUC')

        records.append(record)

    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
from sklearn import metrics

identity_columns = [

    'male', 

    'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

TOXICITY_COLUMN = 'target'

bias_metrics_df = compute_bias_metrics_for_model(df_val, identity_columns, MODEL_NAME, TOXICITY_COLUMN)

bias_metrics_df
def calculate_overall_auc(df, model_name):

    true_labels = df[TOXICITY_COLUMN]

    predicted_labels = df[model_name]

    return metrics.roc_auc_score(true_labels, predicted_labels)



def power_mean(series, p):

    total = sum(np.power(series, p))

    return np.power(total / len(series), 1 / p)



def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):

    bias_score = np.average([

        power_mean(bias_df[SUBGROUP_AUC], POWER),

        power_mean(bias_df[BPSN_AUC], POWER),

        power_mean(bias_df[BNSP_AUC], POWER)

    ])

    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

    

get_final_metric(bias_metrics_df, calculate_overall_auc(df_val, MODEL_NAME))
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
ts_set = MakingDataset(input_dataframe = test, max_char = max_char, is_it_test = True)

ts_loader = torch.utils.data.DataLoader(ts_set, batch_size=batch_size, shuffle=False, num_workers=1)



# Prediction

net.eval()

output_list = np.array([])

with torch.no_grad():

    for _, data in enumerate(ts_loader):

        text = data

        text = text.to(device)



        outputs = net(text)

        outputs = nn.Softmax(dim = 1)(outputs)

        output_list = np.append(output_list, outputs[:,1].detach().cpu().numpy().flatten())
output_list[output_list>0.5].shape[0] / output_list.shape[0]
submission['prediction'] = output_list

submission.to_csv('submission.csv')