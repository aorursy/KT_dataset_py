%matplotlib inline
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = '../input/avito-demand-prediction/'
textdata_path = '../input/adp-prepare-kfold-text/textdata.csv'
EMB_PATH = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
#EMB_PATH = '../input/fasttext-russian-2m/wiki.ru.vec'
target_col = 'deal_probability'
os.listdir(DATA_DIR)
dtype = {
    #'context': 'object',    
    'text': 'object',
    'eval_set': 'int8',
    'label': 'float64'
}
usecols = ['text', 'eval_set', 'label']
df = pd.read_csv(textdata_path, usecols=usecols, dtype=dtype)
df.head()
max_features = 30000
maxlen = 100
embed_size = 300
%%time
import keras
from keras.preprocessing import text, sequence
print('tokenizing...')
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['text'].values.tolist())#+df['context'].values.tolist()
%%time
def get_coefs(word, *arr, tokenizer=None):
    if tokenizer is None:
        return word, np.asarray(arr, dtype='float32')
    else:
        if word not in tokenizer.word_index:
            return None
        else:
            return word, np.asarray(arr, dtype='float32')
nb_words = min(max_features, len(tokenizer.word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for o in tqdm(open(EMB_PATH), desc='getting embeddings'):
    res = get_coefs(*o.rstrip().rsplit(' '), tokenizer=tokenizer)
    if res is not None:
        idx = tokenizer.word_index[res[0]]
        if idx < max_features:
            embedding_matrix[idx] = res[1]
gc.collect()
%%time
def fill_rand_norm(embedding_matrix):
    mask = embedding_matrix.sum(axis=1)==0
    zero_ratio = (mask).sum() / embedding_matrix.shape[0]
    print('zero ratio:', zero_ratio)
    emb_zero_shape = ((mask).sum(), embedding_matrix.shape[1])
    emb_non_zero_mean = embedding_matrix[~mask].mean()
    emb_non_zero_std = embedding_matrix[~mask].std()
    embedding_matrix[mask] = np.random.normal(emb_non_zero_mean, 
                                              emb_non_zero_std, 
                                              emb_zero_shape)
    return embedding_matrix
embedding_matrix = fill_rand_norm(embedding_matrix)
%%time
text = df['text'].values
eval_sets = df['eval_set'].values
labels = df['label'].values
train_num = (df['label']<2).sum()
del df; gc.collect()
text = tokenizer.texts_to_sequences(text)
text = sequence.pad_sequences(text, maxlen=maxlen)
del tokenizer; gc.collect()

#print((text!=0).sum(axis=0)/text.shape[0])
plt.figure()
plt.plot((text!=0).sum(axis=0)/text.shape[0])
plt.title('Average Non-zero Element Ratio Along Sequences')
plt.axis([0., maxlen-1, 0., 1.])
plt.show()
from time import time
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class DataBuilder(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.size = len(X)
    def __getitem__(self, index):
        X_i = torch.LongTensor(self.X[index].tolist())
        if self.y is not None:
            y_i = torch.FloatTensor([float(self.y[index])])
        else:
            y_i = torch.FloatTensor([-1])
        return X_i, y_i
    def __len__(self):
        return self.size
    
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def init_args(self):
        raise NotImplementedError('subclass must implement this')
    
    def check_args(self):
        default_values = {
            'optimizer_type': 'adam',
            'save_sub_dir': '.', 
            #'valid_scores_topk': [-1, -1, -1],
            'log_interval': 50,
            'valid_interval': 200,
            'patience': 3,
            'valid_interval_re_decay': 0.7,
            'valid_interval_min': 50,
            'lr_re_decay': 0.5,
            'batch_size': 32,
            'lr': 0.001,
            'weight_decay': 0.0,
            'n_epochs': 2,
        }
        args = self.args
        if 'greater_is_better' not in args.__dict__ or \
            args.greater_is_better not in [True, False]:
            raise NotImplementedError('args.greater_is_better must be in [True, False]')
        if args.greater_is_better:
            default_values['valid_scores_topk'] = [-999]*default_values['patience']
        else:
            default_values['valid_scores_topk'] = [999]*default_values['patience']
        for k, v in default_values.items():
            if k not in args.__dict__:
                args.__dict__[k] = v
                print('Fill in arg %s with default value'%(k), v)
    
    def forward(self, x):
        raise NotImplementedError('subclass must implement this')
    
    def get_optimizer_caller(self, optimizer_type):
        choice_d = {'sgd' : torch.optim.SGD,
                    'adam': torch.optim.Adam,
                    'rmsp': torch.optim.RMSprop,
                    'adag': torch.optim.Adagrad}
        assert optimizer_type in choice_d
        return choice_d[optimizer_type]
    
    def logit2label(self, logits, onehot=False):
        logits_ = np.array(logits)
        if onehot:
            return (logits_/logits_.max()).astype(np.int8).astype(np.float32)
        else:
            return np.argmax(logits_, axis=len(logits_.shape)-1)
            
    def save(self, path):
        torch.save(self.state_dict(), path)
        print('model saved at', path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        print('model loaded from', path)
    
    def save_args_dict(self, args, path):
        with open(path, 'wb') as f:
            pickle.dump(args.__dict__, f)
        print('args_dict saved at', path)
                    
    def load_args_dict(self, path):
        with open(path, 'rb') as f:
            args_dict = pickle.load(f)
        print('args_dict loaded from', path)
        print('returned')
        return args_dict

    def save_finished_args(self, args):
        args.finished = True
        args_fin_path = os.path.join(args.save_sub_dir, 'args.pkl')
        self.save_args_dict(args, args_fin_path)
        print('Finished! Topk:', args.valid_scores_topk)
        return args  

    def fit_batch(self, 
                  X_batch, y_batch, weight=None):
        model = self.train()
        x = Variable(X_batch)
        y = Variable(y_batch).view(-1)
        if self.use_cuda:
            x, y = x.cuda(), y.cuda()
        self.optimizer.zero_grad()
        outputs = model(x)
        pred = outputs
        if self.use_cuda:
            pred = pred.cpu()
        loss = self.criterion(outputs, y, weight=weight)
        loss.backward()
        self.optimizer.step()
        return loss.data[0], pred.data.numpy()
    
    def valid_batch(self, 
                    X_batch, y_batch, weight=None):
        model = self.eval()
        x = Variable(X_batch, volatile=True)
        y = Variable(y_batch).view(-1)
        if self.use_cuda:
            x, y = x.cuda(), y.cuda()
        outputs = model(x)
        pred = outputs
        if self.use_cuda:
            pred = pred.cpu()
        loss = self.criterion(outputs, y, weight=weight)
        return loss.data[0], pred.data.numpy()

    def predict_batch(self,
                      X_batch):
        model = self.eval()
        x = Variable(X_batch, volatile=True)
        if self.use_cuda:
            x = x.cuda()
        outputs = model(x)
        pred = outputs
        if self.use_cuda:
            pred = pred.cpu()
        return pred.data.numpy()

    def predict(self, 
                X_test=None, use_topk=-1, reduce=True, reduce_mode='weighted'):
        assert X_test is not None or self.test_generator is not None, \
        "Either 'X_test' or 'self.test_generator' need to be provided"
        if X_test is not None:
            test_dataset = DataBuilder(X_test)
            self.test_generator = DataLoader(dataset=test_dataset,
                                             batch_size=self.batch_size)
            print("'self.test_generator' is updated by 'X_test'")
        return self.predict_generator(self.test_generator, 
                                      use_topk=use_topk, 
                                      reduce=reduce, 
                                      reduce_mode=reduce_mode)
    
    def predict_generator(self, 
                          test_generator, 
                          use_topk=-1,
                          reduce=True,
                          reduce_mode='weighted'):
        args = self.args
        model = self.eval()
        print('predict with checkpoint at', args.save_sub_dir)
        pred_all = []
        cnt = 0
        n_pred = len(args.valid_scores_topk)
        if use_topk==-1:
            use_topk = n_pred
        for top_idx in range(use_topk):
            cnt += 1
            pred = []
            model_path = os.path.join(args.save_sub_dir, str(top_idx)+'.pth')
            if os.path.exists(model_path):
                model.load(model_path)
            else:
                continue
            model.eval()
            for bx, _ in test_generator:
                p = model.predict_batch(bx)
                pred.extend(p)
            pred_all.append(pred)
            if cnt==use_topk:
                break
        if not reduce:
            pred_res = pred_all
        elif reduce and reduce_mode=='mean':
            pred_res = np.mean(pred_all, axis=0)
        elif reduce and reduce_mode=='weighted':
            weights = np.array(args.valid_scores_topk[:len(pred_all)])
            weights = np.exp(-weights)/np.exp(-weights).sum()
            pred_res = np.sum([np.array(pred_all[i])*weights[i] for i in range(len(pred_all))], axis=0)
        print('prediction done!')
        return pred_res

    def fit(self, 
            X_train=None, y_train=None, 
            X_valid=None, y_valid=None):
        TRAIN_NULL_FLAG = (X_train is None) or (y_train is None)
        VALID_NULL_FLAG = (X_valid is None) or (y_valid is None)
        assert TRAIN_NULL_FLAG==False or self.train_generator is not None, \
        "Either 'X/y_train' or 'self.train_generator' need to be provided"
        
        args = self.args

        if args.save_sub_dir and not os.path.exists(args.save_sub_dir):
            print("Save path is not existed!")
            print('Creating dir at', args.save_sub_dir)
            os.makedirs(args.save_sub_dir, exist_ok=True)
        
        if not TRAIN_NULL_FLAG:
            train_dataset = DataBuilder(X_train, y_train)
            self.train_generator = DataLoader(dataset=train_dataset,
                                              batch_size=self.batch_size)
            print("'self.train_generator' is updated by 'X/y_train'")
            print('Train with {} samples'.format(len(y_train)))
            
        if not VALID_NULL_FLAG:
            valid_dataset = DataBuilder(X_valid, y_valid)
            self.valid_generator = DataLoader(dataset=valid_dataset,
                                              batch_size=self.batch_size)
            print("'self.valid_generator' is updated by 'X/y_valid'")
            print('Validate with {} samples'.format(len(y_valid)))
        
        args = self.fit_generator(args, 
                                  self.train_generator,
                                  self.valid_generator)
    
    def fit_generator(self, 
                      args, train_generator, valid_generator=None):
        args.n_iter = 0
        args.restarted = 0
        args.finished = False
        args.valid_scores = []
        args.train_begin_time = time()
        
        self.optimizer = self.optimizer_caller(self.parameters(), 
                                               lr=args.lr,
                                               weight_decay=args.weight_decay)
        
        total_loss = 0.0
        for epoch in range(args.n_epochs):
            if args.finished:
                break
            batch_begin_time = time()
            ma_loss = 0.0
            running_pred_train = []
            running_y_train = []
            for batch_idx, (bx, by) in enumerate(train_generator):
                if args.finished:
                    break
                args.n_iter += 1
                loss_tr, pred_tr = self.fit_batch(bx, by)
                total_loss += loss_tr
                ma_loss += loss_tr
                running_pred_train.extend(pred_tr)
                running_y_train.extend(by)
                if args.n_iter % args.log_interval == 0:
                    score = self.eval_metric(running_y_train, 
                                             running_pred_train)
                    print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' % \
                          (epoch + 1, 
                           args.n_iter, 
                           ma_loss/args.log_interval, 
                           score, 
                           time()-batch_begin_time))
                    ma_loss = 0.0
                    running_pred_train = []
                    running_y_train = []
                    batch_begin_time = time()
                if valid_generator is not None and \
                args.n_iter % args.valid_interval == 0:
                    args = self.evaluate_generator(args, valid_generator)
                    args = self.check_early_stopping(args)
                    print('valid time: %.1f s' % (time()-batch_begin_time))
                    batch_begin_time = time()
        return args
    
    def valid(self, 
              X_valid=None, y_valid=None):
        VALID_NULL_FLAG = (X_valid is None) or (y_valid is None)
        assert VALID_NULL_FLAG==False or self.valid_generator is not None, \
        "Either 'X/y_valid' or 'self.valid_generator' need to be provided"
        
        args = self.args
        
        if not VALID_NULL_FLAG:
            valid_dataset = DataBuilder(X_valid, y_valid)
            self.valid_generator = DataLoader(dataset=valid_dataset,
                                              batch_size=self.batch_size)
            print("'self.valid_generator' is updated by 'X/y_valid'")
            print('Validate with {} samples'.format(len(y_valid)))
        begin_time = time()
        args = self.evaluate_generator(args, self.valid_generator)
        print('valid time: %.1f s' % (time()-begin_time))
    
    def evaluate_generator(self, 
                           args, valid_generator):
        running_pred_valid = []
        running_y_valid = []
        val_total_loss = 0.0
        for _bx,_y in valid_generator:
            loss_val, pred_val = self.valid_batch(_bx,_y)
            running_pred_valid.extend(pred_val)
            running_y_valid.extend(_y)
            val_total_loss += loss_val*len(_y)
        _score = self.eval_metric(running_y_valid, running_pred_valid)
        args.valid_scores.append(_score)
        print('*'*50)
        print('valid loss: %.6f metric: %.6f total time: %.1f s' %
              (val_total_loss/len(running_y_valid), 
               _score, 
               time()-args.train_begin_time))
        print('*'*50)
        return args
    
    def check_early_stopping(self, args):
        _score = args.valid_scores[-1]
        early_stopping_flag = True
        for top_idx, top_scr in enumerate(args.valid_scores_topk):
            if (_score - top_scr > 0) == args.greater_is_better:
                args.valid_scores_topk[top_idx] = _score
                print('Best %d-th valid score:' % top_idx, _score)
                save_topkth_path = os.path.join(args.save_sub_dir, 
                                                str(top_idx)+'.pth')
                self.save(save_topkth_path)
                early_stopping_flag = False
                break
        if early_stopping_flag:
            if args.restarted < args.patience:
                save_top0th_path = os.path.join(args.save_sub_dir, str(0)+'.pth')
                print()
                print('\t\tEarly stopped, restarting from', save_top0th_path)
                print()
                self.load(save_top0th_path)
                args.restarted += 1
                args.valid_interval = max(int(args.valid_interval * \
                                              args.valid_interval_re_decay), 
                    args.valid_interval_min)
                args.lr = args.lr * args.lr_re_decay
                self.optimizer = self.optimizer_caller(self.parameters(), 
                                                       lr=args.lr, 
                                                       weight_decay=args.weight_decay)
            else:
                args = self.save_finished_args(args)
        return args
from sklearn.metrics import mean_squared_error

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

## https://github.com/zachAlbus/pyTorch-text-classification/blob/master/Yoon/model.py
class TextCNN(BaseModel):
    def __init__(self):
        super(TextCNN, self).__init__()
    def _eval_metric(self, labels, preds):
        return np.sqrt(mean_squared_error(labels, preds))
    def _criterion(self, input, target, weight=None):
        if weight is None:
            return torch.sqrt(F.mse_loss(input, target, size_average=True))
        else:
            return torch.sum(weight * (input - target) ** 2)
    
    def init_args(self, args, n_output,
                  maxlen, max_features, embed_size, embedding_init, max_pooling_k,
                  in_channel, out_channel, kernel_sizes, dilations,
                  dropout, n_final_state):
        if not torch.cuda.is_available():
            args.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")
        else:
            args.use_cuda = True
        
        torch.manual_seed(233)
        
        self.use_cuda = args.use_cuda
        self.optimizer_caller = self.get_optimizer_caller(args.optimizer_type)
        
        self.criterion = self._criterion
        self.eval_metric = self._eval_metric
        self.batch_size = args.batch_size
        self.args = args
        self.check_args()
        print('args initialized')

        self.n_output = n_output
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.maxlen = maxlen
        self.max_features = max_features
        self.embed_size = embed_size
        self.max_pooling_k = max_pooling_k
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.n_final_state = n_final_state
        self.dropout = dropout
        
        self.return_final_state = False
        
        self.embed = nn.Embedding(max_features, embed_size)
        if embedding_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embedding_init))
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channel, out_channel, 
                      kernel_size=(K, embed_size), dilation=(D, 1)) \
            for K, D in zip(kernel_sizes, dilations)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_state = nn.Linear(len(kernel_sizes) * out_channel * max_pooling_k, n_final_state)
        self.fc = nn.Linear(n_final_state, n_output)
        
        if self.use_cuda:
            return self.cuda()
        else:
            return self
    def forward(self, x):
        batch_size = x.size(0)
        # Embedding
        x = self.embed(x)  # dim: (batch_size, max_len, embed_size)
        # Conv & max pool
        x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)
        # turns to be a list of ele with dim: ([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv] # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        #x = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in x]
        x = [kmax_pooling(conv_out, 2, self.max_pooling_k).view(batch_size, -1) for conv_out in x]
        x = torch.cat(x, 1)
        # Dropout & output
        x = self.dropout(x)  #dim: (batch_size, len(kernel_sizes)*num_kernels)
        x = self.final_state(x)
        if self.return_final_state:
            return x
        else:
            x = F.relu(x)
            x = self.fc(x)
            return torch.clamp(x, 0, 1)
batch_size = 256
n_epochs = 5

from argparse import Namespace
args = Namespace()

args.use_cuda = True
args.optimizer_type = 'rmsp'
args.save_sub_dir = '.'
args.patience = 5
args.valid_scores_topk = [999] * args.patience
args.greater_is_better = False
args.log_interval = 100
args.valid_interval = 500
args.valid_interval_re_decay = 0.7
args.valid_interval_min = 50
args.lr_re_decay = 0.5

args.batch_size = batch_size
args.lr=0.0001
args.weight_decay=0.0
args.n_epochs=n_epochs
args.model_name='TextCNN'

args.model_params = dict(n_output=1, 
                         maxlen=maxlen, max_features=max_features, embed_size=embed_size, 
                         max_pooling_k=3,
                         embedding_init=embedding_matrix, 
                         in_channel=1, out_channel=16, 
                         kernel_sizes=[1, 2, 4, 8], 
                         dilations=[1, 1, 1, 1], 
                         dropout=0.5, 
                         n_final_state=16)
def get_split_masks(eval_sets, valid_fold, test_fold):
    mask_val = eval_sets==valid_fold
    mask_te = eval_sets==test_fold
    mask_tr = ~mask_val & ~mask_te
    return mask_tr, mask_val, mask_te
valid_fold = 0
mask_tr, mask_val, mask_te = get_split_masks(eval_sets, valid_fold, 10)
model = TextCNN()
model = model.init_args(args, **args.model_params)
model.fit(text[mask_tr], labels[mask_tr], text[mask_val], labels[mask_val])
pred_val_all = model.predict(text[mask_val], use_topk=-1, reduce=False)
for pred_val in pred_val_all:
    print(np.sqrt(mean_squared_error(labels[mask_val], pred_val)))
topk_avg_scores = []
for idx, pred_val in enumerate(pred_val_all):
    topk_avg_scores.append(np.sqrt(mean_squared_error(labels[mask_val], np.mean(pred_val_all[:idx+1], 0))))
    print('top %d'%(idx+1), topk_avg_scores[-1])
topk_wavg_scores = []
for idx, pred_val in enumerate(pred_val_all):
    weights = np.array(args.valid_scores_topk[:idx+1])
    weights = np.exp(-weights)/np.exp(-weights).sum()
    topk_wavg_scores.append(np.sqrt(mean_squared_error(labels[mask_val], 
                                                       np.dot(np.hstack(pred_val_all[:idx+1]), weights.reshape(-1, 1)))))
    print('top %d weighted'%(idx+1), topk_wavg_scores[-1])
if min(topk_avg_scores)<=min(topk_wavg_scores):
    best_topk = np.argmin(topk_avg_scores)+1
    best_reduce_mode = 'mean'
else:
    best_topk = np.argmin(topk_wavg_scores)+1
    best_reduce_mode = 'weighted'
best_valid_score = min(min(topk_avg_scores), min(topk_wavg_scores))
print('best top:', best_topk, 'mode:', best_reduce_mode)
pred_val = model.predict(text[mask_val], use_topk=best_topk, reduce_mode=best_reduce_mode)
np.save('valid_%d_pred.npy'%valid_fold, pred_val)
pred_test = model.predict(text[mask_te], use_topk=best_topk, reduce_mode=best_reduce_mode)
np.save('test_pred.npy', pred_test)
sns.distplot(pred_test)
sns.distplot(labels[mask_val])
sub = pd.read_csv(DATA_DIR+'sample_submission.csv')
sub[target_col] = pred_test
sub.to_csv('textcnn_%.6f.csv'%best_valid_score, index=False)
print('save to', 'textcnn_%.6f.csv'%best_valid_score)
sub.head()
del model; gc.collect()
model = TextCNN()
model = model.init_args(args, **args.model_params)
model.return_final_state = True
test_state = model.predict(text[mask_te], use_topk=best_topk, reduce_mode=best_reduce_mode)
test_state.shape
plt.plot(test_state.mean(0))
valid_state = model.predict(text[mask_val], use_topk=best_topk, reduce_mode=best_reduce_mode)
from scipy import sparse
valid_state = sparse.csr_matrix(valid_state)
test_state = sparse.csr_matrix(test_state)
sparse.save_npz('valid_%d_state.npz'%valid_fold, valid_state, compressed=True)
sparse.save_npz('test_state.npz', test_state, compressed=True)
!ls .