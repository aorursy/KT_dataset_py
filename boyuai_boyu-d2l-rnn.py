import torch

import torch.nn as nn

import time

import math

import sys

sys.path.append("../input/")

import d2lzhpytorch as d2l

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def one_hot(x, n_class, dtype=torch.float32):

    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)  # shape: (n, n_class)

    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1

    return result

    

x = torch.tensor([0, 2])

x_one_hot = one_hot(x, vocab_size)

print(x_one_hot)

print(x_one_hot.shape)

print(x_one_hot.sum(axis=1))
def to_onehot(X, n_class):

    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]



X = torch.arange(10).view(2, 5)

inputs = to_onehot(X, vocab_size)

print(len(inputs), inputs[0].shape)
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

# num_inputs: d

# num_hiddens: h, 隐藏单元的个数是超参数

# num_outputs: q



def get_params():

    def _one(shape):

        param = torch.zeros(shape, device=device, dtype=torch.float32)

        nn.init.normal_(param, 0, 0.01)

        return torch.nn.Parameter(param)



    # 隐藏层参数

    W_xh = _one((num_inputs, num_hiddens))

    W_hh = _one((num_hiddens, num_hiddens))

    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device))

    # 输出层参数

    W_hq = _one((num_hiddens, num_outputs))

    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device))

    return (W_xh, W_hh, b_h, W_hq, b_q)
def rnn(inputs, state, params):

    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵

    W_xh, W_hh, b_h, W_hq, b_q = params

    H, = state

    outputs = []

    for X in inputs:

        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)

        Y = torch.matmul(H, W_hq) + b_q

        outputs.append(Y)

    return outputs, (H,)
def init_rnn_state(batch_size, num_hiddens, device):

    return (torch.zeros((batch_size, num_hiddens), device=device), )
print(X.shape)

print(num_hiddens)

print(vocab_size)

state = init_rnn_state(X.shape[0], num_hiddens, device)

inputs = to_onehot(X.to(device), vocab_size)

params = get_params()

outputs, state_new = rnn(inputs, state, params)

print(len(inputs), inputs[0].shape)

print(len(outputs), outputs[0].shape)

print(len(state), state[0].shape)

print(len(state_new), state_new[0].shape)
def grad_clipping(params, theta, device):

    norm = torch.tensor([0.0], device=device)

    for param in params:

        norm += (param.grad.data ** 2).sum()

    norm = norm.sqrt().item()

    if norm > theta:

        for param in params:

            param.grad.data *= (theta / norm)
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,

                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):

    state = init_rnn_state(1, num_hiddens, device)

    output = [char_to_idx[prefix[0]]]   # output记录prefix加上预测的num_chars个字符

    for t in range(num_chars + len(prefix) - 1):

        # 将上一时间步的输出作为当前时间步的输入

        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)

        # 计算输出和更新隐藏状态

        (Y, state) = rnn(X, state, params)

        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符

        if t < len(prefix) - 1:

            output.append(char_to_idx[prefix[t + 1]])

        else:

            output.append(Y[0].argmax(dim=1).item())

    return ''.join([idx_to_char[i] for i in output])
predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,

            device, idx_to_char, char_to_idx)
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,

                          vocab_size, device, corpus_indices, idx_to_char,

                          char_to_idx, is_random_iter, num_epochs, num_steps,

                          lr, clipping_theta, batch_size, pred_period,

                          pred_len, prefixes):

    if is_random_iter:

        data_iter_fn = d2l.data_iter_random

    else:

        data_iter_fn = d2l.data_iter_consecutive

    params = get_params()

    loss = nn.CrossEntropyLoss()



    for epoch in range(num_epochs):

        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态

            state = init_rnn_state(batch_size, num_hiddens, device)

        l_sum, n, start = 0.0, 0, time.time()

        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)

        for X, Y in data_iter:

            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态

                state = init_rnn_state(batch_size, num_hiddens, device)

            else:  # 否则需要使用detach函数从计算图分离隐藏状态

                for s in state:

                    s.detach_()

            # inputs是num_steps个形状为(batch_size, vocab_size)的矩阵

            inputs = to_onehot(X, vocab_size)

            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵

            (outputs, state) = rnn(inputs, state, params)

            # 拼接之后形状为(num_steps * batch_size, vocab_size)

            outputs = torch.cat(outputs, dim=0)

            # Y的形状是(batch_size, num_steps)，转置后再变成形状为

            # (num_steps * batch_size,)的向量，这样跟输出的行一一对应

            y = torch.flatten(Y.T)

            # 使用交叉熵损失计算平均分类误差

            l = loss(outputs, y.long())

            

            # 梯度清0

            if params[0].grad is not None:

                for param in params:

                    param.grad.data.zero_()

            l.backward()

            grad_clipping(params, clipping_theta, device)  # 裁剪梯度

            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均

            l_sum += l.item() * y.shape[0]

            n += y.shape[0]



        if (epoch + 1) % pred_period == 0:

            print('epoch %d, perplexity %f, time %.2f sec' % (

                epoch + 1, math.exp(l_sum / n), time.time() - start))

            for prefix in prefixes:

                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,

                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2

pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,

                      vocab_size, device, corpus_indices, idx_to_char,

                      char_to_idx, True, num_epochs, num_steps, lr,

                      clipping_theta, batch_size, pred_period, pred_len,

                      prefixes)
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,

                      vocab_size, device, corpus_indices, idx_to_char,

                      char_to_idx, False, num_epochs, num_steps, lr,

                      clipping_theta, batch_size, pred_period, pred_len,

                      prefixes)
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps, batch_size = 35, 2

X = torch.rand(num_steps, batch_size, vocab_size)

state = None

Y, state_new = rnn_layer(X, state)

print(Y.shape, state_new.shape)
class RNNModel(nn.Module):

    def __init__(self, rnn_layer, vocab_size):

        super(RNNModel, self).__init__()

        self.rnn = rnn_layer

        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 

        self.vocab_size = vocab_size

        self.dense = nn.Linear(self.hidden_size, vocab_size)



    def forward(self, inputs, state):

        # inputs.shape: (batch_size, num_steps)

        X = to_onehot(inputs, vocab_size)

        X = torch.stack(X)  # X.shape: (num_steps, batch_size, vocab_size)

        hiddens, state = self.rnn(X, state)

        hiddens = hiddens.view(-1, hiddens.shape[-1])  # hiddens.shape: (num_steps * batch_size, hidden_size)

        output = self.dense(hiddens)

        return output, state
def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,

                      char_to_idx):

    state = None

    output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符

    for t in range(num_chars + len(prefix) - 1):

        X = torch.tensor([output[-1]], device=device).view(1, 1)

        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数

        if t < len(prefix) - 1:

            output.append(char_to_idx[prefix[t + 1]])

        else:

            output.append(Y.argmax(dim=1).item())

    return ''.join([idx_to_char[i] for i in output])
model = RNNModel(rnn_layer, vocab_size).to(device)

predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,

                                corpus_indices, idx_to_char, char_to_idx,

                                num_epochs, num_steps, lr, clipping_theta,

                                batch_size, pred_period, pred_len, prefixes):

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(num_epochs):

        l_sum, n, start = 0.0, 0, time.time()

        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样

        state = None

        for X, Y in data_iter:

            if state is not None:

                # 使用detach函数从计算图分离隐藏状态

                if isinstance (state, tuple): # LSTM, state:(h, c)  

                    state[0].detach_()

                    state[1].detach_()

                else: 

                    state.detach_()

            (output, state) = model(X, state) # output.shape: (num_steps * batch_size, vocab_size)

            y = torch.flatten(Y.T)

            l = loss(output, y.long())

            

            optimizer.zero_grad()

            l.backward()

            grad_clipping(model.parameters(), clipping_theta, device)

            optimizer.step()

            l_sum += l.item() * y.shape[0]

            n += y.shape[0]

        



        if (epoch + 1) % pred_period == 0:

            print('epoch %d, perplexity %f, time %.2f sec' % (

                epoch + 1, math.exp(l_sum / n), time.time() - start))

            for prefix in prefixes:

                print(' -', predict_rnn_pytorch(

                    prefix, pred_len, model, vocab_size, device, idx_to_char,

                    char_to_idx))
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2

pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,

                            corpus_indices, idx_to_char, char_to_idx,

                            num_epochs, num_steps, lr, clipping_theta,

                            batch_size, pred_period, pred_len, prefixes)