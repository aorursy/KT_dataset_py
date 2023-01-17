import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
x_h = np.linspace(-1, 2, 201)
fig, axes = plt.subplots(figsize=(5, 3))
pd.DataFrame(np.exp(x_h), index=x_h, columns=['exp']).plot(ax=axes)
x_dot = [0, 1]
axes.plot(x_dot, np.exp(x_dot), 'ro', alpha=.6)
print(f"{pd.DataFrame(np.exp(x_dot), index=x_dot, columns=['exp'])}")
def softmax(x):
    if x.ndim == 2:
        x_reshape = x - np.max(x, axis=1).reshape(-1, 1)
        x_exp = np.exp(x_reshape) / np.sum(np.exp(x_reshape), axis=1).reshape(-1, 1)
        
    else:
        x_reshape = x - np.max(x)
#         print(f"{x_reshape}")
        x_exp = np.exp(x_reshape) / np.sum(np.exp(x_reshape))
    
    return x_exp
x_1d = np.array([8, 4, 6]) # axis=1
x_2d = np.array([[8, 4, 6],[4, 5, 3]])
print(f"{x_1d}, {x_1d.shape}, {x_1d.ndim}\n{softmax(x_1d)}, {softmax(x_1d).sum()}")
print(f"{x_2d}, {x_2d.shape}, {x_2d.ndim}\n{softmax(x_2d)}, {[r.sum() for r in softmax(x_2d)]}")

x = np.random.randn(5)
print(f"{x}")
df_list = [x, np.exp(x), np.exp(x) / sum(np.exp(x)), x - np.max(x), np.exp(x - np.max(x)), softmax(x)]
fig, axes = plt.subplots(2, 3, figsize=(10, 3), subplot_kw={'xticks': []})
for i, ax in enumerate(axes.ravel()):
    pd.DataFrame(df_list[i]).plot.bar(ax=ax, legend=None) ; ax.set_xticklabels('')
print(f"{np.exp(x)}, {np.argmax(np.exp(x))}, {np.max(np.exp(x))}")
print(f"{x - np.max(x)}")
print(f"{np.exp(x - np.max(x))}")
print(f"{softmax(x)}, {np.sum(softmax(x))}")
fig, axes = plt.subplots(figsize=(5, 4))
x_log = [n for n in x_h if n >0]
x_dot = [np.min(x_log), 1, np.max(x_log)]
pd.DataFrame(np.log(x_log), index=x_log, columns=['log']).plot(ax=axes)
axes.plot(x_dot, np.log(x_dot), 'ro', alpha=.6)
pd.DataFrame(np.log(x_dot), index=x_dot, columns=['log'])
a = np.array([[4.24149613,  1.46506529,  0.52470428,  0.24003547, -1.90880457,
         2.6975341 ,  0.23592045, -0.93879469, -0.23129324, -0.24120573],
       [ 0.09215836,  4.47913275,  0.69673186,  1.04265416,  2.51861902,
        -0.66210524,  0.5992323 , -0.71243305,  0.31144567,  0.45094324],
       [ 0.19574476, -0.68073994, -1.59140216, 3.48846268,  0.57213068,
        -0.90115776,  0.82886671,  0.26646719, -3.66120162,  0.18969919]])
y = softmax(a)
print(f"{y.shape}, {y.ndim}, {y.size}\n{y.argmax(axis=1)}")

fig, axes = plt.subplots(3, 1, figsize=(8, 3))
for i, ax in enumerate(axes.ravel()):
    pd.DataFrame(y[i]).plot.bar(ax=ax, legend=None)
fig.tight_layout()
t_array = np.array([0, 1, 3])
t = np.identity(10)[t_array]
print(f"{t_array}, {type(t_array)}, {t_array.shape}, {t_array.ndim}, {t_array.size}")
print(f"{t}, {t.shape}, {t.ndim}, {t.size}")
print(f"{np.argmax(t, axis=1)}, {np.argmax(t, axis=1).shape}, {np.argmax(t, axis=1).ndim}, {np.argmax(t, axis=1).size}")
print(f"what we want: {y[[0, 1, 2], [0, 1, 3]]}") # この書き方
batch_size = y.shape[0] # y.shape[0] でバッチサイズを取得して
print(f"batch_size: {batch_size}, {np.arange(batch_size)}") # np.arange(batch_size) でインデクス
print(f"what we got:  {y[np.arange(batch_size), np.argmax(t, axis=1)]}") # この書き方
print(f"{-np.log(y[np.arange(batch_size), np.argmax(t, axis=1)])}")
cross_entropy_err = -np.sum(np.log(y[np.arange(batch_size), np.argmax(t, axis=1)])) / batch_size
fig, axes = plt.subplots(figsize=(6,1))
pd.DataFrame(-np.log(y[np.arange(batch_size), np.argmax(t, axis=1)])).plot.bar(ax=axes, legend=None)
axes.hlines(cross_entropy_err, -1, batch_size, colors='r', alpha=.5)
print(f"{cross_entropy_err}")
t_array_b = np.array([4, 5, 8])
t_b = np.identity(10)[t_array_b]
print(f"{t_b}")
print(f"{y[np.arange(batch_size), np.argmax(t_b, axis=1)]} # 値が小さい（確率が低い）smaller values (low probability)")
print(f"{-np.log(y[np.arange(batch_size), np.argmax(t_b, axis=1)])}")
cross_entropy_err_b = -np.sum(np.log(y[np.arange(batch_size), np.argmax(t_b, axis=1)])) / batch_size
fig, axes = plt.subplots(figsize=(6,1))
pd.DataFrame(-np.log(y[np.arange(batch_size), np.argmax(t_b, axis=1)])).plot.bar(ax=axes, legend=None)
axes.hlines(cross_entropy_err_b, -1, batch_size, colors='r', alpha=.5)
print(f"{cross_entropy_err_b} # larger value of cross_entropy_error") # 交差エントロピー誤差の値が大きくなる
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    
    if y.shape == t.shape: # 教師データが one_hot_vector の場合は shape が一致するので
        t = t.argmax(axis=1) #; print(f"t = t.argmax(axis=1): {t}")
    
    batch_size = y.shape[0] #; print(f"batch_size: {batch_size}")
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
t_one = np.array([2])
y_one = softmax(np.array([ 0.80460243,  0.10041554, 4.38357109,  0.05092654,  1.78566402,
     0.44697127, -1.96202594, -0.05063143, -0.1603952 ,  1.4166856 ])) # y_one 再定義（上で二次元に変更しているので）
print(f"{y_one}, {y_one.shape}, {y_one.ndim}, {y_one.size}")
print(f"t_one: {t_one}, {t_one.shape}, {t_one.ndim}, {t_one.size}")
t_one_hot = np.identity(10)[t_one] # データ１個の ohe_hot_vector
print(f"t_one_hot: {t_one_hot}, {t_one_hot.shape}, {t_one_hot.ndim}, {t_one_hot.size}")
print(f"{cross_entropy_error(y_one, t_one)}")  # 正解１個のラベル
print(f"{cross_entropy_error(y_one, t_one_hot)}")  # 正解１個の one_hot_vector
print(f"{cross_entropy_error(y, t_array)}")  # 正解複数のラベル
print(f"{cross_entropy_error(y, t)}")  # 正解複数の one_hot_vector
x = np.linspace(-5, 5, 101)
y1 = np.exp(x)
y2 = np.exp(-x)
y3 = y2 + 1
y4 = 1 / y3
y_list, colors = [y1, y2, y3, y4], ['red', 'pink', 'yellow', 'blue']
fig, axes = plt.subplots(figsize=(4, 3))
axes.set_ylim(-0.1, 3.1)

for i, y in enumerate(y_list):
    axes.plot(x, y, c=colors[i], alpha=.8)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 確認
x = np.linspace(-5, 5, 101)
y = sigmoid(x)
fig, axes = plt.subplots(figsize=(4, 2))
axes.plot(x, y)
axes.set_xticks([n for n in x if n%1 == 0]) ; fig.tight_layout()
def f_six(x):
    return (x[0] - 3)**2 + (x[1] - 2)**2 + (x[2] - 1)**2 + (x[3])**2 + (x[4] +1)**2 + (x[5] + 2)**2
t_weight = np.random.randint(-10, 10, (2, 3)).astype(float)
# t_weight = np.zeros(6).reshape(2, 3) + 1
shape_zero = t_weight.shape[0]
print(f"{t_weight}, {shape_zero}")
t_weight = t_weight.reshape(-1) # 上記関数の x のインデクスに合わせるために一次に変換
print(f"{t_weight}, {t_weight.shape}")
print(f"{f_six(t_weight)}")
h = 1e-4  # 0.0001
grad = np.zeros_like(t_weight) ; print(f"{grad}")
it = np.nditer(t_weight, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index
    print(f"{idx}: {t_weight[idx]: >5}")
    var_tmp = t_weight[idx] # 変数 var_tmp に値を代入
    
    t_weight[idx] = var_tmp + h # h を加えた値を戻して
    fplus = f_six(t_weight) ; print(f"{t_weight[idx]: >7}, {fplus}") # 前方差分 forward differential
    
    t_weight[idx] = var_tmp - h
    fminus = f_six(t_weight) ; print(f"{t_weight[idx]: >7}, {fminus}") # 後方差分 backward differential
    
    diff = (fplus - fminus) / (2 * h)
    grad[idx] = diff ; print(f"{diff}\n{grad}")
    
    t_weight[idx] = var_tmp
    it.iternext()
# t_weight 再設定
t_weight = np.random.randint(-20, 20, (2, 3)).astype(float)
shape_zero = t_weight.shape[0]
print(f"{t_weight}, {shape_zero}")
t_weight = t_weight.reshape(-1) # 上記関数の x のインデクスに合わせるために一次に変換
print(f"{t_weight}, {t_weight.shape}")
print(f"{f_six(t_weight)}")
fig, axes = plt.subplots(figsize=(8, 1.5))
pd.DataFrame(t_weight).plot.bar(ax=axes, legend=False) ; fig.tight_layout()
h = 1e-4
learning_rate = 0.1
n_iter = 50
for i in range(n_iter):
    grad = np.zeros_like(t_weight)
    it = np.nditer(t_weight, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        var_tmp = t_weight[idx] #; print(f"{idx}: {t_weight[idx]: }")
        
        t_weight[idx] = var_tmp + h
        fplus = f_six(t_weight) #; print(f"{t_weight[idx]}, {fplus}")
        
        t_weight[idx] = var_tmp - h
        fminus = f_six(t_weight) #; print(f"{t_weight[idx]}, {fminus}")
        
        diff = (fplus - fminus) / (2 * h)
        grad[idx] = diff #; print(f"{diff}\n{grad}")
        
        t_weight[idx] = var_tmp
        it.iternext()
    
    t_weight -= learning_rate * grad
    
    print_batch = 10
    if i % print_batch == 0:
        print(f"{i:>2}:{t_weight}")
    
    if i % print_batch == 0:
        fig, axes = plt.subplots(figsize=(8, 1.5))
        pd.DataFrame(t_weight).plot.bar(ax=axes, legend=False) ; fig.tight_layout()
        axes.set_title(i)
print(f"{t_weight}") # ほぼ最小値に収束している almost minimun values
t_weight.reshape(shape_zero, -1)
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        var_tmp = x[idx]
        
        x[idx] = var_tmp + h
        fplus = f(x)
        
        x[idx] = var_tmp - h
        fminus = f(x)
        
        diff = (fplus - fminus) / (2 * h)
        grad[idx] = diff
        
        x[idx] = var_tmp
        it.iternext()
    
    return grad    
# t_weight 再設定
t_weight = np.random.randint(-20, 20, (2, 3)).astype(float)
shape_zero = t_weight.shape[0]
print(f"{t_weight}, {shape_zero}")
t_weight = t_weight.reshape(-1) # 上記関数の x のインデクスに合わせるために一次に変換
print(f"{t_weight}, {t_weight.shape}")
print(f"{f_six(t_weight)}")
fig, axes = plt.subplots(figsize=(8, 1.5))
pd.DataFrame(t_weight).plot.bar(ax=axes, legend=False) ; fig.tight_layout()
learning_rate = 0.1
n_iter = 50
interval = 10
for i in range(n_iter):
    grad = numerical_gradient(f_six, t_weight)
    t_weight -= learning_rate * grad
    if i % interval == 0:
        print(f"{t_weight}")
        fig, axes = plt.subplots(figsize=(8, 1.5))
        pd.DataFrame(t_weight).plot.bar(ax=axes, legend=False) ; fig.tight_layout()
        axes.set_title(i)
x_size, m_size, out_size = 5, 3, 10
simple_x = np.random.randn(x_size, m_size)
s_weight = np.random.randn(m_size, out_size)
print(f"{simple_x.shape}, {s_weight.shape}")
t = np.random.randint(0, x_size, x_size)
print(f"{t}")
simple_t = np.eye(x_size, out_size)[t]
print(f"{simple_t.shape}\n{simple_t.argmax(axis=1)}")
def simple_predict(x):
    z = np.dot(x, s_weight)
    y = softmax(z)
    return y

def s_loss(x, t):
    y = simple_predict(x)
    return cross_entropy_error(y, t)    
s_pred = simple_predict(simple_x)
s_pred_argmax = s_pred.argmax(axis=1)
print(f"{s_pred.shape}\n{s_pred_argmax}\n{s_pred_argmax == t}\n{np.mean(s_pred_argmax == t)}")
print(f"{cross_entropy_error(s_pred, simple_t)}")
print(f"{s_loss(simple_x, simple_t)}")
dummy_loss = lambda _: s_loss(simple_x, simple_t)
print(f"{type(dummy_loss)}, {dummy_loss(3)}, {dummy_loss(24)}")
learning_rate = 0.1
step_num, n_show = 1000, 100

loss_list, accuracy_list = [], []
for i in range(step_num):
    grad = numerical_gradient(dummy_loss, s_weight) # get grad
    s_weight -= learning_rate * grad  ### gradient descent
    loss_tmp = s_loss(simple_x, simple_t)
    loss_list.append(loss_tmp)
    pred_argmax = simple_predict(simple_x).argmax(axis=1)
    acc_rate = np.mean(pred_argmax == t)
    accuracy_list.append(acc_rate)
    if i % n_show == 0:
        pred = simple_predict(simple_x)
        n_row = pred.shape[0]
        fig, axes = plt.subplots(n_row, 1, figsize=(8, n_row))
        for idx, ax in enumerate(axes.ravel()):
            pd.DataFrame(pred[idx]).plot.bar(ax=ax, legend=False)
        fig.suptitle("{}: pred: {}, t: {}, {}".format(i, pred_argmax, t, acc_rate))
fig, axes = plt.subplots(2, 1, figsize=(12, 3))
list_two = [loss_list, accuracy_list]
for i, ax in enumerate(axes.ravel()):
    pd.DataFrame(list_two[i]).plot(ax=ax)
weight_init_sd = 0.01
input_size, hidden_size, output_size = 784, 10, 10

W1 = weight_init_sd * np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = weight_init_sd * np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

print(f"{W1.shape}, {b1.shape}, {W2.shape}, {b1.shape}")
x_size = 5
x_test = np.random.randn(x_size, input_size)
x_test.shape
def predict(x):
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)
    
    return y
pred_test = predict(x_test)
print(f"{pred_test.shape}\n{pred_test.argmax(axis=1)}")
for i, r in enumerate(pred_test):
    print(f"{i}: {r.sum()}")
select_r = np.random.randint(0, x_size, x_size)
t_test = np.eye(x_size, output_size)[select_r]
print(f"{t_test}, {t_test.shape}, {t_test.ndim}\n{t_test.argmax(axis=1)}")
def loss(x, t):
    y = predict(x)
    return cross_entropy_error(y, t)   
loss(x_test, t_test)
def get_accuracy(x, t):
    pred = predict(x).argmax(axis=1)
    if t.ndim == 2:
        t = t.argmax(axis=1)
    return np.mean(pred == t)
get_accuracy(x_test, t_test)
# def dummy_loss(_):
#     return loss(x_test, t_test)
dummy_loss = lambda _: loss(x_test, t_test)
dummy_loss(3)
lr = 0.05 ; step_n = 100
loss_list, acc_list = [], []
n_show = step_n / 5

import time ; stime = time.time()

b_list = [W1, b1, W2, b2]
for i in range(step_n):
    for bias in b_list:
        b_time = time.time()
        grad = numerical_gradient(dummy_loss, bias)
        bias -= lr * grad
        loss_val = loss(x_test, t_test)
        loss_list.append(loss_val)
        acc_list.append(get_accuracy(x_test, t_test))
        p_argmax, t_argmax = predict(x_test).argmax(axis=1), t_test.argmax(axis=1)
    
    if i % n_show == 0 or i == [n for n in range(step_n)][-1]:
        pred = predict(x_test)
        n_row = pred.shape[0]
        fig, axes = plt.subplots(n_row, 1, figsize=(8, n_row))
        for idx, ax in enumerate(axes.ravel()):
            pd.DataFrame(pred[idx]).plot.bar(ax=ax, legend=False)
        fig.suptitle("{}: pred: {}, t: {}, {}, {:.2f} sec.".format(i, p_argmax, t_argmax, get_accuracy(x_test, t_test), 
                                                                   time.time() - stime))
lists = [loss_list, acc_list]
fig, axes = plt.subplots(len(lists), 1, figsize=(8, 3))
for i, ax in enumerate(axes.ravel()):
    pd.DataFrame(lists[i]).plot(ax=ax)
fig, axes = plt.subplots(n_row, 1, figsize=(8, n_row))
for i, ax in enumerate(axes.ravel()):
    pd.DataFrame(predict(x_test)[i]).plot.bar(ax=ax, legend=False)
print(f"{p_argmax}, {t_argmax}")
import os
os.listdir('../input')
train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')
# train_csv = pd.read_csv('./data/kaggle_mnist/train.csv')
# test_csv = pd.read_csv('./data/kaggle_mnist/test.csv')
X_train = train_csv.iloc[:, 1:].values.astype('float32') / 255.0
y_train = train_csv.iloc[:, [0]].values.ravel().astype('int32')
X_test = test_csv.values.astype('float32') / 255.0
print(f"{X_train.shape}, {y_train.shape}, {X_test.shape}")
fig, axes = plt.subplots(figsize=(7, 2))
pd.DataFrame(y_train).iloc[:, 0].value_counts(sort=False).plot.bar(ax = axes, legend=None) ; fig.tight_layout()
print(f"{pd.DataFrame(X_train).isnull().any().value_counts()}")
print(f"{pd.DataFrame(X_test).isnull().any().value_counts()}")
weight_init_sd = 0.01
input_size, hidden_size, output_size = 784, 10, 10

W1 = weight_init_sd * np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = weight_init_sd * np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

print(f"{W1.shape}, {b1.shape}, {W2.shape}, {b1.shape}")
fig, axes = plt.subplots(3, 15, figsize=(15, 3), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_train[i].reshape(28, 28), cmap=plt.cm.Greys_r)
    ax.set_title(y_train[i])
fig.tight_layout()
fig, axes = plt.subplots(3, 15, figsize=(15, 3), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(28, 28), cmap=plt.cm.Greys_r)
    ax.set_title(predict(X_test).argmax(axis=1)[i])
fig.tight_layout()
get_accuracy(X_train, y_train)
np.mean(predict(X_train).argmax(axis=1) == y_train)
fig, axes = plt.subplots(1, 10, figsize=(12, 1), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.ravel()):
    ax.imshow(W1[:, i].reshape(28, 28), cmap=plt.cm.gray_r)
train_size = X_train.shape[0]
batch_size = 100
epoch_num = train_size / batch_size ; print(f"train_size: {train_size}, batch_size: {batch_size}, epoch_num: {epoch_num}")
batch_mask = np.random.choice(train_size, batch_size)
batch_mask.shape
X_batch, y_batch = X_train[batch_mask], y_train[batch_mask]
print(f"{X_batch.shape}, {y_batch.shape}")
loss(X_batch, y_batch)
dummy_loss_batch = lambda _: loss(X_batch, y_batch)
dummy_loss_batch(98)
train_loss_list = []
train_acc_list = []
# from sklearn.externals import joblib
# W1 = joblib.load("dump_W1.pkl")
# b1 = joblib.load("dump_b1.pkl")
# W2 = joblib.load("dump_W2.pkl")
# b2 = joblib.load("dump_b2.pkl")
# train_loss_list = joblib.load("dump_train_loss_list.pkl")
# train_acc_list  = joblib.load("dump_train_acc_list.pkl")
import time ; s_time = time.time()

train_size = X_train.shape[0]
batch_size = 100

iters_n = 3000
n_show = iters_n / 10
lr = 0.1

for i in range(iters_n):
    i_time = time.time()
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch, y_batch = X_train[batch_mask], y_train[batch_mask]
    dummy_loss_batch = lambda _: loss(X_batch, y_batch)
    param_list = [W1, b1, W2, b2]
    for ip, param in enumerate(param_list):
        p_time = time.time()
        grad = numerical_gradient(dummy_loss_batch, param)
        param -= lr * grad
        new_loss = loss(X_batch, y_batch)
    train_loss_list.append(new_loss)
    train_acc_list.append(get_accuracy(X_batch, y_batch))
    
    if i % n_show == 0 or i == [n for n in range(iters_n)][-1]:
        print(f"{i:>4}: {new_loss:.6f}, {get_accuracy(X_train, y_train):.6f}, {time.time() - i_time:>8.2f} sec., \
 {time.time() - s_time:>8.2f} sec., {(time.time() - s_time) / 60:>6.2f} min., {(time.time() - s_time) / 3600:.2f} hour.")    
lists, list_names = [train_loss_list, train_acc_list], ['loss', 'accuracy']
fig, axes = plt.subplots(len(lists), 1, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    pd.DataFrame(lists[i]).plot(ax=ax)
    ax.set_title(list_names[i])
fig.tight_layout()
# from sklearn.externals import joblib
# dump_list = [W1, b1, W2, b2, train_loss_list, train_acc_list]
# dump_names = ["dump_W1", "dump_b1", "dump_W2", "dump_b2", "dump_train_loss_list", 
#               "dump_train_acc_list"]

# for i, param in enumerate(dump_list):
#     print(dump_names[i] + '.pkl')
#     joblib.dump(param, dump_names[i] + '.pkl')
get_accuracy(X_train, y_train)
fig, axes = plt.subplots(1, 10, figsize=(12, 1), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.ravel()):
    ax.imshow(W1[:, i].reshape(28, 28), cmap=plt.cm.gray_r)
fig, axes = plt.subplots(3, 15, figsize=(15, 3), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(28, 28), cmap=plt.cm.Greys_r)
    ax.set_title(predict(X_test).argmax(axis=1)[i])
fig.tight_layout()
false_i = ~(predict(X_train).argmax(axis=1) == y_train)

fig, axes = plt.subplots(3, 15, figsize=(15, 3), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_train[false_i][i].reshape(28, 28), cmap=plt.cm.Greys_r)
    ax.set_title("{} to {}".format(predict(X_train).argmax(axis=1)[false_i][i], y_train[false_i][i]))
fig.tight_layout()
label_id = np.arange(X_test.shape[0]) + 1
print(f"{label_id}")
df_submission = pd.DataFrame({'ImageId': label_id, 'Label': predict(X_test).argmax(axis=1)})
df_submission.to_csv('submission.csv', index=False)