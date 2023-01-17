import numpy as np

FEAT_TYPE = {0: 'cat', 1: 'text', 2: 'cat', 3: 'num', 4: 'cat', 5: 'cat', 6: 'text', 7: 'num', 8: 'set', 9: 'cat'}



# parse a string into fields, skip quotations

def parse_feat(line):

    quota = False

    j = 0

    feats = []

    for i in range(len(line)):

        if line[i] == '\"':

            quota = not quota

        if line[i] == ',' and not quota:

            feat = line[j:i]

            feats.append(feat)    # make [a,b,c,d,e] to [a b c d e]

            j = i+1

    return feats + [line[j:]]



# load a csv file, use parse_feat() to convert format

def load_file(file_name):

    data = []

    with open(file_name, 'r') as fin:

        print('field_names:', fin.readline().strip().split(',')) # print the first row

        for line in fin:

            line = line.strip() # 去掉杂七杂八的部分，剩下字符串

            data.append(parse_feat(line))

    return np.array(data)



train_data = load_file('../input/train.csv')

test_data = load_file('../input/test.csv')



train_id, train_label, train_feat = train_data[:, 0], train_data[:, 1], train_data[:, 2:]

test_id, test_feat = test_data[:, 0], test_data[:, 1:]



train_feat[:, [1, 6]] = None

test_feat[:, [1, 6]] = None   # get rid of useless information



print('train_feat:\n', train_feat[0])

print('test_feat:\n', test_feat[0])
# for type 'category', key of dictionary is (index:feature_name),such as 2:male

def get_feat_name(field, feat_val):

    assert field != 'text'

    if FEAT_TYPE[field] == 'cat':  

        return str(field) + ':' + feat_val

    elif FEAT_TYPE[field] == 'num':

        return str(field) + ':'

    elif FEAT_TYPE[field] == 'set':

        return [str(field) + ':' + fv for  fv in feat_val.split()]



    

# find out alternatives of each feature.

# for example, 'Embark' can be 'S' 'Q' 'C' or 'NULL'

# value of dictionary is the sequence number of itself in the dictionary

def build_feat_map(data):  

    feat_map = {}   # a dictionary

    for i in range(len(FEAT_TYPE)):  # for each item in the dictionary

        if FEAT_TYPE[i] == 'num':

            fn = get_feat_name(i, None)

            if fn not in feat_map:

                feat_map[fn] = len(feat_map)

            continue

        elif FEAT_TYPE[i] == 'text': # ignore text

            continue

            

        feat = data[:, i]

        for f in feat:

            if FEAT_TYPE[i] == 'cat':

                fn = get_feat_name(i, f)

                if fn not in feat_map:

                    feat_map[fn] = len(feat_map)

            elif FEAT_TYPE[i] == 'set':

                for fn in get_feat_name(i, f):

                    if fn not in feat_map:

                        feat_map[fn] = len(feat_map)

                    

    return feat_map

    

feat_map = build_feat_map(np.vstack([train_feat, test_feat])) # vertical stack, means expand in row

print(feat_map)

print(np.array(feat_map.keys()))

print(np.array(feat_map.values()))
def to_float(x):

    if len(x):

        return float(x)

    return -1





# for catagories, returns (sequence number,1)

# for numbers, returns (sequence number, item itself) or (sequence number, -1)

# the seqence number is the first time the new feature appears

def get_feat_id_val(field, feat_val):

    assert field != 'text'

    feat_name = get_feat_name(field, feat_val)

    if FEAT_TYPE[field] == 'cat':

        return feat_map[feat_name], 1  

    elif FEAT_TYPE[field] == 'num':

        return feat_map[feat_name], to_float(feat_val)

    elif FEAT_TYPE[field] == 'set':

        return [feat_map[fn] for fn in feat_name], [1] * len(feat_name)





# fv is (a,b)

# a is the value in the dictionary 'feat_map'

# b is the number(not 1) or 1(it represents the key in dictionary, for example, a = 40,b = 1, the value of the key '8:G73' is 40)

def to_libsvm(data):

    libsvm_data = []

    for d in data:  # each row in data

        libsvm_data.append([])

        for i in range(len(FEAT_TYPE)):

            if FEAT_TYPE[i] == 'cat' or FEAT_TYPE[i] == 'num':

                fv = get_feat_id_val(i, d[i]) # each item in the row d

                libsvm_data[-1].append(fv) #倒数第一个位置append

            elif FEAT_TYPE[i] == 'set':

                fvs = get_feat_id_val(i, d[i])

                for fv in zip(*fvs):

                    libsvm_data[-1].append(fv)

    return libsvm_data



# attention! some without several feature (without Age/Cabin, etc.)

train_data = to_libsvm(train_feat)

test_data = to_libsvm(test_feat)





# 80% of the origin train set as new train set and others valid set

# np.arrange(x) creates a array [1 2 3 ... x]

# [:x] 直接用于取数组的前x个元素 

size = len(train_data)

train_ind = np.arange(size)[: int(0.8 * size)]

valid_ind = np.arange(size)[int(0.8 * size): ]



# %d 是带符号的十进制整形数

# \n 单纯是换个行

MAX_FEAT = ' %d:0\n' % len(feat_map) 



flag = True

with open('train.svm', 'w') as fout:

    for i in train_ind:

        line = train_label[i]  

        for fv in train_data[i]:

            line += ' {}:{}'.format(*fv)

        if flag:

            line += MAX_FEAT

            flag = False

        else:

            line += '\n'

        fout.write(line)

        

# While train_data[0] is[(0, 1), (3, 1), (5, 22.0), (6, 1), (13, 1), (21, 7.25), (224, 1)]

# line0 is 0 0:1 3:1 5:22.0 6:1 13:1 21:7.25 224:1 228:0

        

flag = True

with open('valid.svm', 'w') as fout:

    for i in valid_ind:

        line = train_label[i]

        for fv in train_data[i]:

            line += ' {}:{}'.format(*fv)

        if flag:

            line += MAX_FEAT

            flag = False

        else:

            line += '\n'

        fout.write(line)
import xgboost as xgb

# read in data

dtrain = xgb.DMatrix('train.svm')

dtest = xgb.DMatrix('valid.svm')



# TODO tune params for gbtree

param = {

    # learner params

    'booster': 'gbtree', # gbtree or gblinear

    'nthread': 1,        # CPU线程数

    'silent': 1,         # 1则无运行信息输出，0则有

    # tree params

    'eta': 0.8,            # 类似学习率，0.8

    'gamma': 0,          # 用于控制是否后剪枝的参数，越大越保守，0

    'max_depth': 4,      # 树的深度，越大越容易过拟合，4

    'subsample': 0.8,      # 随机采样训练样本，越大越容易过拟合，一般0.5-1，0.8

    'lambda': 2,         # 控制模型复杂度的权重的L2正则化参数，越大越不容易过拟合，2/1.95

    'alpha': 0,          # 权重的L1正则化项，0

    # learning params

    'objective': 'binary:logistic',  

    'eval_metric': 'error', } # 还有rmse，mae，logloss，auc，error



# [5]	train-error:0.13624	eval-error:0.12849

evallist = [(dtrain, 'train'), (dtest, 'eval')]

num_round = 100

bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=5)
import lightgbm as lgb



dtrain = lgb.Dataset('train.svm')

dtest = lgb.Dataset('valid.svm')



# TODO tune params for gbdt

param = {

    'objective':'binary',

    'boosting': 'gbdt',

    'num_threads': 1,     # 1

    'learning_rate': 1,   # 学习率,1

    'num_leaves': 31,     # 31

    'max_depth': 11,       # 11

    'metric': 'binary_error',

    'lambda_l1': 0,       # 0

    'lambda_l2': 2,       # 2

    }

#[4]	valid_0's binary_error: 0.117318

num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dtest], early_stopping_rounds=5)
gbdt_train_data, gbdt_valid_data = train_data[: int(0.8 * len(train_data))], train_data[int(0.8 * len(train_data)): ]

gbdt_train_label = list(map(int, train_label))

gbdt_train_label, gbdt_valid_label = gbdt_train_label[: int(0.8 * len(gbdt_train_label))], gbdt_train_label[int(0.8 * len(gbdt_train_label)): ]





# 1的位置表明样本拥有的具体属性，与dictionary顺序相同

# 数字位置同理

def to_train_data(input):

    output = list()

    for sample in input:   #input[0] = [(0, 1), (3, 1), (5, 22.0), (6, 1), (13, 1), (21, 7.25), (224, 1)]

        sample_vec = [0] * len(feat_map)  # zero copy len(feat_map)=228 times, and sample_vec = [0 0 0 ... 0]

        for feature in sample:

            sample_vec[feature[0]] = feature[1]

        output.append(sample_vec)

    return output



# 1在前表示死了，1在后表示活了

def to_train_label(input):

    output = list()

    for i in input:

        label_vec = [0] * 2

        label_vec[i] = 1

        output.append(label_vec)

    return output



gbdt_train_data = to_train_data(gbdt_train_data)

gbdt_valid_data = to_train_data(gbdt_valid_data)

gbdt_train_label = to_train_label(gbdt_train_label)



print('gbdt train data: ', gbdt_train_data[0])

print('gbdt train label: ', gbdt_train_label[0])

# 1. finish the class my_gbdt

# It should be noted that,

# (1). you can use function DecisionTreeRegressor in sklearn as a simple regression tree;

# (2). you need to complete the parameters required by gbdt in the class my_gbdt.



from sklearn.tree import DecisionTreeRegressor

import numpy as np

    

class my_gbdt:    

    def __init__(self, iter_n, learning_rate, max_depth):

        self.iter_n = iter_n

        self.learning_rate = learning_rate

        self.max_depth = max_depth

        

        self.trees = []

        for i in range(self.iter_n):

            self.trees.append(DecisionTreeRegressor(min_samples_split = 2,

                                                    max_depth = self.max_depth

                                                    ))

    

    

    def fit(self, X, y):

        # gradient of Softmax Loss

        def cal_gradient(y_pred, y):

#             gradient = list()

#             for i in range(len(y)):

#                 vec = [0] * 2

#                 vec[0] = np.exp(y_pred[i][0])/(np.exp(y_pred[i][0]) + np.exp(y_pred[i][1]))

#                 vec[1] = np.exp(y_pred[i][1])/(np.exp(y_pred[i][0]) + np.exp(y_pred[i][1]))

#                 if y[i][0] == 1:

#                     vec[0] -=1

#                 else:

#                     vec[1] -=1        

#                 gradient.append(vec)

            gradient = np.subtract(np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1), y)

            return gradient

    

        self.trees[0].fit(X, y)

        y_pred = self.trees[0].predict(X)

        for i in range(1, self.iter_n):

            gradient = cal_gradient(y_pred, y)

            self.trees[i].fit(X, gradient)

            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))

    

    

    def predict(self, X):

        y_pred = self.trees[0].predict(X)

        for i in range(1, self.iter_n):

            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))

        # softmax possibility, in fact in this case it has no effect

        # y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)

        return np.argmax(y_pred, axis=1)
my_model = my_gbdt(35, 0.82, 5)

my_model.fit(gbdt_train_data, gbdt_train_label)

pred_label = my_model.predict(gbdt_valid_data)

accuracy = np.sum(np.array(gbdt_valid_label) == pred_label, axis=0) / len(pred_label)

print('gbdt accuracy: ', accuracy)