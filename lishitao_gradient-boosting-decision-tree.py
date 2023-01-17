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

            feats.append(feat)

            j = i+1

    return feats + [line[j:]]



# load a csv file, use parse_feat() to convert format

def load_file(file_name):

    data = []

    with open(file_name, 'r') as fin:

        print('field_names:', fin.readline().strip().split(','))

        for line in fin:

            line = line.strip()

            data.append(parse_feat(line))

    return np.array(data)



train_data = load_file('../input/train.csv')

test_data = load_file('../input/test.csv')



train_id, train_label, train_feat = train_data[:, 0], train_data[:, 1], train_data[:, 2:]

test_id, test_feat = test_data[:, 0], test_data[:, 1:]



train_feat[:, [1, 6]] = None

test_feat[:, [1, 6]] = None



print('train_feat:\n', train_feat[0])

print('test_feat:\n', test_feat[0])
def get_feat_name(field, feat_val):

    assert field != 'text'

    if FEAT_TYPE[field] == 'cat':

        return str(field) + ':' + feat_val

    elif FEAT_TYPE[field] == 'num':

        return str(field) + ':'

    elif FEAT_TYPE[field] == 'set':

        return [str(field) + ':' + fv for  fv in feat_val.split()]

    

def build_feat_map(data):

    feat_map = {}

    for i in range(len(FEAT_TYPE)):

        if FEAT_TYPE[i] == 'num':

            fn = get_feat_name(i, None)

            if fn not in feat_map:

                feat_map[fn] = len(feat_map)

            continue

        elif FEAT_TYPE[i] == 'text':

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

    

feat_map = build_feat_map(np.vstack([train_feat, test_feat]))

print(feat_map)

print(np.array(feat_map.keys()))

print(np.array(feat_map.values()))
def to_float(x):

    if len(x):

        return float(x)

    return -1



def get_feat_id_val(field, feat_val):

    assert field != 'text'

    feat_name = get_feat_name(field, feat_val)

    if FEAT_TYPE[field] == 'cat':

        return feat_map[feat_name], 1

    elif FEAT_TYPE[field] == 'num':

        return feat_map[feat_name], to_float(feat_val)

    elif FEAT_TYPE[field] == 'set':

        return [feat_map[fn] for fn in feat_name], [1] * len(feat_name)



def to_libsvm(data):

    libsvm_data = []

    for d in data:

        libsvm_data.append([])

        for i in range(len(FEAT_TYPE)):

            if FEAT_TYPE[i] == 'cat' or FEAT_TYPE[i] == 'num':

                fv = get_feat_id_val(i, d[i])

                libsvm_data[-1].append(fv)

            elif FEAT_TYPE[i] == 'set':

                fvs = get_feat_id_val(i, d[i])

                for fv in zip(*fvs):

                    libsvm_data[-1].append(fv)

    return libsvm_data



train_data = to_libsvm(train_feat)

test_data = to_libsvm(test_feat)



size = len(train_data)

train_ind = np.arange(size)[: int(0.8 * size)]

valid_ind = np.arange(size)[int(0.8 * size): ]



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

    'nthread': 1, 

    'silent': 1, 

    # tree params

    'eta': 1, 

    'gamma': 0,

    'max_depth': 4, 

    'subsample': 1,

    'lambda': 1,

    'alpha': 0,

    # learning params

    'objective': 'binary:logistic', 

    'eval_metric': 'error', }

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

    'num_threads': 1,

    'learning_rate': 1,

    'num_leaves': 31, 

    'max_depth': 9,

    'metric': 'binary_error',

    'lambda_l1': 0,

    'lambda_l2': 0,

    }



num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dtest], early_stopping_rounds=5)
gbdt_train_data, gbdt_valid_data = train_data[: int(0.8 * len(train_data))], train_data[int(0.8 * len(train_data)): ]

gbdt_train_label = list(map(int, train_label))

gbdt_train_label, gbdt_valid_label = gbdt_train_label[: int(0.8 * len(gbdt_train_label))], gbdt_train_label[int(0.8 * len(gbdt_train_label)): ]



def to_train_data(input):

    output = list()

    for sample in input:

        sample_vec = [0] * len(feat_map)

        for feature in sample:

            sample_vec[feature[0]] = feature[1]

        output.append(sample_vec)

    return output



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

import progressbar

from sklearn.tree import DecisionTreeRegressor

bar_widgets = [

    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),

    ' ', progressbar.ETA()

]

class SquareLoss():

    def __init__(self): pass



    def loss(self, y, y_pred):

        return 0.5 * np.power((y - y_pred), 2)



    def gradient(self, y, y_pred):

        return -(y - y_pred)

class my_gbdt(object):

    

    def __init__(self):

        # TO DO

        self.learning_rate = 0.1

        self.min_samples_split = 2

        self.min_impurity = 1e-7

        self.max_depth = 4

        self.regression = False

        self.n_estimators=200

        self.loss = SquareLoss()

        

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        self.trees = []

        for i in range(self.n_estimators):

            self.trees.append(DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth))

        #self.tree = DecisionTreeRegressor(min_samples_split=2, max_depth=2)

    

    def fit(self, X, y):

        # TO DO

        self.trees[0].fit(X, y)

        y_pred = self.trees[0].predict(X)

        for i in self.bar(range(1, self.n_estimators)):

            gradient = self.loss.gradient(y, y_pred)

            self.trees[i].fit(X, gradient)

            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))

    

    def predict(self, X):

        # TO DO

        y_pred = self.trees[0].predict(X)

        for i in range(1, self.n_estimators):

            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))

 

        if not self.regression:

            # Turn into probability distribution

            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)

        # Set label to the value that maximizes probability

            y_pred = np.argmax(y_pred, axis=1)

        return y_pred
my_model = my_gbdt()

my_model.fit(gbdt_train_data, gbdt_train_label)

pred_label = my_model.predict(gbdt_valid_data)

accuracy = np.sum(np.array(gbdt_valid_label) == pred_label, axis=0) / len(pred_label)

print('gbdt accuracy: ', accuracy)