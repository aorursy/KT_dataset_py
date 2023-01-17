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



from sklearn.tree import DecisionTreeRegressor

class my_gbdt(object):

    def __init__(self, n_estimators=10, min_samples_split=2, max_depth=2, learning_rate=0.5):

        self.n_estimators = n_estimators

        self.learning_rate = learning_rate

        self.min_samples_split = min_samples_split

        self.max_depth = max_depth

        self.trees1 = []

        self.trees2 = []

        for i in range(self.n_estimators):

            self.trees1.append(

                DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth))

            self.trees2.append(

                DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth))



    def fit(self, X, y):



        y1 = [[y[i][0]] for i in range(len(y))]

        y1 = np.array(y1)

        y2 = [[y[i][1]] for i in range(len(y))]

        y2 = np.array(y2)

        F1 = np.zeros((len(y), 1))

        F2 = np.zeros((len(y), 1))

        for i in range(self.n_estimators):

            p1 = np.exp(F1) / (np.exp(F1) + np.exp(F2))

            p2 = np.exp(F2) / (np.exp(F1) + np.exp(F2))

            gradient1 = y1 - p1

            gradient2 = y2 - p2

            self.trees1[i].fit(X, gradient1)

            self.trees2[i].fit(X, gradient2)

            pre1 = []

            x1 = []

            g1 = []

            y_pre1 = self.trees1[i].predict(X).reshape(len(X), 1)

            for j in range(len(X)):

                if y_pre1[j][0] not in pre1:

                    pre1.append(y_pre1[j][0])

                    x1.append([X[j]])

                    g1.append([gradient1[j]])

                else:

                    for k in range(len(pre1)):

                        if pre1[k] == y_pre1[j][0]:

                            x1[k].append(X[j])

                            g1[k].append(gradient1[j])

            f1 = []

            for k in g1:

                a = 0

                b = 0

                for l in k:

                    a += l

                    b += abs(l) * (1 - abs(l))

                f1.append(0.5 * a / b)

            for k in range(F1.shape[0]):

                for z in range(len(x1)):

                    if X[k] in x1[z]:

                        F1[k][0] += self.learning_rate * f1[z]



            pre2 = []

            x2 = []

            g2 = []

            y_pre2 = self.trees2[i].predict(X).reshape((len(X), 1))

            for j in range(len(X)):

                if y_pre2[j][0] not in pre2:

                    pre2.append(y_pre2[j][0])

                    x2.append([X[j]])

                    g2.append([gradient2[j]])

                else:

                    for k in range(len(pre2)):

                        if pre2[k] == y_pre2[j][0]:

                            x2[k].append(X[j])

                            g2[k].append(gradient2[j])

            f2 = []

            for k in g2:

                a = 0

                b = 0

                for l in k:

                    a += l

                    b += abs(l) * (1 - abs(l))

                f2.append(0.5 * a / b)

            for k in range(F2.shape[0]):

                for z in range(len(x2)):

                    if X[k] in x2[z]:

                        F2[k][0] += self.learning_rate * f2[z]



    def predict(self, X):

        y_pred1 = np.zeros((len(X), 1))

        y_pred2 = np.zeros((len(X), 1))

        for i in range(self.n_estimators):

            y_pred1 += self.trees1[i].predict(X).reshape((len(X), 1))

            y_pred2 += self.trees2[i].predict(X).reshape((len(X), 1))

        y_pred = [[y_pred1[i][0], y_pred2[i][0]] for i in range(y_pred1.shape[0])]

        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)

        y_pred = np.argmax(y_pred, axis=1)

        return y_pred







my_model = my_gbdt(n_estimators=16, max_depth=5, learning_rate=0.2)

my_model.fit(gbdt_train_data, gbdt_train_label)

pred_label = my_model.predict(gbdt_valid_data)

accuracy = np.sum(np.array(gbdt_valid_label) == pred_label, axis=0) / len(pred_label)

print('gbdt accuracy: ', accuracy)