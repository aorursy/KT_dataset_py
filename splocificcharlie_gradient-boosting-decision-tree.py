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

    def __init__(self,num_round,lr,early_stop):

        # TO DO

        self.tree = DecisionTreeRegressor(min_samples_split=2, max_depth=2)

        self.lr = lr

        self.early_stop=early_stop

        self.num_round = num_round

    

    def fit(self, X, y, Val):

        # TO DO

        xx,yy=Val

        yy=np.array(yy)

        def negative_gradient(y, pred):

            return y - pred

        self.y0 = np.array(y)[:,1]

        yp = np.array([np.mean(self.y0)]*self.y0.shape[0])

        r  = negative_gradient(self.y0,yp)

        self.trees = []

        scores= []

        counter=0

        self.idx=0

        for i in range(num_round):

            if counter==self.early_stop:break

            tree = DecisionTreeRegressor(

                    criterion='friedman_mse',

                    splitter='best',

                    min_samples_split=2, 

                    max_depth=4)

            tree.fit(X,r)

            rp = tree.predict(X)

            yp = yp + self.lr*rp

            val_pred = self.predict(xx)

            val_score=np.sum(yy == val_pred, axis=0) / len(yy)

            scores.append(val_score)

            print('[%d]'%i,val_score)

            if val_score<max(scores[-self.early_stop-1:]):

                counter+=1

            else:

                self.idx=i

                counter=0

            r = negative_gradient(self.y0,yp)

            self.trees.append(tree)

        self.trees=self.trees[:self.idx]

    def predict(self, X):

        # TO DO

        #pred_proba = self.tree.predict(X)

        pred = np.array([np.mean(self.y0)]*len(X),dtype=np.float64)

        for tree in self.trees:

            pred+=self.lr*tree.predict(X)

        pred[pred>0.5]=1

        pred[pred!=1]=0

        return pred
my_model = my_gbdt(100,1,5)

my_model.fit(gbdt_train_data, gbdt_train_label,[gbdt_valid_data,gbdt_valid_label])

pred_label = my_model.predict(gbdt_valid_data)

accuracy = np.sum(np.array(gbdt_valid_label) == pred_label, axis=0) / len(pred_label)

print('gbdt accuracy at [%d]: '%my_model.idx, accuracy)