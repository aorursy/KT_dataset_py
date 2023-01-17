import pandas as pd

import numpy as np

import random, math

import multiprocessing

from tqdm import tqdm

import numba

tqdm.pandas()



xtrain = pd.read_csv('../input/X_train.csv')

ytrain = pd.read_csv('../input/y_train.csv')

train = pd.merge(xtrain, ytrain, how='left', on='series_id')



xtest = pd.read_csv('../input/X_test.csv')

ytest = pd.read_csv('../input/sample_submission.csv')

test = pd.merge(xtest, ytest, how='left', on='series_id')

print(train.shape, test.shape)
def features(df):

    for c in ['angular_velocity_', 'linear_acceleration_']:

        col = [c + c1 for c1 in ['X','Y','Z']]

        for agg in ['min(', 'max(', 'sum(', 'mean(', 'std(', 'skew(', 'kurtosis(', 'quantile(.25,', 'quantile(.5,', 'quantile(.75,']:

            df[c+agg] = eval('df[col].' + agg + 'axis=1)')

            df[c+'a'+agg] = eval('df[col].abs().' + agg + 'axis=1)')

    return df



train = features(train).fillna(0)

test = features(test).fillna(0)

print(train.shape, test.shape)
col = [c for c in train.columns if c not in ['row_id', 'series_id', 'measurement_number', 'group_id', 'surface']] + ['surface']
#Inspiration from http://searene.me/2017/12/23/Write-Machine-Learning-Algorithms-From-Scratch-Random-Forest/

#@numba.jitclass()

class tree_node:

    def __init__(self, data):

        self.data = data

        self.left = None

        self.right = None

        self.category = None

        self.split_point = (None, None)

        self.split_value = None



def get_most_common_category(data):

    gdata = data[data.columns[-1]].value_counts()

    return gdata.index[0]



def get_categories(data):

    return data[data.columns[-1]].unique()



def get_gini(left, right, categories): #Try different metrics here

    left = left[left.columns[-1]].value_counts()

    right = right[right.columns[-1]].value_counts()

    gini = 0

    for group in left, right:

        if len(group) == 0:

            continue

        score = 0

        for category in categories:

            if category in group.index:

                p = group[category] / sum(group)

            else:

                p = 0

            score += p * p

        gini += (1 - score) * (sum(group) / (sum(left) + sum(right)))

    return gini



def split(data, x, y):

    split_value = data.iloc[x][y]

    left = data[data[y]<=split_value]

    right = data[data[y]>split_value]

    return left, right



def get_split_point(data, split_search, split_min_gini): #Add histogram option

    features = list(data.columns[:-1])

    categories = get_categories(data)

    x, y, gini = None, None, None

    for i_ in range(split_search):

        feature = random.choice(features) 

        i = random.choice(range(len(data)))

        left, right = split(data, i, feature)

        current_gini = get_gini(left, right, categories)

        if gini is None or current_gini < gini:

            x, y, gini = i, feature, current_gini

        if gini <= split_min_gini:

            break

    return x, y



def build_tree(data, depth, max_depth, min_size, n_sample_rate, split_search, split_min_gini):

    if depth==1: data = data.sample(frac=n_sample_rate, random_state=None).copy()

    root = tree_node(data.copy())

    x, y = get_split_point(data, split_search, split_min_gini)

    left_branch, right_branch = split(data, x, y)

    root.split_point = (x, y)

    root.split_value = root.data.iloc[x][y]

    if len(left_branch) == 0 or len(right_branch) == 0 or depth >= max_depth:

        root.category = get_most_common_category(pd.concat((left_branch, right_branch)))

        #if depth < max_depth:print(depth, root.category)

    else:

        if len(left_branch) < min_size:

            root.left = tree_node(left_branch)

            root.left.category = get_most_common_category(left_branch)

        else:

            root.left = build_tree(left_branch, depth + 1, max_depth, min_size, n_sample_rate, split_search, split_min_gini)



        if len(right_branch) < min_size:

            root.right = tree_node(right_branch)

            root.right.category = get_most_common_category(right_branch)

        else:

            root.right = build_tree(right_branch, depth + 1, max_depth, min_size, n_sample_rate, split_search, split_min_gini)

    root.data = None #clean up

    if depth == 1: print('tree created...')

    return root



def RandomForest(df, n_trees, max_depth, min_size, n_sample_rate, split_search, split_min_gini):

    #TO DO: GPU enable and add numba if possible

    p = multiprocessing.Pool(multiprocessing.cpu_count())

    trees = [[df, 1, max_depth, min_size, n_sample_rate, split_search, split_min_gini] for i in range(n_trees)]

    trees=p.starmap(build_tree, trees)

    p.close(); p.join();

    return trees



model = RandomForest(df=train[col], n_trees=14, max_depth=10, min_size=2, n_sample_rate=0.4, split_search=10, split_min_gini=0.0001)
def print_tree(tree, l, feature='', comment=''):

    i, feature = tree.split_point

    if tree.category != None:

        print('\t'*l, l, comment, tree.category, feature, tree.split_value)

    l += 1

    try: print_tree(tree.left, l, comment='left')

    except: pass

    try: print_tree(tree.right, l, comment='right')

    except: pass



for tree in model:

    print_tree(tree,0,comment='root')

    break
#@numba.jit()

def predict_tree(tree, r):

    if tree.category is not None:

        return tree.category

    i, feature = tree.split_point

    split_value = tree.split_value

    if r[feature] <= split_value:

        return predict_tree(tree.left, r)

    else:

        return predict_tree(tree.right, r)



#@numba.jit()

def predict(trees, r):

    prediction = []

    for tree in trees:

        prediction.append(predict_tree(tree, r))

    return max(set(prediction), key=prediction.count)



def mpredict_apply(df, model, col, target_name):

    df = pd.DataFrame(df)

    df[target_name] = df[col].progress_apply(lambda r: predict(model, r), axis=1).values

    return df



def mpredict(df, model, col, target_name, split):

    p = multiprocessing.Pool(multiprocessing.cpu_count())

    df = np.array_split(df, split)

    df = [[df_, model, col, target_name] for df_ in df]

    df = p.starmap(mpredict_apply, df)

    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)

    p.close(); p.join()

    return df



val = train.tail(87680)

val = mpredict(val, model, col, 'surface2', split=4)

print('Accuracy Score: ', np.mean(val['surface'] ==val['surface2']))
test = mpredict(test, model, col, 'surface', split=4)

sub = test.groupby(by=['series_id', 'surface'], as_index=False)['row_id'].count()

sub = sub.sort_values(by=['series_id', 'surface', 'row_id'], ascending=[True, True, False]).reset_index(drop=True)

sub.drop_duplicates(subset=['series_id'], keep='first', inplace=True)

sub[['series_id', 'surface']].to_csv('submission.csv', index=False)