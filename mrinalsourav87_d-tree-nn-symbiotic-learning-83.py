import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import numpy as np 
from math import log
import json 
import time
from random import randint

# keras imports for neural network
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import optimizers
kick_start_2016 = pd.read_csv('../input/ks-projects-201612.csv', encoding = 'ISO-8859-1')
kick_start_2016.head()
kick_start_2018 = pd.read_csv('../input/ks-projects-201801.csv')
kick_start_2018.head()
print(len(kick_start_2018) - len(kick_start_2016))
kick_start_2016.columns
kick_start_2018.columns
# renaming the 2016 data columns
kick_start_2016.columns = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline',
       'goal', 'launched', 'pledged', 'state', 'backers', 'country',
       'usd_pledged', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',
       'Unnamed: 16']
kick_start_common = kick_start_2018.loc[kick_start_2018.ID.isin(kick_start_2016.ID)].reset_index(drop=True)
len(kick_start_common)
kick_start_common.head()
kick_start_unique = kick_start_2018.loc[-kick_start_2018.ID.isin(kick_start_2016.ID)].reset_index(drop=True)
len(kick_start_unique)
kick_start_unique.head()
kick_start = kick_start_common
kick_start["state"].unique()
pd.value_counts(kick_start['state']).plot.pie()
x = pd.to_datetime(kick_start["deadline"])
y = pd.to_datetime(kick_start["launched"])
z = x - y  

print (z.min())
print (z.mean())
print (z.max())
num_days = z.apply(lambda x: str(x.days))
pd.value_counts(num_days).plot.pie()

dt_kstart = kick_start[['category', 'main_category', 'currency', 'country', 'state']]
dt_kstart['num_days'] = num_days
dt_kstart.state.value_counts()
# this function calculates the probability distribution of the unique items for a specified column in a dataframe
def get_probabilities(df, column):
    freqs = df[column].value_counts()
    summation = sum(freqs)
    probabilities = freqs / summation
    return (probabilities) 

# function to return log base 2
def ln(x):
    return log(x)/log(2)

# to return the entropy given the probability distribution    
def get_entropy(probabilities):
    return sum(probabilities * probabilities.apply(ln)) * -1       # Claude Elwood Shannon
# Initial entropy of the 'state' column in the training data
get_entropy(get_probabilities(dt_kstart, 'state'))
# splitting a data-frame, on an index/column and value 
# returns the new dataframe after the split
def split_data(df, column, value):
    return df[df[column] == value]

# to get the best feature based on information gain 
def get_best_feature(df, target, to_print=False):
    initial_entropy = get_entropy(get_probabilities(df, target))
    best_gain = 0.0
    best_feature = None
    feature_list = list(df.columns)
    feature_list.remove(target)
    for feature in feature_list:
        uniques = df[feature].unique()
        new_entropy = 0 
        for value in uniques:
            subset =  split_data (df, feature, value) 
            probability = len(subset) / len(df)
            new_entropy += probability * get_entropy(get_probabilities(subset, target))
        info_gain = initial_entropy - new_entropy
        if to_print:
            print (info_gain, feature)
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = feature
            
    return best_feature
get_best_feature(dt_kstart, 'state', True)
# returns true if there is only one label in the target field 
def is_pure(df, target):
    return len(df[target].unique()) == 1
        
def create_tree(df, target):
    # condition for pure data (when there is only one possible 'state')
    if is_pure(df, target):
        return {'state' : dict(get_probabilities(df, target))}
    
    # condition for leaf nodes
    if len(df.columns) <= 2:
        features = list(df.columns)
        features.remove(target)
        feature = features[0]
        leaf_node = {feature:{}}
        uniques = df[feature].unique()
        for value in uniques:
            subset = split_data(df, feature, value)
            leaf_node[feature][value] = {'state' : dict(get_probabilities(subset, target))}
        return leaf_node
    
    # recursive call to create the nested tree/dictionary
    best_feature = get_best_feature(df, target)
    if best_feature:
        my_tree = {best_feature:{}}
        uniques = df[best_feature].unique()
        for value in uniques:
            subset = split_data(df, best_feature, value)
            subset = subset.drop(best_feature, axis=1)
            my_tree[best_feature][value] = create_tree(subset, target)
    else: 
        my_tree = {'state' : dict(get_probabilities(df, target))}
        
    return my_tree
            
# start time
start = time.perf_counter()

# creating the tree
d_tree = create_tree(dt_kstart, 'state')

# saving the dictionary
filename = '/kaggle/working/decision_tree.txt'
with open(filename, 'w') as f:
    json.dump(d_tree, f)
    
# end time
stop = time.perf_counter()

print('Creating the Decision Tree took close to ' + str((stop-start)/60.0) + ' minutes')
# loading the tree
def load_tree(filename):
    with open(filename, 'r') as f:
        return json.load(f)

d_tree = load_tree('/kaggle/working/decision_tree.txt')
# to predict a single instance of a feature using the decision tree
# inputs: the tree; an instance of features of type pandas.Series.series
# returns: the probability distribution of the states as a dictionary
def partial_predict(tree, features):
    probabs = {}
    first_dict = next(iter(tree))
    second_dict = tree[first_dict]
    feat_value = features[first_dict]
    if first_dict != 'state':
        for key in second_dict.keys():
            if feat_value == key:
                probabs = partial_predict(second_dict[key], features)
    else: 
        probabs = second_dict
    return probabs
labels = dict(dt_kstart.loc[89])
labels
partial_predict(d_tree, labels)
# retrieving data from the tree
d_tree['category']['Restaurants']['num_days']['59']['country']['US']
# to translate a predicted distribution (a dictionary) to its corresponding numpy version
def translate(distribution):
    array = np.zeros([6])
    # to hardcode positions in the numpy array 
    positions = {'failed':0, 
                 'successful':1, 
                 'canceled':2, 
                 'undefined':3, 
                 'live':4, 
                 'suspended':5} 
    for key in positions:
        if key in distribution.keys():
            array[positions[key]] = distribution[key]
    return array
x = partial_predict(d_tree, labels) #same example as above
print(x)
y = translate(x)
print (y) # this is now translated to a numpy version
# since this is a probability distribution, all entries must sum to 1 
y.sum()
to_predict = dt_kstart
# to get predictions for an entire dataframe
def get_partial_predictions(tree, inputs):
    partial_predictions = []
    for index,row in inputs.iterrows():
        features = dict(row)
        probabs = partial_predict(tree, features)
        arr = translate(probabs)
        partial_predictions.append(arr)
    return np.array(partial_predictions)

start = time.perf_counter()
part_predict = get_partial_predictions(d_tree, to_predict)
stop = time.perf_counter()

print(part_predict.shape)
print('This process took ' + str(stop-start) + ' seconds')

nn_train_part1 = part_predict
nn_train_part2 = np.array(kick_start[['backers', 'usd_pledged_real', 'usd_goal_real']])

print (nn_train_part1.shape)
print (nn_train_part2.shape)
nn_inputs = np.concatenate((nn_train_part1, nn_train_part2), axis=1)
print(nn_inputs.shape)
# save the training_inputs for the neural network
np.save('/kaggle/working/nn_inputs', nn_inputs)
# for the targets of the neural network
states = np.array(kick_start['state'])
states.shape
# to translate the state into integers for one-hot encoding
def translate_states(states):
    array = np.empty([len(states)], dtype = 'int8')
    positions = {'failed':0, 
                 'successful':1, 
                 'canceled':2, 
                 'undefined':3, 
                 'live':4, 
                 'suspended':5} 
    for i, state in enumerate(list(states)):
        array[i] = int(positions[state])
    return array
translated = translate_states(states)
translated[:10]
# One hot encoding
nb_classes = 6
one_hot_targets = np.eye(nb_classes)[translated]
one_hot_targets[:10]
# saving the targets for the neural network
np.save('/kaggle/working/one_hot_targets', one_hot_targets)
inputs = np.load('/kaggle/working/nn_inputs.npy')
targets = np.load('/kaggle/working/one_hot_targets.npy')
model = Sequential()
model.add(Dense(9, input_dim=9, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(6, activation = 'softmax')) 

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.fit(nn_inputs, one_hot_targets, validation_split = 0.1, epochs = 6, batch_size=120)
model.save('/kaggle/working/nn_model.h5')
# helper function to convert dates to num_days
def convert_dates(features):
    x = pd.to_datetime(features["deadline"])
    y = pd.to_datetime(features["launched"])
    z = x - y 
    num_days = str(z.days)
    return num_days
    
# for final predictions using both the decision tree and the neural network 
# inputs: a pandas.Series.series object called features, and,
#         the trained decision tree and neural network
# outputs: the predicted 'state' for the provided features as a numpy array 
def predict(features, d_tree, model):
    expected_out = one_hot = None
    reverse_hot = {0:'failed', 
                   1:'successful', 
                   2:'canceled', 
                   3:'undefined', 
                   4:'live', 
                   5:'suspended'}
    num_days = convert_dates(features)
    features = dict(features)
    features['num_days'] = num_days
    part_preds = partial_predict(d_tree, features)
    part1 = translate(part_preds)
    part2 = np.array([features['backers'], 
                      features['usd_pledged_real'], 
                      features['usd_goal_real']])
    to_predict = np.concatenate((part1, part2))
    to_predict = np.array([to_predict])
    predicted_numpy = model.predict(to_predict)
    prediction = np.array(reverse_hot[predicted_numpy.argmax()])
    return prediction
model = load_model('/kaggle/working/nn_model.h5')
d_tree = load_tree('/kaggle/working/decision_tree.txt')
kick_start_test = kick_start.drop(['state'], axis=1)
kick_start_test['state'] = None
expected_outputs = kick_start['state']
start = time.perf_counter()

results = {'predicted':[], 'expected': list(expected_outputs)}
for index,row in kick_start_test.iterrows():
    results['predicted'].append(predict(row, d_tree, model))  

end = time.perf_counter()
print ('Getting predictions on the training dataset took ' + 
       str((end - start) / 60.0) + ' minutes')
results_df = pd.DataFrame(results)
results_df.to_csv('/kaggle/working/results_training_data.csv')
def display_results_data(result_df):
    matches = result_df.loc[(result_df['predicted'] == result_df['expected'])]
    match_percentage = len(matches)/len(result_df) * 100
    errors =  result_df.loc[(result_df['predicted'] != result_df['expected'])]
    error_percentage = len(errors)/len(result_df) * 100
    
    print ('\nTrue Positives = ' + str(len(matches)) + 
           '\t\t' + 'True Pos. Percentage = ' + 
           str(match_percentage))
    print ('\nErrors = ' + str(len(errors)) +
           '\t\t\t' + 'Error Percentage = ' + 
           str(error_percentage))

display_results_data(results_df)

model = load_model('/kaggle/working/nn_model.h5')
d_tree = load_tree('/kaggle/working/decision_tree.txt')
kick_start_unique.head()
expected_outputs = kick_start_unique['state']
test_data = kick_start_unique.drop(['state'], axis=1)
test_data['state'] = None
test_data.head()
start = time.perf_counter()

results = {'predicted':[], 'expected': list(expected_outputs)}
for index,row in test_data.iterrows():
    results['predicted'].append(predict(row, d_tree, model))  

end = time.perf_counter()
print ('Getting predictions on the Test dataset took ' + 
       str(end - start) + ' seconds')
results_df = pd.DataFrame(results)
results_df.to_csv('/kaggle/working/unseen_test_predictions.csv')
display_results_data(results_df)
predicted_np = np.array(results['predicted'])
expected_np = np.array(results['expected'])
df_confusion = pd.crosstab(expected_np, 
                           predicted_np, 
                           rownames=['Actuals'], 
                           colnames=['Predicted'], 
                           margins=True)
df_confusion
