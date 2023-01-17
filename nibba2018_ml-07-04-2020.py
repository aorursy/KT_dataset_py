# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
petrol_data = pd.read_csv("/kaggle/input/petrol-consumption/petrol_consumption.csv")

petrol_data.head()
data_reg_y = petrol_data['Petrol_Consumption']

data_reg_X = petrol_data.drop(['Petrol_Consumption'], axis=1)
bill_data = pd.read_csv("/kaggle/input/bill_authentication/bill_authentication.csv")

bill_data.head()
data_class_y = bill_data['Class']

data_class_X = bill_data.drop(['Class'], axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
train_class_X, test_class_X, train_class_y, test_class_y = train_test_split(data_class_X, data_class_y, shuffle = True)
random_class = RandomForestClassifier()

random_class.fit(train_class_X, train_class_y)

random_class.score(test_class_X, test_class_y)
from sklearn.ensemble import RandomForestRegressor
train_reg_X, test_reg_X, train_reg_y, test_reg_y = train_test_split(data_reg_X, data_reg_y, shuffle = True)
random_reg = RandomForestRegressor()

random_reg.fit(train_reg_X, train_reg_y)

random_reg.score(test_reg_X, test_reg_y)
def get_potential_splits(data,random_subspace):

    

    potential_splits = {}

    column_indices = list(range(data.shape[1]-1))

    

    

    if random_subspace and random_subspace < data.shape[1]:

        column_indices = random.sample(population = column_indices,k = random_subspace)

    

    for column_index in  column_indices :

            

            values =data[:,column_index] 

            

            if FEATURE_TYPES[column_index] == 'Continious':

                

                unique_values = np.unique(values)

                potential_splits[column_index] = []

                

                for i in range(len(unique_values)-1):

                    current_value = unique_values[i]

                    next_value = unique_values[i+1]

                    potential_split = (current_value+next_value)/2

                

                    potential_splits[column_index].append(potential_split)

            

            else:

                potential_splits[column_index]=list(set(values))

             

            

    return potential_splits
def determine_type_of_feature(data):

    

    feature_types = []

    threshold = 15

    

    for column_index in range(data.shape[1]-1):

        

        unique_values = np.unique(data[:,column_index])

            

        if(len(unique_values)<=threshold)or isinstance(unique_values[0],str):

            feature_types.append('Categorical')

        else:

            feature_types.append('Continious')

            

    return feature_types
def split_data(data,split_column,split_value):

    

    values = data[:,split_column]

    type_of_feature = FEATURE_TYPES[split_column] 

    

    if type_of_feature == 'Continious':

        data_above = data[values > split_value]

        data_below = data[values <= split_value]

    else:

        data_below = data[values == split_value]

        data_above = data[values != split_value]

        

    return data_below,data_above
def gini(data):

    

    label_column= data[:,-1]

    _,counts = np.unique(label_column,return_counts=True)

    

    p=counts/counts.sum()

    gini =1- np.dot(p,p)

    

    return gini
def entropy(data):

    

    label_columns = data[:,-1]

    _,counts = np.unique(label_columns,return_counts= True)

    

    p = counts/counts.sum()

    entropy = sum(p*-np.log2(p))

    

    

    return entropy
def overall_metric(data_below,data_above,metric_function):

    

    n=len(data_above)+len(data_below)

    p_data_below = len(data_below)/n

    p_data_above = len(data_above)/n

    

    overall_metric = p_data_above*metric_function(data_above) + p_data_below*metric_function(data_below)

    

    return overall_metric
def get_best_split(data, potential_splits, metric_function = gini):

    

    first_iteration = True

    for column_index in potential_splits:

        for value in potential_splits[column_index]:

            

            data_below,data_above = split_data(data,split_column=column_index,split_value = value)

            current_metric = overall_metric(data_above,data_below,metric_function)

            

            if first_iteration:

                

                best_metric = current_metric

                first_iteration = False

            

            if current_metric <= best_metric :

                

                best_metric = current_metric

                best_column =column_index

                best_value = value

                

                

    return best_column,best_value
def  check_purity(data):

    label_columns = data[:,-1]

    

    if len(np.unique(label_columns))==1:

        return True

    else:

        return False
def create_leaf(data):

    

    label_columns = data[:,-1]

    unique_labels,counts = np.unique(label_columns,return_counts =True)

    

    index = counts.argmax()

    leaf = unique_labels[index]

    

    return leaf
def bootstrap(data,n_bootstrap):

    

    indices =np.random.randint(low=0,high=len(data),size=n_bootstrap)

    

    return data[indices]

def decision_tree_algorithm(data,counter =0, max_depth =5,min_samples = 10,random_subspace=None,metric_function = gini):

    

    if counter == 0:

    

        global FEATURE_TYPES

        FEATURE_TYPES = determine_type_of_feature(data)

    

    

    if (check_purity(data)) or (counter == max_depth) or (len(data) < min_samples):

        return create_leaf(data)

    

    else:

        

        counter += 1

        

        potential_splits = get_potential_splits(data, random_subspace)

        column_index,split_value = get_best_split(data, potential_splits, metric_function)

        data_below,data_above = split_data(data, column_index, split_value)

         

        if len(data_below)==0 or len(data_above)==0 :

            return create_leaf(data)

        

        

        type_of_feature = FEATURE_TYPES[column_index]

        #column_name = COLUMN_NAMES[column_index]

        

        if type_of_feature == 'Continious':

            question = "{} <= {}".format(column_index,split_value)

        else:

            question ="{} = {}".format(column_index,split_value)

        sub_tree={question:[]}

        

        yes_answer = decision_tree_algorithm(data_below, counter, max_depth, min_samples,random_subspace  ,metric_function )

        no_answer = decision_tree_algorithm(data_above, counter, max_depth, min_samples,random_subspace  ,metric_function )

        

        if yes_answer == no_answer:

            sub_tree =yes_answer

        else:

            sub_tree[question].append(yes_answer)

            sub_tree[question].append(no_answer)

       

        return sub_tree
def decision_tree_classifer(example,tree):

    question = list(tree.keys())[0]

    column_index,comparison_operator,value =question.split()

    column_index =int(column_index)

    

    if comparison_operator == "<=":

        if example[column_index] <= float(value):

            answer = tree[question][0]

        else:

            answer = tree[question][1]

    

    

    else:

        if str(example[column_index]) == value:

            answer = tree[question][0]

        else:

            answer = tree[question][1]



    if not isinstance(answer,dict):

        return answer

    

    else:

        residual_tree = answer

        return decision_tree_classifer(example, residual_tree)
def random_forest_algorithm(train_data, n_trees,max_depth = 5,min_samples =10,random_state = 123, n_features = 3, n_bootstrap=50,metric_function =gini):

    

    np.random.seed(random_state)

    forest = []

    for i in range(n_trees):

        

        bootstrapped_data = bootstrap(train_data,n_bootstrap)

        tree = decision_tree_algorithm(data = bootstrapped_data, counter=0, random_subspace = n_features, max_depth = max_depth,metric_function=metric_function)

        forest.append(tree)

        

    return forest
def random_tree_classifier(example,forest):

    

    results =[]

    for index in range(len(forest)):

        

        result = decision_tree_classifer(example, forest[index] )

        results.append(result)

        

    mode = max(set(results),key=results.count)

    return mode
def classify_data(test_df,forest):

    

    Predictions = test_df.apply(func = random_tree_classifier, axis = 1, raw=True,args=(forest,))

    

    return Predictions

def calculate_accuracy(labels,predictions):

        

 

    accuracy = np.array(labels == predictions).mean()

    

    return accuracy
train_df = pd.concat([train_class_X, train_class_y], axis=1)

train_df.head()
test_df = pd.concat([test_class_X, test_class_y], axis=1)

test_df.head()
forest=random_forest_algorithm(train_df.values,n_trees = 5,n_features=2,n_bootstrap=100,random_state =120)
predictions = classify_data(test_df.iloc[:,:-1],forest)
labels = test_df.iloc[:,-1]



print("Accuracy is : {}".format(calculate_accuracy(predictions,labels)*100))