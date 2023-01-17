import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os, math, collections

from sklearn import cross_validation

from sklearn import tree





datadir = '../input'



N_ROWS = 200000

gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col=None, nrows=N_ROWS)

gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col=None, nrows=N_ROWS)

#phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

# Get rid of duplicate device ids in phone

#phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')



events = pd.read_csv(os.path.join(datadir,'events.csv'), parse_dates=['timestamp'], index_col=None, nrows=N_ROWS)

appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), usecols=['event_id','app_id','is_active'], dtype={'is_active':bool}, nrows=N_ROWS)

applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'), nrows=N_ROWS)





appevents_merged_with_applabels = appevents.merge(applabels, on='app_id', how='outer').dropna()

#print(appevents_merged_with_applabels)



events_merged_with_gatrain = events.merge(gatrain, on='device_id', how='outer').dropna()

#print(events_merged_with_gatrain)



label_for_each_event = events_merged_with_gatrain.merge(appevents_merged_with_applabels, on='event_id', how='outer').dropna()

#print(label_for_each_event)



all_device_ids = pd.unique(label_for_each_event['device_id'].ravel())

#print(all_device_ids)



all_demographic_groups = ['M39+', 'M22-', 'M27-28', 'M23-26', 'M32-38', 'F27-28', 'F29-32', 'M29-31', 'F43+', 'F33-42', 'F24-26', 'F23-']

#pd.unique(label_for_each_event['group'].ravel()).tolist()



print(all_demographic_groups)





all_user_dict = []



for deviceid in all_device_ids:

    #select all events for a single device_id

    all_events_for_single_device_id = label_for_each_event.loc[label_for_each_event['device_id'] == deviceid]

    #tally up the app events for each app label

    event_counts = all_events_for_single_device_id['label_id'].value_counts()

    #using group as a key...

    user_group_for_this_device_id = all_events_for_single_device_id.iloc[0]['group']

    #...add the event count dictionary for this user to a larger dictionary

    tuple_of_group_user_dict = (event_counts.to_dict(), user_group_for_this_device_id )

    all_user_dict += [tuple_of_group_user_dict]



    



#print(all_user_dict)



#new_inputs = []



#for user_group, user_dict in all_user_dict.items():

#    for key, value in user_dict.items():

#        if value < 3:

#            user_dict[key] = 'l'

#        elif value < 6:

#            user_dict[key] = 'm'

#        elif value < 10:

#           user_dict[key] = 'h'

#        else:

#            user_dict[key] = 's'

#    new_inputs += [(user_dict, user_group)]

#print(new_inputs)







#munge data for sklearn



sklearn_X = []



sklearn_Y = []



i=0



for user_dict, user_group in all_user_dict:

    #print(user_dict)

    sklearn_X_temp = []

    for number in np.arange(1021):

        if number in user_dict:

            sklearn_X_temp += [user_dict[number]]

        else:

            sklearn_X_temp += [0]



    #print(sklearn_X_temp)

    sklearn_X += [sklearn_X_temp]

    sklearn_Y += [all_demographic_groups.index(user_group)]

    i+=1    



print(str(i))

#print(sklearn_X)

#print(sklearn_Y)





### Using sklearn decision tree?



#X data are event_counts of each app label for each user

#app labels are features (the keys from the inner dicts)

#Y target data are group_id's for each users list of event counts 



X_train, X_test, y_train, y_test = cross_validation.train_test_split(sklearn_X, sklearn_Y, test_size=0.3, random_state=0)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)



print(clf.score(X_test, y_test))

print(y_test)

print(clf.predict(X_test))



print(y_test-clf.predict(X_test))







### Making decision tree classifier from scratch







def entropy(class_probabilities):

    #given a list of class probabilities compute entropy

    return sum([-p*math.log(p,2) for p in class_probabilities if p])



def class_probabilities(labels):

    total_count = len(labels)

    return [count/total_count for count in collections.Counter(labels).values()]







def data_entropy(labeled_data):

    labels = [label for _, label in labeled_data]

    #print("labels")

    #print(labels)

    probabilities = class_probabilities(labels)

    return entropy(probabilities)







def partition_entropy(subsets):

    #find the entropy from this partition of data into subsets

    #a subset is a list of list of labeled data

    total_count = sum([len(subset) for subset in subsets])

    #print(subsets)

    return sum(data_entropy(subset)*len(subset)/total_count for subset in subsets)





def partition_by(inputs, attribute):

    #each input is a key:value pair {label: {attribute_dict}}

    #returns a dict



    groups = defaultdict(list)

    for input_row in inputs:

        if attribute in input_row[0]:

            key = input_row[0][attribute]

            groups[key].append(input_row)

    return groups







def partition_entropy_by(inputs, attribute):

    #computes entropy corresponding to the given partition

    partitions = partition_by(inputs, attribute)

    return partition_entropy(partitions.values())



    



for label_id in np.arange(1021):

    sample = partition_entropy_by(new_inputs, label_id)

    if sample != 0:

        print(label_id, sample)



      



def classify(tree, input):

    #classify the input using the given decision tree

    ##if tree is leaf node, return value

    if tree in all_demographic_groups:

        return tree

    #otherwise this tree consists of an attribute to split on

    #and a dictionary whose keys are values of that attribute

    #and whose values of are subtrees to consider next

    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)



    if subtree_key not in subtree_dict:

        subtree_key = None



    subtree = subtree_dict[subtree_key]

    return classify(subtree, input)







#def build_tree_id3(inputs, split_candidates=None):