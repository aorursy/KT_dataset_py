# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from collections import Counter

import collections

import pandas as pd



def traces2features(inputfile, outputfile, maxsec):

    features = ["hotkey"+str(i)+str(j) for i in range (0,9) for j in range(0,9)] + ['s', 'Base', 'SingleMineral'] 

    data = []

    print (features)

    

    races = []

    urls = []

    with open(inputfile, "r") as traces:

        for trace in traces:

            trace = trace[:len(trace)-1]  # remove the '\n'

            actions = trace.split(",")

            urls += [actions[0]]

            races += [actions[1]]

            if "t"+str(maxsec) in actions[1:]:

                actions = actions[:actions.index("t"+str(maxsec))]

            actions = actions[2:]

            #actions = [a for a in actions if not a[0] == "t"]

            #dico = Counter(actions) # collections.OrderedDict()

            dataline= [0 for i in range (0,len(features))]

            for a in actions:

                if not a[0] == "t":

                    dataline[features.index(a)]+=1

            print(dataline)

            #features += [dico]

        





            

#  features_name_list = sorted(set([action for f in features for action in f.keys()]))

##for i in range(0, len(features_name_list)):

#  print(features_name_list[i])

# print(features[i])

# 

#       df[features_name_list[i]] = features[i]

#  df["race"] = races

# df["urls"] = urls

    

    #with open(outputfile, "w") as text_file:

    #    text_file.write(",".join(features_name_list) + ",race,battleneturl\n" )

    #    index = 0

    #    for f in features:

    #        text_file.write(",".join([str(f[k]) for k in features_name_list]) + "," + races[index] + "," + urls[index] + '\n')

    #        index += 1

traces2features("../input/TRAIN.CSV", "../output/features-train.txt", 100) 



import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation

from sklearn import metrics

from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn import tree

from collections import Counter



def encode_target(df, target_column):

    """Add column to df with integers for the target.

    Args

    ----

    df -- pandas DataFrame.

    target_column -- column to map to int, producing

                     new Target column.

    Returns

    -------

    df_mod -- modified DataFrame.

    targets -- list of target names.

    """

    df_mod = df.copy()

    targets = df_mod[target_column].unique()

    map_to_int = {name: n for n, name in enumerate(targets)}

    df_mod[target_column] = df_mod[target_column].replace(map_to_int)

    return(df_mod, targets)





def remove_unpopulated_classes(_df, target_column, threshold):

    """

    Removes any row of the df for which the label in target_column appears less

    than threshold times in the whole frame (not enough populated classes)

    :param df: The dataframe to filter

    :param target_column: The target column with labels

    :param threshold: the number of appearances a label must respect

    :return: The filtered dataframe

    """

    count = Counter(_df[target_column])

    valid = [k for k in count.keys() if count[k] >= threshold]

    _df = _df[_df[target_column].isin(valid)]

    return _df





def learn(inputfile, minlabels):



    df = pd.read_csv(inputfile)

    print (len(df), " input rows before filter unpopulated classes ")

    df2, _ = encode_target(df, "battleneturl")

    df3, _ = encode_target(df2, "race")

    df4 = remove_unpopulated_classes(df3, "battleneturl", minlabels)

    print(len(df4), " rows after")

    train_data = df4.values

    target = train_data[:, -1:].ravel()

    features = train_data[:, :-1]

    print("Data read.")



    # Cross validation

    model = RandomForestClassifier(max_features=None)

    print(model.get_params())

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features,target, test_size=0.2, random_state=0)

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    print(score)



    # 5 Cross validation

    #model = RandomForestClassifier(max_features=None)

    #print(model.get_params())

    #scores = cross_validation.cross_val_score(model, features, target, cv=5, scoring='accuracy')

    #print("accuracy: ", scores)

    #scores = cross_validation.cross_val_score(model, features, target, cv=5, scoring='precision_micro')

    #print("precision:", scores)



    return score

traces2features("../input/TRAIN.CSV", "../output/features-train.txt", 100) 

#score = learn("../input/features-train.txt", 1)

#print (score)
