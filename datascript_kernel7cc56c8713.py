# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv # import the library that enabling read/write csv files



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
filepath = "/kaggle/input/pima-indians-diabetes-database/diabetes.csv"

with open(filepath) as csvfile:

    dataset = csv.reader(csvfile)

    dataset = np.array(list(dataset))

    train_set_limit = int(len(dataset)*0.80)

    train_set = dataset[1:train_set_limit]

    test_set = dataset[train_set_limit:]

    print("data size : ", len(dataset))

    print("train data size : ",len(train_set))

    print("test data size : ", len(test_set))

    print(train_set[0])

    print(train_set[1])



# link the index and the name of the different variables/features (headers)

headers = {}

ind = 0

for header in dataset[0]:

    headers[header]=ind

    ind+=1

print(headers)
# arrays containing indexes of the positive/negative outcome rows

indexes_positive_outcomes = []

indexes_negative_outcomes = []



# counter of ages and glucose depending on outcomes

ages_by_outcome = {"positive":{}, "negative":{}}

glucose_by_outcome = {"positive":{}, "negative":{}}



index = 0 # the current index in the loop



# possible to create a function to adapt it for every variable

# For each row of the dataset

for h in train_set:

    # I count the number of positive outcome

    if h[headers["Outcome"]]=='1':

        # stacking the positive outcome

        indexes_positive_outcomes.append(index)

    else:

        # stacking the negative outcome

        indexes_negative_outcomes.append(index)

    index+=1

    

nb_negative_outcomes = len(indexes_negative_outcomes)

nb_positive_outcomes = len(indexes_positive_outcomes)

print("number of positive outcomes : ", nb_positive_outcomes)

print("number of negative outcomes : ", nb_negative_outcomes)
# Create a dictionnary for each feature (with a section for positive outcome and another one for negative ones)

{'Pregnancies': 0, 'Glucose': 1, 'BloodPressure': 2, 'SkinThickness': 3, 'Insulin': 4, 'BMI': 5, 'DiabetesPedigreeFunction': 6, 'Age': 7, 'Outcome': 8}

age_value_pos = [train_set[pos_row][headers["Age"]] for pos_row in indexes_positive_outcomes] # get the values of age variable for positive outcome

# Return a tuple with values for positive outcome first then for negative outcome

def get_values_by_variable_by_outcome(var_name): 

    values_for_positive_outcome = [train_set[pos_row][headers[var_name]] for pos_row in indexes_positive_outcomes] 

    values_for_negative_outcome = [train_set[neg_row][headers[var_name]] for neg_row in indexes_negative_outcomes] 

    return (values_for_positive_outcome, values_for_negative_outcome)



# Function counting the frequency of the given variable

def count_variable_frequency_from_values(var_name):

    final_dico = {}

    dico = {}

    pos_values,neg_values = get_values_by_variable_by_outcome(var_name)

    for val in pos_values:

        if val not in dico.keys():

            dico[val] = 1

        else:

            dico[val] += 1

    final_dico["positive"]=dico

    dico = {}

    for val in neg_values:

        if val not in dico.keys():

            dico[val] = 1

        else:

            dico[val] += 1

    final_dico["negative"]=dico

    return final_dico
# Getting density by feature by value by outcome

def proba_from_dict(var_name, var, outcome="na"):

    # compute the density_by_outcome

    var_density_by_outcome = count_variable_frequency_from_values(var_name)

    #print("variable name : ", var_name)

    #print("value : ", var)

    #print("density for this value for positive outcome :", var_density_by_outcome["positive"][str(var)])

    #print("total positive outcome :", sum(var_density_by_outcome["positive"].values()))

    # the density of var

    nb_val_positive_outcome = sum(var_density_by_outcome["positive"].values())

    nb_val_negative_outcome = sum(var_density_by_outcome["negative"].values())

    nb_val = nb_val_negative_outcome + nb_val_positive_outcome

    # TODO: take in account when the value doesn't exist for an given outcome

    #print("P(var=", var, "|Outcome=positive)", var_density_by_outcome["positive"][str(var)]/nb_val_positive_outcome)

    #print("P(var=", var, "|Outcome=negative)", var_density_by_outcome["negative"][str(var)]/nb_val_negative_outcome)

    #print("P(var=", var,")=",(var_density_by_outcome["positive"][str(var)]+var_density_by_outcome["negative"][str(var)])/(nb_val_positive_outcome+nb_val_negative_outcome))

    if outcome=="positive":

        return var_density_by_outcome["positive"][str(var)]/nb_val_positive_outcome

    elif outcome=="negative":

        return var_density_by_outcome["negative"][str(var)]/nb_val_negative_outcome

    else:

        return var_density_by_outcome["positive"][str(var)]+var_density_by_outcome["negative"][str(var)]/(nb_val_positive_outcome+nb_val_negative_outcome)

# Show the density of the given variable

def qty_2_density_dict(var_name,quantity_dict_var, outcome):

    density_dict = {}

    for key in quantity_dict_var:

        density_dict[key]= proba("Age")
age_quantities = count_variable_frequency_from_values("Age")

print(age_quantities)
# 1) function computing probabilities

test_dict = {'10':10, '23':4, '17':5, '27':1}

_KEY_INDEX = 0

_VAL_INDEX = 1

_PROB_INDEX = 1



def get_data_shape_from_dict(contingences_dict):

    keys = np.array([key for key in contingences_dict.keys()])

    vals = np.array([contingences_dict[key] for key in keys])

    return (keys,vals)



def get_total_inputs_from_dict(dict):

    data = get_data_shape_from_dict(dict)

    return np.add.reduce(data[_VAL_INDEX])



def get_total_inputs_from_array(vect):

    return np.add.reduce(vect)



def get_probabilities_from_contingences(data_cont):

    total_inputs = get_total_inputs_from_array(data_cont[_VAL_INDEX])

    probabilities = np.array([val/total_inputs for val in data_cont[_VAL_INDEX]])

    return (data_cont[_KEY_INDEX],probabilities)



def valid_probabilities(data_cont):

    data_probabilities = get_probabilities_from_contingences(data_cont)

    somme_prob = np.add.reduce(data_probabilities[_PROB_INDEX])

    print("somme des prob ", somme_prob)



# import plot library

import matplotlib.pyplot as plt

    

def plot_probabilities(data_prob, var_name):

    df = pd.DataFrame({str(var_name):data_prob[_KEY_INDEX], 'probabilities':data_prob[_VAL_INDEX]})

    df.plot(kind="bar", x=var_name, y="probabilities")

    plt.show()



# find the probability of x (var name) to have the value val 

# unused var_name right now

def proba(val, data_prob, var_name=""):

    print(data_prob)

    val = str(val)

    if val in data_prob[_KEY_INDEX]:

        p = [data_prob[_PROB_INDEX][i] for i in range(len(data_prob[_KEY_INDEX])) if data_prob[_KEY_INDEX][i]==str(val)]

        return p

    else:

        print("Erreur : cette valeur n'est pas pris en compte dans les données")
test_dict2 = {'Small':15, 'Medium':22, 'LARGE':34, 'XS':2}

data2_cont = get_data_shape_from_dict(test_dict2)

data2_prob = get_probabilities_from_contingences(data2_cont)

data2_prob = get_probabilities_from_contingences(data2_cont)



data_cont = get_data_shape_from_dict(test_dict)

data_prob = get_probabilities_from_contingences(data_cont)

valid_probabilities(data_prob)
#plot_probabilities(data_prob,"Age")

#plot_probabilities(data2_prob, "Size")

proba(23, data_prob)
def get_proba_var(x_val,data_prob_x):

    proba_x = [data_prob_x[_PROB_INDEX][ind] for ind in range(len(data_prob_x[_KEY_INDEX])) if data_prob_x[_KEY_INDEX][ind]==str(x_val)]

    return proba_x[0]

# 2) Simple function to compute a simple conditional probability

# var x and y are tuples with first the key value and in second the value associated to the variable

def proba_x_knowing_y(x_val, data_prob_x, y_val, data_prob_y):

    # supposing that variable x and y are indeendant

    x_prob = get_proba_var(x_val,data_prob_x)

    y_prob = get_proba_var(y_val, data_prob_y)

    return x_prob*y_prob



_DATA_INDEX = 1

# by the law of Markov Chain

# conditional probaility considering several variables observations

# Z list is a list of observations of variables (key,values) and their respective probabilities vector

def proba_x_knowing_Zs_variables(x_val, data_prob_x, Z_list):

    x_prob = proba(x_val, data_prob_x)

    proba_zs_values = [proba(Z_list[_KEY_INDEX][i], Z_list)]
get_index_proba_var(23,data_prob)

#proba_x_knowing_y(23, data_prob, 'Small', data2_prob)