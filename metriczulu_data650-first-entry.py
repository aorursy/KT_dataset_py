import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



train = pd.read_csv('../input/cars-train.csv')

val = pd.read_csv('../input/cars-test.csv')



train.head()
train = train.drop(['car.id'], axis=1)

val = val.drop(['car.id'], axis=1)
x_train = train.drop(['class'], axis = 1)

y_train = train['class']

x_val = val.drop(['class'], axis = 1)

y_val = val['class']
train.describe(include='all')
def plot_all_hists_df(df):

    size = df.shape[1]

    col_num = 4

    row_num = int(np.ceil(size/4))

    fig = plt.figure(figsize=(9,5))

    for i, name in enumerate(df):

        ax=fig.add_subplot(row_num,col_num,i+1)

        plt.hist(list(df[name]))

        ax.set_title(name)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show()



plot_all_hists_df(train)

plot_all_hists_df(val)
print('Total number of each times each category of class shows up training set:')

print(train['class'].value_counts())

print('*****')

print('Total number of each times each category of class shows up validation set:')

print(val['class'].value_counts())
class_counts_train = [680, 216, 38, 36]

class_counts_val = [227, 72, 13, 12]
total_tuples = set()

for index, row in train.iterrows():

    total_tuples.add((row[0],row[1],row[2],row[3],row[4],row[5]))

print('Total number of unique combinations of independent variables: {}'.format(len(total_tuples)))
train_num = train.replace({'unacc': 0, 'acc':1, 'good':2, 'vgood': 3, 'low':0, 'med':1, 'high':2, 'vhigh':3, 'small':0, 'big':2, "2":0, "3":1, "4":2, "more":3, "5more":3})

val_num = val.replace({'unacc': 0, 'acc':1, 'good':2, 'vgood': 3, 'low':0, 'med':1, 'high':2, 'vhigh':3, 'small':0, 'big':2, "2":0, "3":1, "4":2, "more":3, "5more":3})
train_num.head()
varXclass = list()

depvar_name = [('class X '+name) for name in train_num]

name = [x for x in train_num]

depvar_name = depvar_name[:6]

for i in range(6): varXclass.append(list())

for index, row in train_num.iterrows():

    for i in range(6):

        varXclass[i].append('class: '+str(row[6])+' X '+name[i]+' '+str(row[i]))
for i, x in enumerate(varXclass): 

    print(depvar_name[i]+' unique combos: {}'.format(len(set(x))))

    varXclass[i].sort()
def combos_to_hist(list_of_sets, names):

    fig = plt.figure(figsize=(15,10))

    num_cols = 4

    num_rows = int(np.ceil(len(list_of_sets)/4))

    for i, sets in enumerate(list_of_sets):

        ax=fig.add_subplot(num_rows,num_cols,i+1)

        plt.hist(sets)

        plt.xticks(fontsize = 12,rotation='vertical')

        ax.set_title(names[i])

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show()

    

combos_to_hist(varXclass, depvar_name)
def freq_dict(in_list):

    return {i:in_list.count(i) for i in set(in_list)}



freq_dict_list=[]



for combo in varXclass:

    combo_freq_dict = freq_dict(combo)

    for pair in combo_freq_dict:

        norm = class_counts_train[int(pair[7])]

        combo_freq_dict[pair] = combo_freq_dict[pair]/norm

    freq_dict_list.append(combo_freq_dict)



def dict_to_two_lists(input_dict):

    key_list = []

    value_list = []

    for key, value in input_dict.items():

        key_list.append(key)

        value_list.append(value)

    return key_list, value_list



def freq_dict_list_hist(input_list, names):

    size = len(input_list)

    col_num = 4

    row_num = int(np.ceil(size/4))

    fig = plt.figure(figsize=(15,10))

    for i, freq_dict in enumerate(input_list):

        ax=fig.add_subplot(row_num,col_num,i+1)

        key_list, value_list = dict_to_two_lists(freq_dict)

        plt.bar(key_list, value_list)

        plt.xticks(fontsize = 12,rotation='vertical')

        ax.set_title(names[i])

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show()

    

freq_dict_list_hist(freq_dict_list, depvar_name)
x_train = pd.get_dummies(x_train)

y_train = pd.get_dummies(y_train)

x_val = pd.get_dummies(x_val)

y_val = pd.get_dummies(y_val)



x_train_num = train_num.drop(['class'], axis=1)

y_train_num = train_num['class']

x_val_num = val_num.drop(['class'], axis=1)

y_val_num = val_num['class']
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score



clf = RandomForestClassifier(n_estimators = 500)

clf.fit(x_train, y_train)

print(classification_report(y_train, clf.predict(x_train)))

print(classification_report(y_val, clf.predict(x_val)))

print(accuracy_score(y_val, clf.predict(x_val)))
clf = RandomForestClassifier(n_estimators = 500)

clf.fit(x_train_num, y_train_num)

print(classification_report(y_train_num, clf.predict(x_train_num)))

print(classification_report(y_val_num, clf.predict(x_val_num)))

print(accuracy_score(y_val_num, clf.predict(x_val_num)))
from sklearn.naive_bayes import MultinomialNB



clf = MultinomialNB()

clf.fit(x_train, y_train_num)

print(classification_report(y_train_num, clf.predict(x_train)))

print(classification_report(y_val_num, clf.predict(x_val)))

print(accuracy_score(y_val_num, clf.predict(x_val)))
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()

clf.fit(x_train, y_train_num)

print(classification_report(y_train_num, clf.predict(x_train)))

print(classification_report(y_val_num, clf.predict(x_val)))

print(accuracy_score(y_val_num, clf.predict(x_val)))
from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier()

clf.fit(x_train, y_train)

print(classification_report(y_train, clf.predict(x_train)))

print(classification_report(y_val, clf.predict(x_val)))

print(accuracy_score(y_val, clf.predict(x_val)))
clf = KNeighborsClassifier()

clf.fit(x_train_num, y_train_num)

print(classification_report(y_train_num, clf.predict(x_train_num)))

print(classification_report(y_val_num, clf.predict(x_val_num)))

print(accuracy_score(y_val_num, clf.predict(x_val_num)))
from xgboost import XGBClassifier



clf = XGBClassifier(n_estimators = 1000)

clf.fit(x_train_num, y_train_num)

print(classification_report(y_train_num, clf.predict(x_train_num)))

print(classification_report(y_val_num, clf.predict(x_val_num)))

print(accuracy_score(y_val_num, clf.predict(x_val_num)))
n_range = []

acc_score = []

for n in range(100, 3000, 100):

    n_range.append(n)

    clf = XGBClassifier(n_estimators = n)

    clf.fit(x_train_num, y_train_num)

    acc_score.append(accuracy_score(y_val_num, clf.predict(x_val_num)))



plt.title("N estimators to accuracy")

plt.plot(n_range, acc_score, ls='-', marker='o', color='red', label='One-hot encoded')

plt.legend()
from xgboost import XGBClassifier



clf = XGBClassifier(n_estimators = 650)

clf.fit(x_train_num, y_train_num)

print(classification_report(y_train_num, clf.predict(x_train_num)))

print(classification_report(y_val_num, clf.predict(x_val_num)))

print(accuracy_score(y_val_num, clf.predict(x_val_num)))
test = pd.read_csv('../input/cars-final-prediction.csv')

test.head()
test_num = test.replace({'low':0, 'med':1, 'high':2, 'vhigh':3, 'small':0, 'big':2, "2":0, "3":1, "4":2, "more":3, "5more":3})

test_num.head()
test_num['num_predictions']=clf.predict(test_num.drop(['car.id'], axis=1))
test_num['class']=test_num['num_predictions'].replace({0:'unacc', 1:'acc', 2:'good', 3:'vgood'})

test_num.head()
comp_output=test_num[['car.id', 'class']]

comp_output.head()
plt.hist(list(comp_output['class']))

comp_output['class'].value_counts()
comp_output.to_csv('cars-submission.csv', index=False)