import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
df_train = pd.read_csv('../input/minimise/train_data.csv.xls')
df_train.tail()
df_train.sample(5)
df_train.loc[df_train['amount'] > -2000]['amount'].hist(bins = 100, figsize=(15, 5))
plt.show()
df_train['category'].value_counts()
df_train.groupby('category')['amount'].agg(['mean', 'median', 'std'])
df_train.loc[df_train['amount'] > -1000].groupby('category')['amount'].hist(bins = 50, figsize=(15, 5), alpha = 0.3)
plt.show()
df_train['merchant'].value_counts()[:10]
from dateutil.parser import parse
def get_day_of_week(entry):
    date = parse(entry)
    return date.isoweekday()
    
def get_time_of_day(entry):
    date = parse(entry)
    return date.hour
df_train['dow'] = df_train['transactionDate'].apply(get_day_of_week)
df_train['tod'] = df_train['transactionDate'].apply(get_time_of_day)
df_train.groupby('dow')['amount'].agg(['count', 'mean', 'median', 'std'])
df_train.groupby('category')['tod'].hist(bins = 24, figsize=(15, 5), alpha = 0.3)
plt.show()
df_train.loc[df_train['category'] == 'Education']['tod'].hist(bins = 24, figsize=(15, 5), alpha=0.4)
df_train.loc[df_train['category'] == 'Home']['tod'].hist(bins = 24, figsize=(15, 5), alpha=0.4)
df_train.loc[df_train['category'] == 'Groceries']['tod'].hist(bins = 24, figsize=(15, 5), alpha=0.4)
df_train.loc[df_train['category'] == 'Holiday_Travel']['tod'].hist(bins = 24, figsize=(15, 5), alpha=0.4)
df_train.loc[df_train['category'] == 'Pets']['tod'].hist(bins = 24, figsize=(15, 5), alpha=0.4)
df_train.loc[df_train['category'] == 'Transport']['tod'].hist(bins = 24, figsize=(15, 5), alpha=0.4)
df_train['description'].value_counts()[0:15]
df_train['description'] = df_train['merchant'] + " " + df_train['description']
df_train['description'].value_counts()[0:15]
di = {}
all_text = list(df_train['description'])

for t in all_text:
    # cast string to lowercase and split on whitespace
    tokens = t.lower().split()
    for c in tokens:
        if di.get(c) != None:
            di[c] += 1
        else:
            di[c] = 1
df_text = pd.DataFrame(pd.Series(di))
df_text['freq'] = df_text[0]
df_text['prop'] = df_text['freq']/df_text['freq'].sum()
df_text = df_text.sort_values('prop', ascending = False)
df_text['prop'].sum()
2000/df_text.shape[0]
df_text[:8700]['prop'].sum()
df_text[:2000]['prop'].sum()
keys = list(df_text[:500].index)
del df_text
def vectorize_text(entry):
    X = np.zeros(500)
    for i in range(500):
        if entry.find(keys[i]) > -1:
            X[i] += 1
    return X
def numerate_label(entry):
    if entry == 'Clothing':
        return 1
    elif entry == 'Eat_Out':
        return 2
    elif entry == 'Education':
        return 3
    elif entry == 'Entertainment':
        return 4
    elif entry == 'Gifts_Donations':
        return 5
    elif entry == 'Groceries':
        return 6
    elif entry == 'Health_Fitness':
        return 7
    elif entry == 'Holiday_Travel':
        return 8
    elif entry == 'Home':
        return 9
    elif entry == 'Medical':
        return 10
    elif entry == 'Pets':
        return 11
    elif entry == 'Transport':
        return 12
    else:
        print('Somethings up')
df_train['number_label'] = df_train['category'].apply(numerate_label)
df_train['number_label'].value_counts()
df_train['vector_text'] = df_train['description'].apply(vectorize_text)
train_X = df_train['vector_text'].as_matrix()
Y_train = df_train['number_label'].as_matrix()
del df_train
x_train = []
for x in train_X:
    x_train.append(x)
del train_X
X_train = np.asarray(x_train)
df_test = pd.read_csv('../input/minimise/test_data.csv.xls', date_parser='transactionDate')
df_test['dow'] = df_test['transactionDate'].apply(get_day_of_week)
df_test['tod'] = df_test['transactionDate'].apply(get_time_of_day)
df_test['description'] = df_test['merchant'] + " " + df_test['description']
df_test['vector_text'] = df_test['description'].apply(vectorize_text)
test_X = df_test['vector_text'].as_matrix()
x_test = []
for x in test_X:
    x_test.append(x)

X_test = np.asarray(x_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200, )
rfc.fit(X_train, Y_train)
prob_pos = rfc.predict_proba(X_test)
import csv

id_tags = list(df_test['id'])

with open('submission.csv', 'a') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['id', '"Clothing"', '"Eat_Out"', '"Education"', '"Entertainment"', 
                     '"Gifts_Donations"', '"Groceries"', '"Health_Fitness"', 
                     '"Holiday_Travel"', '"Home"', '"Medical"', '"Pets"', '"Transport"'])
    for i in range(len(prob_pos)):
        #Write item to outcsv
        writer.writerow([id_tags[i], prob_pos[i][0], prob_pos[i][1], prob_pos[i][2], prob_pos[i][3], prob_pos[i][4], 
                         prob_pos[i][5], prob_pos[i][6], prob_pos[i][7], prob_pos[i][8], prob_pos[i][9], prob_pos[i][10],
                       prob_pos[i][11]])