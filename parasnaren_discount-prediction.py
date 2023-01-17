import numpy as np

import pandas as pd

import os

from sklearn.linear_model import LogisticRegression as LR

from sklearn.model_selection import train_test_split as tts, cross_val_score

from sklearn.svm import SVC

from sklearn.metrics import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import *

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the files

train = pd.read_csv('../input/Train.csv').dropna(how='all').fillna(0)

test = pd.read_csv('../input/test.csv')

y = train.iloc[:, 3:].dropna(how='all')

data = pd.concat([train.iloc[:,:3], test], axis=0)



basket = pd.read_csv('../input/Product_sales_train_and_test.csv')

sample = pd.read_csv('../input/Sample_Submission.csv')
def view_counts(train):

    for frame, group in train.groupby('Customer'):

        print(frame, 'present ', len(group), 'times')

        print("5% : ", (group['5'] == 1).sum())

        print("12% : ", (group['12'] == 1).sum())

        print("18% : ", (group['18'] == 1).sum())

        print("28% : ", (group['28'] == 1).sum())
baskets = []

for i in basket['Customer_Basket'].values:

    i = i.strip('][')

    l = pd.to_numeric(i.split(' '))

    baskets.append(l)

    

basket['Baskets'] = baskets

basket = basket.drop('Customer_Basket', axis=1)
tmp = pd.DataFrame()

tmp['BillNo'] = data['BillNo']

tmp = pd.merge(tmp, basket, how='inner', on='BillNo')

counts = []

for i, row in tmp.iterrows():

    c = len(row['Baskets'])

    counts.append(c)
X = pd.DataFrame(columns = ['BillNo'])

X['BillNo'] = data['BillNo']

X = pd.merge(X, basket, how='inner', on='BillNo')

x = pd.DataFrame(columns=[str(i) for i in range(1001, 1810)])

X = pd.concat([X, x], axis=1)
def get_products(X):

    for i, row in X.iterrows():

        for value in row['Baskets']:

            if row[str(value)] == 1:

                row[str(value)] += 1

            else:

                row[str(value)] = 1

    return X
X = get_products(X)

X = X.fillna(0)

X.drop('Baskets', axis=1, inplace=True)

X['counts'] = counts
# Prepare y 

label=[]

y.columns = ['5','12','18','28']

for i, row in y.iterrows():

    if row['5'] == 1:

        label.append(1)

    elif row['12'] == 1:

        label.append(2)

    elif row['18'] == 1:

        label.append(3)

    elif row['28'] == 1:

        label.append(4)

    else:

        label.append(0)

        

y['label'] = label

y.drop(['5','12','18','28'], axis=1, inplace=True)

y = pd.Series(y['label'].values)
dates = pd.get_dummies(data['Date'])

dates = dates.reset_index(drop=True)

cust = pd.get_dummies(data['Customer'])

cust = cust.reset_index(drop=True)

#X_tmp = pd.concat([X, dates], axis=1)

X_tmp = pd.concat([X, dates, cust], axis=1)
X_train, X_test = X_tmp.iloc[:12200, 1:], X_tmp.iloc[12200:, 1:]
def prob_to_labels(dec):

    dec_pred = []

    for i, row in dec.iterrows():

        l = [0,0,0,0,0]

        index = np.argmax(row)

        l[index] = 1

        dec_pred.append(l)

    dec_pred = pd.DataFrame(dec_pred, columns = [1,2,3,4])

    return dec_pred



def get_prob_scores(dec):

    normalizedArray = []

    for row in range(0, len(dec)):

        l = []

        Min =  min(dec[row])

        Max = max(dec[row])

        for element in dec[row]:

            l.append(float(element-Min)/float(Max- Min) )

        normalizedArray.append(l)

        

    #Normalize to 1

    newArray = []

    for row in range(0, len(normalizedArray)):

        li = [x / sum(normalizedArray[row]) for x in normalizedArray[row]]

        newArray.append(li)

        

    sample_p = pd.DataFrame(newArray, columns=[1,2,3,4])

    return sample_p
def cross_validate(clf, X_train, y, cv=3):

    prob_scores, scores = 0, 0

    for j in range(cv):

        train, test, y_train, y_test = tts(X_train, y, test_size=0.2, 

                                           random_state=j)

    

    

        clf.fit(train, y_train)

        

        ### Decision scores

        #dec = clf.decision_function(test)

        #dec = get_prob_scores(dec)

        #dec_pred = prob_to_label(dec)

        

        # Verify

        #y_dec = pd.get_dummies(y_test)

        #print("\n", j, " iteration")

        #print("log loss: ", log_loss(y_dec, dec))

        #print("predicted log loss: ", log_loss(y_dec, dec_pred))

        #a = accuracy_score(pd.get_dummies(y_test), dec_pred)

        #print("Accuracy using prob scores: ", a)

        

        ### Straight up predictions

        pred = clf.predict(test)

        s = accuracy_score(y_test, pred)

        #print("Normal predictions: ", s)

        #prob_scores += a

        scores += s

        

    #print("final prob scores mean: ", prob_scores/cv)

    return (scores/cv)
clf1 = LR()

#clf2 = GradientBoostingClassifier()

clf3 = RandomForestClassifier(n_estimators=100)

#clf4 = SVC()
train, test, y_train, y_test = tts(X_train, y, test_size=0.2, random_state=0)

clf1.fit(train, y_train)

#clf2.fit(train, y_train)

clf3.fit(train, y_train)
print("Logistic Regression : ", clf1.score(test, y_test))

print("Random Forest: ", clf3.score(test, y_test))
"""clf = RandomForestClassifier()

grid_values = {'n_estimators': [100, 200]}



grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)

grid_clf_acc.fit(train, y_train)



grid_pred = grid_clf_acc.predict(test)

print("Grid Search accuracy: ", accuracy_score(y_test, grid_pred))



print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)"""
print("Logistic Regression cross-validation: ", cross_validate(clf1, X_train, y, cv=3))
print("Random Forest cross-validation: ", cross_validate(clf3, X_train, y, cv=3))
clf3.fit(train, y_train)
pred_prob = clf3.predict_proba(test)[:, 1:]

pred = clf3.predict(test)

pred_dummy = pd.get_dummies(pred)

y_test_dummy = pd.get_dummies(y_test).iloc[:, 1:]
if 1 not in pred_dummy.columns:

    temp = pd.DataFrame(columns = [1])

    pred_dummy = pd.concat([temp, pred_dummy], axis=1)

    pred_dummy[1] = 0

pred_dummy
print("Probability log loss: ", log_loss(y_test_dummy, pred_prob))

print("Labels log loss: ", log_loss(y_test_dummy, pred_dummy))
final_clf = RandomForestClassifier(n_estimators=100)

final_clf.fit(X_train, y)
sample_pred_prob = pd.DataFrame(final_clf.predict_proba(X_test)[:, 1:], columns=[1,2,3,4])

sample_pred = final_clf.predict(X_test)

sample_pred = pd.get_dummies(sample_pred)

if 1 not in sample_pred.columns:

    sample_pred = pd.concat([pd.DataFrame(columns=[1]), sample_pred], axis=1)

    sample_pred[1] = 0

sample_pred.head()

#sample_pred_prob.head()

#sample_pred_prob.shape
sample = pd.read_csv('../input/Sample_Submission.csv')

sample['Discount 5%'] = sample_pred_prob[1]

sample['Discount 12%'] = sample_pred_prob[2]

sample['Discount 18%'] = sample_pred_prob[3]

sample['Discount 28%'] = sample_pred_prob[4]

sample.to_csv('output_probabilities.csv', index=False)



sample['Discount 5%'] = sample_pred[1]

sample['Discount 12%'] = sample_pred[2]

sample['Discount 18%'] = sample_pred[3]

sample['Discount 28%'] = sample_pred[4]

sample.to_csv('output_labels.csv', index=False)
def get_predictions(per, sample_pred_prob):

    actual = []

    for i, row in sample_pred_prob.iterrows():

        r = list(row.sort_values(ascending=False))

        maxx = r[0]

        second_max = r[1]

        l = [0,0,0,0]

        if maxx - second_max > per:

            for j in range(1,5):

                if row[j] == maxx:

                    l[j-1] = 1

                    break

        actual.append(l)

    return actual
def check_result(one):

    s = 0

    for col in one.columns:

        print(len(one) - one[col].value_counts()[0])

        s += len(one) - one[col].value_counts()[0]

    print("Count: ", s, '/', len(one))
for i in [50, 98]:

    one = pd.DataFrame(get_predictions(i/100.00, sample_pred_prob), columns=[1,2,3,4])

    sample['Discount 5%'] = one[1]

    sample['Discount 12%'] = one[2]

    sample['Discount 18%'] = one[3]

    sample['Discount 28%'] = one[4]

    filename = 'output_' + str(i) + '.csv'

    sample.to_csv(filename, index=False)