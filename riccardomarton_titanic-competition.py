# Import declarations



import re # regular expressions

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # plotting



from sklearn import preprocessing

from sklearn.utils import shuffle # Shuffle dataset function

from sklearn.decomposition import PCA # Principal component Analysis dimensionality reduction

from sklearn.svm import SVC # State vector machine

from sklearn.model_selection import learning_curve # Learning curve

from sklearn.metrics import f1_score # benchmarking
# read data

raw_data = pd.read_csv('../input/train.csv')

raw_data = shuffle(raw_data)



test_data = pd.read_csv('../input/test.csv')



# replace gender with index

gender_dic = {"male": 0, "female": 1}

raw_data.ix[:,4] = raw_data.ix[:,4].replace(gender_dic)

test_data.ix[:,3] = test_data.ix[:,3].replace(gender_dic)



# replace port with index

port_dic = {"S": 0, "C": 1, "Q": 2}

raw_data.ix[:,11] = raw_data.ix[:,11].replace(port_dic)

test_data.ix[:,10] = test_data.ix[:,10].replace(port_dic)



#raw_data.fillna(0, inplace=True)



# cabin data

raw_data = raw_data.assign(Cabin_count=np.nan);

raw_data = raw_data.assign(Cabin_letter=np.nan);

raw_data = raw_data.assign(Cabin_number=np.nan);

test_data = test_data.assign(Cabin_count=np.nan);

test_data = test_data.assign(Cabin_letter=np.nan);

test_data = test_data.assign(Cabin_number=np.nan);



def map_cabin(row):

   

    cabin_str = row['Cabin']

    if not isinstance(cabin_str, str):

        return row

        

    cabin_parts = cabin_str.split(" ")

    row['Cabin_count'] = len(cabin_parts)

    

    lc = cabin_parts[len(cabin_parts)-1]

    cl = re.findall('[A-Z]', lc)

    cn = re.findall('\d+', lc)

    

    if len(cl) > 0:

        row['Cabin_letter'] = ord(cl[0])

    else:

        row['Cabin_letter'] = np.nan

    

    if len(cn) > 0:

        row['Cabin_number'] = int(cn[0])

    else:

        row['Cabin_number'] = np.nan

    

    return row



raw_data = raw_data.apply(map_cabin, axis=1)

test_data = test_data.apply(map_cabin, axis=1)



# extract columns

data_X = raw_data.ix[:, [2,4,5,6,7,9,12,13,14]]

test_X = test_data.ix[:, [1,3,4,5,6,8,11,12,13]]

#data_X = raw_data.ix[:, [4,5]]

data_y = raw_data.ix[:,1]



#remove NaN and normalize

data_X = data_X.fillna(data_X.mean())

test_X = test_X.fillna(data_X.mean())



#poly = preprocessing.PolynomialFeatures(3)

#data_X = poly.fit_transform(data_X)



min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(data_X)

data_X = pd.DataFrame(np_scaled)



np_scaled = min_max_scaler.fit_transform(test_X)

test_X = pd.DataFrame(np_scaled)



# seaparate training and cross validation

msk = np.random.rand(len(data_X)) < 0.8



train_X = data_X[msk]

train_y = data_y[msk]



cv_X = data_X[~msk]

cv_y = data_y[~msk]



#raw_data.head()

#data_X.head()

test_X.head()
pca = PCA(n_components=2)

t = pca.fit_transform(data_X)

t_x, t_y = t.T



plt.scatter(t_x, t_y, c=data_y)

plt.show()
est = SVC(C=0.4)



ts = [10,25,50,100,200,300,400,445]

train_sizes, train_scores, valid_scores = learning_curve(est, data_X, data_y, train_sizes=ts, cv=2)



line1, = plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train score")

line2, = plt.plot(train_sizes, np.mean(valid_scores, axis=1), 'o-', label="CV score")

plt.legend(handles=[line1, line2])

plt.show()
# Train classifier and print score

est.fit(train_X, train_y)

predict = est.predict(cv_X)

score = f1_score(cv_y, predict, average='macro')

print(score)
# read test set and predict results

predict = est.predict(test_X)



test_data = pd.read_csv('../input/test.csv')

result = {'PassengerId': test_data.ix[:,0], 'Survived': predict}

resultDf = pd.DataFrame(result)



print(resultDf.to_csv())