import numpy as np

import pandas as pd

from sklearn.metrics import f1_score

from sklearn.linear_model import SGDClassifier

f1 = {}
df = pd.read_csv('../input/pulsar_stars.csv')

df = df.sample(frac=1)

df.head()
X = df.drop('target_class', axis=1)

Y = df['target_class']
split = int(len(X)*0.6)
x_train = X[:split]

y_train = Y[:split]

x_test = X[split:]

y_test = Y[split:]
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

x_train = scalar.fit_transform(x_train)

x_test = scalar.transform(x_test)
sgd = SGDClassifier(loss='log',random_state=10, class_weight='balanced', alpha=0.01,n_jobs=-1)

sgd.fit(x_train, y_train)

print('Train accuracy = {0}, Test accuracy = {1}'.format(sgd.score(x_train, y_train), sgd.score(x_test, y_test)))

f1['full'] = [f1_score(sgd.predict(x_train), y_train),

          f1_score(sgd.predict(x_test), y_test)]
p=0.3

ind = np.array(np.random.choice(X.shape[0]*X.shape[1],

                                size=(int(X.shape[0]*X.shape[1]*p)), 

                                replace=False))

for i in ind:

    X.iloc[i%X.shape[0],i%X.shape[1]]=np.nan
Xm = X.copy()
Xm_filled = Xm.fillna(Xm.mean())

x_train = Xm_filled[:split]

y_train = Y[:split]

x_test = Xm_filled[split:]

y_test = Y[split:]
x_train = scalar.fit_transform(x_train)

x_test = scalar.transform(x_test)
sgd.fit(x_train, y_train)

print('Train accuracy = {0}, Test accuracy = {1}'.format(sgd.score(x_train, y_train), sgd.score(x_test, y_test)))

f1['mean'] = [f1_score(sgd.predict(x_train), y_train),

          f1_score(sgd.predict(x_test), y_test)]
def knn_na_filler(Xin, delta=0.01, n_neighbors=5):

    Xin = Xin.copy()

    from sklearn.neighbors import KNeighborsRegressor

    from sklearn.pipeline import Pipeline

    from sklearn.preprocessing import StandardScaler

    print('Filling Na with KNN...')

    missindxs = {col:Xin[Xin.loc[:,col].isnull()].index for col in Xin.columns}

    

    prev_mean_score = 0

    i=0

    while True:

        i+=1

        mean_score = 0 #mean score for column values prediction on test

        for col in Xin.columns:

            missindx = missindxs[col] #get indeces of missing values

            

            xtarget = Xin.loc[Xin.index.isin(missindx),:].drop(axis=1, columns=col) # features for missing values

            xtarget = xtarget.fillna(np.mean(xtarget)) # Na values from other columns filling with mean

            xtr = Xin.loc[~Xin.index.isin(missindx),:].drop(axis=1, columns=col) # features for training data

            xtr = xtr.fillna(np.mean(xtr))

            ytr = Xin.loc[~Xin.index.isin(missindx),col]

            

            knnna = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')

            scalar = StandardScaler()

            pipe = Pipeline([('scalar', scalar), ('model', knnna)])

            pipe.fit(xtr[:int(len(xtr)*0.7)], ytr[:int(len(xtr)*0.7)])

            train_score = pipe.score(xtr[:int(len(xtr)*0.7)], ytr[:int(len(xtr)*0.7)])

            test_score = pipe.score(xtr[int(len(xtr)*0.7):], ytr[int(len(xtr)*0.7):])

            mean_score+=test_score

            

            Xin.loc[Xin.index.isin(missindx), col] = pipe.predict(xtarget) # filling Nan in target column with predictions

        

        

        mean_score /= len(Xin.columns)

        print('Mean score at itteration {0} eq {1}'.format(i, mean_score))

        if (mean_score-prev_mean_score)<delta:

            break

        prev_mean_score = mean_score

        

    return Xin
Xm_filled = Xm.fillna(knn_na_filler(Xm, 0.001, 5))

x_train = Xm_filled[:split]

y_train = Y[:split]

x_test = Xm_filled[split:]

y_test = Y[split:]

x_train = scalar.fit_transform(x_train)

x_test = scalar.transform(x_test)
sgd.fit(x_train, y_train)

print('Train accuracy = {0}, Test accuracy = {1}'.format(sgd.score(x_train, y_train), sgd.score(x_test, y_test)))

f1['knn'] = [f1_score(sgd.predict(x_train), y_train),

             f1_score(sgd.predict(x_test), y_test)]
print('F1 score in full dataset: Train={}, Test={}'.format(f1['full'][0],f1['full'][1]))

print('F1 score with mean replacing: Train={}, Test={}'.format(f1['mean'][0],f1['mean'][1]))

print('F1 score with KNN replacing: Train={}, Test={}'.format(f1['knn'][0],f1['knn'][1]))