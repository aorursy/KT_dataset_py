import pandas as pd

import numpy as np

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import log_loss
datadir = '../input'

gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),

                      index_col='device_id')

gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),

                     index_col = 'device_id')

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

# Get rid of duplicate device ids in phone

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

events = pd.read_csv(os.path.join(datadir,'events.csv'),

                     parse_dates=['timestamp'], index_col='event_id')

appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 

                        usecols=['event_id','app_id','is_active'],

                        dtype={'is_active':bool})

applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
gatrain['trainrow'] = np.arange(gatrain.shape[0])

gatest['testrow'] = np.arange(gatest.shape[0])
brandencoder = LabelEncoder().fit(phone.phone_brand)

phone['brand'] = brandencoder.transform(phone['phone_brand'])

gatrain['brand'] = phone['brand']

gatest['brand'] = phone['brand']

Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 

                       (gatrain.trainrow, gatrain.brand)))

Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 

                       (gatest.testrow, gatest.brand)))

print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))
m = phone.phone_brand.str.cat(phone.device_model)

modelencoder = LabelEncoder().fit(m)

phone['model'] = modelencoder.transform(m)

gatrain['model'] = phone['model']

gatest['model'] = phone['model']

Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 

                       (gatrain.trainrow, gatrain.model)))

Xte_model = csr_matrix((np.ones(gatest.shape[0]), 

                       (gatest.testrow, gatest.model)))

print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))
appencoder = LabelEncoder().fit(appevents.app_id)

appevents['app'] = appencoder.transform(appevents.app_id)

napps = len(appencoder.classes_)

deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)

                       .groupby(['device_id','app'])['app'].agg(['size'])

                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)

                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)

                       .reset_index())

deviceapps.head()
d = deviceapps.dropna(subset=['trainrow'])

Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 

                      shape=(gatrain.shape[0],napps))

d = deviceapps.dropna(subset=['testrow'])

Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 

                      shape=(gatest.shape[0],napps))

print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]

applabels['app'] = appencoder.transform(applabels.app_id)

labelencoder = LabelEncoder().fit(applabels.label_id)

applabels['label'] = labelencoder.transform(applabels.label_id)

nlabels = len(labelencoder.classes_)
devicelabels = (deviceapps[['device_id','app']]

                .merge(applabels[['app','label']])

                .groupby(['device_id','label'])['app'].agg(['size'])

                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)

                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)

                .reset_index())

devicelabels.head()
d = devicelabels.dropna(subset=['trainrow'])

Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 

                      shape=(gatrain.shape[0],nlabels))

d = devicelabels.dropna(subset=['testrow'])

Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 

                      shape=(gatest.shape[0],nlabels))

print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')

Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')

print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))
targetencoder = LabelEncoder().fit(gatrain.group)

y = targetencoder.transform(gatrain.group)

nclasses = len(targetencoder.classes_)
def score(clf, random_state = 0):

    kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)

    pred = np.zeros((y.shape[0],nclasses))

    for itrain, itest in kf:

        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]

        ytr, yte = y[itrain], y[itest]

        clf.fit(Xtr, ytr)

        pred[itest,:] = clf.predict_proba(Xte)

        # Downsize to one fold only for kernels

        return log_loss(yte, pred[itest, :])

        print("{:.5f}".format(log_loss(yte, pred[itest,:])), end=' ')

    print('')

    return log_loss(y, pred)
Cs = np.logspace(-3,0,4)

res = []

for C in Cs:

    res.append(score(LogisticRegression(C = C)))

plt.semilogx(Cs, res,'-o');
score(LogisticRegression(C=0.02))
score(LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs'))
clf = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')

clf.fit(Xtrain, y)

pred = pd.DataFrame(clf.predict_proba(Xtest), index = gatest.index, columns=targetencoder.classes_)

pred.head()
pred.to_csv('logreg_subm.csv',index=True)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

devicelabelstfidf = devicelabels.groupby(['device_id','label'])['size'].agg(['sum']).unstack().fillna(0)

transformedlabels = tfidf.fit_transform(devicelabelstfidf)

transformedlabels = pd.DataFrame(transformedlabels.toarray())

dev_id = devicelabels.groupby('device_id')['size'].size().reset_index()

dev_id.drop(0,1,inplace=True)

transformedlabels = dev_id.join(transformedlabels)

transformedlabels = transformedlabels.merge(gatrain.reset_index()[['trainrow','device_id']], how='left',on='device_id').merge(gatest.reset_index()[['testrow','device_id']], how='left', on='device_id')
f = transformedlabels.dropna(subset=['trainrow'])

f.drop(['testrow','device_id'], axis=1, inplace=True)

f.set_index('trainrow',inplace=True)

f.sort_index(inplace=True)

new_index=np.arange(0,74645)

f = f.reindex(new_index).fillna(0)

Xtr_tfidflabel = csr_matrix(f)

g=transformedlabels.dropna(subset=['testrow'])

g.drop(['trainrow','device_id'], axis=1, inplace=True)

g.set_index('testrow',inplace=True)

g.sort_index(inplace=True)

new_index = np.arange(0,112071)

g = g.reindex(new_index).fillna(0)

Xte_tfidflabel = csr_matrix(g)
events['hour'] = events['timestamp'].map(lambda x:pd.to_datetime(x).hour)

events['hourbin'] = [1 if ((x>=1)&(x<=6)) else 2 if ((x>=7)&(x<=12)) else 3 if ((x>=13)&(x<=18)) else 4 for x in events['hour']]
tfidf = TfidfTransformer()

hourbintfidf = events.groupby(['device_id','hourbin'])['hourbin'].agg(['size']).unstack().fillna(0)

hourbintfidf = tfidf.fit_transform(hourbintfidf)

hourbintfidf = pd.DataFrame(hourbintfidf.toarray())

dev_id = events.groupby('device_id').size().reset_index()

dev_id.drop(0,1,inplace=True)

hourbintfidf = dev_id.join(hourbintfidf)

hourbintfidf = hourbintfidf.merge(gatrain.reset_index()[['trainrow','device_id']], how='left',on='device_id').merge(gatest.reset_index()[['testrow','device_id']], how='left', on='device_id')
f = hourbintfidf.dropna(subset=['trainrow'])

f.drop(['testrow','device_id'], axis=1, inplace=True)

f.set_index('trainrow',inplace=True)

f.sort_index(inplace=True)

new_index=np.arange(0,74645)

f = f.reindex(new_index).fillna(0)

Xtr_tfidfhourbin = csr_matrix(f)

g=hourbintfidf.dropna(subset=['testrow'])

g.drop(['trainrow','device_id'], axis=1, inplace=True)

g.set_index('testrow',inplace=True)

g.sort_index(inplace=True)

new_index = np.arange(0,112071)

g = g.reindex(new_index).fillna(0)

Xte_tfidfhourbin = csr_matrix(g)
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label,Xtr_tfidfhourbin,Xtr_tfidflabel), format='csr')

Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label,Xte_tfidfhourbin,Xte_tfidflabel), format='csr')

print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))
from sklearn.feature_selection import SelectKBest,chi2

selector = SelectKBest(chi2, k=8000).fit(Xtrain, y)
Xtrainkb = selector.transform(Xtrain)

Xtestkb = selector.transform(Xtest)
def batch_generator(X, y, batch_size, shuffle):

    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)

    number_of_batches = np.ceil(X.shape[0]/batch_size)

    counter = 0

    sample_index = np.arange(X.shape[0])

    if shuffle:

        np.random.shuffle(sample_index)

    while True:

        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        X_batch = X[batch_index,:].toarray()

        y_batch = y[batch_index]

        counter += 1

        yield X_batch, y_batch

        if (counter == number_of_batches):

            if shuffle:

                np.random.shuffle(sample_index)

            counter = 0
def batch_generatorp(X, batch_size, shuffle):

    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)

    counter = 0

    sample_index = np.arange(X.shape[0])

    while True:

        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        X_batch = X[batch_index, :].toarray()

        counter += 1

        yield X_batch

        if (counter == number_of_batches):

            counter = 0
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from keras.optimizers import SGD

from keras.layers.advanced_activations import PReLU
def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(150, input_dim=Xtrainkb.shape[1], init='normal'))

    model.add(PReLU())

    model.add(Dropout(0.4))

    model.add(Dense(50, input_dim=Xtrainkb.shape[1], init='normal'))

    model.add(PReLU())

    model.add(Dropout(0.2))

    model.add(Dense(12, init='normal', activation='softmax'))

    # Compile model

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss

    return model
model=baseline_model()
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(Xtrainkb, y, train_size=.98, random_state=10)
fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),

                         nb_epoch=15,

                         samples_per_epoch=69984,

                         validation_data=(X_val.todense(), y_val), verbose=2

                         )
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
print('logloss val {}'.format(log_loss(y_val, scores_val)))
model=baseline_model()
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
print('logloss val {}'.format(log_loss(y_val, scores_val)))
scores = model.predict_generator(generator=batch_generatorp(Xtestkb, 800, False), val_samples=Xtestkb.shape[0])

result = pd.DataFrame(scores , columns=targetencoder.classes_)

result["device_id"] = device_id

result = result.set_index("device_id")

result.to_csv('bagofappskeras', index=True, index_label='device_id')