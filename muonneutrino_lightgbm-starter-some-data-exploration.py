import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
folder = '../input/'
train = pd.read_csv(folder+'training.csv', index_col=0)
test = pd.read_csv(folder+'test.csv',index_col=0)
#check_correlation = pd.read_csv(folder+'check_correlation.csv', index_col='id')
#check_agreement = pd.read_csv(folder+'check_agreement.csv', index_col='id')

train.head()
train.info()
train.signal.value_counts()
signal = train[train.signal==1]
nonsignal = train[train.signal==0]
plt.hist(signal.pt/1000, range=(0,25), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.pt/1000, range=(0,25), bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel('Transverse Momentum [GeV?]')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()
plt.hist(signal.FlightDistance, range=(0,100), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.FlightDistance, range=(0,100), bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel('Flight Distance')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()
plt.hist(signal.dira, range=(0.998,1), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.dira,range=(0.999,1),bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel('Dira')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()
plt.hist(signal.VertexChi2, range=(0,16), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.VertexChi2,range=(0,16),bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel(r'Vertex $\chi^2$')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()
plt.hist(signal.IP, range=(0,0.5), bins=100, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.IP,range=(0,0.5),bins=100, label='Background',alpha=0.7,normed=True)
plt.xlabel('Impact Paraneter')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()
plt.hist(signal.iso, range=(0,20), bins=20, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.iso,range=(0,20),bins=20, label='Background',alpha=0.7,normed=True)
plt.xlabel('Track Isolation Variable')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()
plt.hist(signal.LifeTime, range=(0,1e-2), bins=50, label='Signal', alpha=0.7,normed=True)
plt.hist(nonsignal.LifeTime,range=(0,1e-2),bins=50, label='Background',alpha=0.7,normed=True)
plt.xlabel('Lifetime')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.show()
import lightgbm as lgb
feature_names = ['FlightDistance', 'LifeTime', 'pt', 'IP']
features = train[feature_names]
target = train['signal']
train_set = lgb.Dataset(features,train.signal)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 31,
    'metric': {'auc'},
    'learning_rate': 0.01,
#    'feature_fraction': 0.8,
#    'bagging_fraction': 0.8     
    
}
cv_output = lgb.cv(
    params,
    train_set,
    num_boost_round=400,
    nfold=10,
)
best_niter = np.argmax(cv_output['auc-mean'])
best_score = cv_output['auc-mean'][best_niter]
print('Best number of iterations: {}'.format(best_niter))
print('Best CV score: {}'.format(best_score))
model = lgb.train(params, train_set, num_boost_round=best_niter)
test_features = test[feature_names]
predictions = model.predict(test_features)
test['prediction'] = predictions
test[['prediction']].to_csv('lightgbm_starter.csv')
