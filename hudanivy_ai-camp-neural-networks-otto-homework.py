import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('../input/train.csv',sep=',')
data_test=pd.read_csv('../input/test.csv',sep=',')
columns=data.columns[1:-1]
X = data[columns]
y = np.ravel(data['target'])
data.target.value_counts().plot(kind='bar')

fig, axes = plt.subplots(3, 3, figsize=(10, 6),gridspec_kw=dict(hspace=0.5, wspace=0.4))  
for i, ax in enumerate(axes.flat):
    ax.hist(data[data.target=='Class_'+str(i+1)].feat_20)
    ax.set_title('feat_20 in Class_'+str(i+1))
plt.scatter(data['feat_19'],data['feat_20'])
#X.corr()
sns.heatmap(X.corr(), square=True)
num_fea = X.shape[1]
"""
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(MLPClassifier(), scoring="neg_log_loss", cv=3, verbose=3,
                 param_grid={"solver":['lbfgs','sgd','adam'],"activation": ['logistic','relu'],"alpha":[1e-3],"hidden_layer_sizes":[(20,30)]}, )
gs.fit(X, y)
print("best params",gs.best_params_,"best scores:",gs.best_score_)
"""
model = MLPClassifier(solver='lbfgs',activation='relu', alpha=1e-3, hidden_layer_sizes = (20, 30), random_state = 1, verbose = False)
model.fit(X, y)
model.intercepts_

print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)
pred = model.predict(X)

metrics.accuracy_score(pred, y)
data_test[columns[:]].head()
pred_test_proba=model.predict_proba(data_test[columns[:]])
output=pd.DataFrame(pred_test_proba,columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
output.insert(0,'id',data_test.id)
output.head()
output.to_csv('./my_otto_prediction.csv', index = False)
