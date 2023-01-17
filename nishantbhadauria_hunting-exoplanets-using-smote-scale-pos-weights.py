import pandas as pd

exo_train=pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')

exo_test=pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')

train_exo_y=exo_train[exo_train['LABEL'] >1 ]

train_exo_n=exo_train[exo_train['LABEL'] < 2]

train_t_n=train_exo_n.iloc[:,1:].T

train_t_y=train_exo_y.iloc[:,1:].T

train_t_n.head(1)

exo_train['LABEL'].value_counts()
import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2,subplot_titles=("Flux variation of star 37", "Flux variation of star 5086", 

                                                   "Flux variation of star 3000", "Flux variation of star 3001"))

fig.add_trace(

    go.Scatter(y=train_t_n[37], x=train_t_n.index),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(y=train_t_n[5086], x=train_t_n.index),

    row=1, col=2

)

fig.add_trace(

    go.Scatter(y=train_t_n[3000], x=train_t_n.index),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(y=train_t_n[3001], x=train_t_n.index),

    row=2, col=2

)

fig.update_layout(height=600, width=800, title_text="Non Exoplanets Star examples",showlegend=False)

fig.show()
fig = make_subplots(rows=2, cols=2,subplot_titles=("Flux variation of star 0", "Flux variation of star 1", 

                                                   "Flux variation of star 35", "Flux variation of star 36"))

fig.add_trace(

    go.Scatter(y=train_t_y[0], x=train_t_y.index),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(y=train_t_y[1], x=train_t_y.index),

    row=1, col=2

)

fig.add_trace(

    go.Scatter(y=train_t_y[35], x=train_t_y.index),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(y=train_t_y[36], x=train_t_y.index),

    row=2, col=2

)

fig.update_layout(height=600, width=800, title_text="Exoplanets Stars examples",showlegend=False)
###Normalizing the flux#####

from sklearn.preprocessing import StandardScaler

trainx=exo_train.iloc[:,1:]

textx=exo_test.iloc[:,1:]

scaler=StandardScaler()

train_scaled=scaler.fit_transform(trainx)

test_scaled=scaler.fit_transform(textx)
### Applying SVC with linear Kernel

trainy=exo_train[['LABEL']]

testy=exo_test[['LABEL']]

from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')

svclassifier.fit(train_scaled, trainy['LABEL'])

y_pred = svclassifier.predict(test_scaled)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(testy, y_pred))

print(classification_report(testy, y_pred))
####Polynomial Kernel ###

svclassifier = SVC(kernel='poly')

svclassifier.fit(train_scaled, trainy['LABEL'])

y_pred = svclassifier.predict(test_scaled)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(testy, y_pred))

print(classification_report(testy, y_pred))
### RBF kernel###

svclassifier = SVC(kernel='rbf')

svclassifier.fit(train_scaled, trainy['LABEL'])

y_pred = svclassifier.predict(test_scaled)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(testy, y_pred))

print(classification_report(testy, y_pred))
###Sigmoid kernel###

svclassifier = SVC(kernel='sigmoid')

svclassifier.fit(train_scaled, trainy['LABEL'])

y_pred = svclassifier.predict(test_scaled)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(testy, y_pred))

print(classification_report(testy, y_pred))
import numpy as np

from sklearn.decomposition import PCA

pca = PCA(n_components=6)

pca.fit(train_scaled)

PCA(n_components=6)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

trns_x=pca.transform(train_scaled)

trns_y=pca.transform(test_scaled)

testy
##Applying SVC RBF to new transformed dataset #####

from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf')

svclassifier.fit(trns_x, trainy['LABEL'])

y_pred = svclassifier.predict(trns_y)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(testy['LABEL'], y_pred))

print(classification_report(testy['LABEL'], y_pred))
trainy.loc[trainy['LABEL'] == 1, 'new1'] = 0

trainy.loc[trainy['LABEL'] > 1, 'new1'] = 1

testy.loc[testy['LABEL'] > 1, 'new1'] = 1

testy.loc[testy['LABEL'] == 1, 'new1'] = 0

import statsmodels.api as sm

logit_model=sm.Logit(trainy['new1'],trns_x)

result=logit_model.fit()

print(result.summary2())
###Scikitlearn Logistic regression ###

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(trns_x, trainy['new1'])

y_pred = logreg.predict(trns_y)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(trns_y, testy['new1'])))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

logit_roc_auc = roc_auc_score(testy['new1'], logreg.predict(trns_y))

fpr, tpr, thresholds = roc_curve(testy['new1'], logreg.predict_proba(trns_y)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from imblearn.over_sampling import SMOTE

over = SMOTE(random_state=0)

ov_train_x,ov_train_y=over.fit_sample(trns_x, trainy['new1'])

ov_train_y=ov_train_y.astype('int')

ov_train_y.value_counts()

ov_train_y=ov_train_y.values.tolist()

from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf')

svclassifier.fit(ov_train_x, ov_train_y)

y_pred = svclassifier.predict(trns_y)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(testy['new1'], y_pred))

print(classification_report(testy['new1'], y_pred))
import numpy as np

ov_train_y=np.array(ov_train_y)

ov_train_y.dtype

from sklearn.model_selection import train_test_split

tr_x,v_x,tr_y,V_y= train_test_split(ov_train_x, ov_train_y, test_size=0.2)
import tensorflow as tf

from tensorflow import keras

from keras.layers import LeakyReLU

model=keras.models.Sequential([

    keras.layers.Dense(300,activation="selu",input_shape=(8080,6)),    

    keras.layers.Dense(200,activation="selu",kernel_regularizer=keras.regularizers.l2(0.01)),   

    keras.layers.Dense(100,activation="selu",kernel_regularizer=keras.regularizers.l2(0.01)),   

    keras.layers.Dense(2,activation="softmax")

])

epochs=50

optimizers=keras.optimizers.SGD(clipvalue=1.0)

def exp_decay(lr0,s):

    def exp_decay_fn(epcohs):

        return lr0*0.1**(epochs/s)

    return exp_decay_fn



exp_decay_fn=exp_decay(lr0=0.1,s=20)

lr_sch=keras.callbacks.LearningRateScheduler(exp_decay_fn)

lr_sch2=keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=5)

model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizers,metrics=["accuracy"])

history=model.fit(tr_x,tr_y,epochs=50,callbacks=[lr_sch],validation_data=(v_x,V_y))

import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot()

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()

predict=model.predict_classes(trns_y)

print(confusion_matrix(testy['new1'], predict))

print(classification_report(testy['new1'], predict))
model=keras.models.Sequential([

    keras.layers.Dense(300,activation="swish",input_shape=(8080,6)),    

    keras.layers.Dense(200,activation="swish",kernel_initializer="he_normal"),   

    keras.layers.Dense(100,activation="swish",kernel_initializer="he_normal"),    

    keras.layers.Dense(2,activation="softmax")

])



optimizers=tf.keras.optimizers.Adam(

    learning_rate=0.001,

    beta_1=0.9,

    beta_2=0.999,

    epsilon=1e-07,

    amsgrad=False,

    name="Adam"

    

)

lr_sch2=keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=5)

model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizers,metrics=["accuracy"])

history=model.fit(tr_x,tr_y,epochs=50,validation_data=(v_x,V_y),callbacks=[lr_sch2])

import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot()

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()

predict=model.predict_classes(trns_y)

print(confusion_matrix(testy['new1'], predict))

print(classification_report(testy['new1'], predict))
# T - no. of total samples

# P - no. of positive samples

# scale_pos_weight = percent of negative / percent of positive

# which translates to:

# scale_pos_weight = (100*(T-P)/T) / (100*P/T)

# which further simplifies to beautiful:

#scale_pos_weight = |37/5387 - 1|=0.99

from xgboost import XGBClassifier

scale_pos_weight = [2,0.99,0.60,0.50,0.33,0.20,0.10]

for i in scale_pos_weight:

    print('scale_pos_weight = {}: '.format(i))

    clf = XGBClassifier(scale_pos_weight=i)

    clf.fit(ov_train_x, ov_train_y)

    predict = clf.predict(trns_y)    

    cm = confusion_matrix(testy['new1'], predict)  

    auc=metrics.roc_auc_score(testy['new1'], predict)

    print('Confusion Matrix: \n', cm)

    print('metrics: \n',classification_report(testy['new1'], predict))

    print('AUC of test set: {:.2f} \n'.format(metrics.roc_auc_score(testy['new1'], predict))) 