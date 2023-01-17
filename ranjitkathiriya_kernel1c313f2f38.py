import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as plt

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

cf.go_offline()

from plotly.subplots import make_subplots

import plotly.express as px

import plotly.graph_objects as go



import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold
df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

df_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
df_train.info()
df_test.info()
df_train.describe()
df_train.head()
df_test.head()
sns.pairplot(df_train,hue="Sex")
sns.pairplot(df_train,hue="SmokingStatus")
sns.boxplot(x='SmokingStatus',y='Age',data=df_train,palette='rainbow')
sns.countplot(df_train['Sex'])
sns.countplot(df_train['SmokingStatus'])
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.countplot(df_train['Age'],hue=df_train['Sex'])
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.countplot(df_train['Age'],hue=df_train['SmokingStatus'])
df_train.columns
# sns.barplot(x=df_train.Percent,y=df_train.FVC,hue=df_train.Sex)

sns.jointplot(x=df_train.Percent,y=df_train.FVC,data=df_train)
sns.scatterplot(x=df_train.Percent,y=df_train.FVC,data=df_train,hue='Sex')
sns.scatterplot(x=df_train.Percent,y=df_train.FVC,data=df_train,hue='SmokingStatus')
parallel_diagram = df_train[['Weeks', 'Patient', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']]



fig = px.parallel_categories(parallel_diagram, color_continuous_scale=px.colors.sequential.Inferno)

fig.update_layout(title='Parallel category diagram on trainset')

fig.show()
df_train.info()
df_train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
df_submission = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
df_submission.head()
df_submission['Patient'] = df_submission['Patient_Week'].apply(lambda x:x.split('_')[0])
df_submission.head(1)
df_submission['Weeks'] = df_submission['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
df_submission.head(1)
df_submission =  df_submission[['Patient','Weeks','Confidence','Patient_Week']]
df_submission.head()


df_submission = df_submission.merge(df_test.drop('Weeks', axis=1), on="Patient")
df_submission.head()
df_train['WHERE'] = 'train'

df_test['WHERE'] = 'val'

df_submission['WHERE'] = 'test'

data = df_train.append([df_test, df_submission])
data.head()
print(df_train.shape, df_test.shape, df_submission.shape, data.shape)

print(df_train.Patient.nunique(), df_test.Patient.nunique(), df_submission.Patient.nunique(), 

      data.Patient.nunique())
data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
data.head()
base = data.loc[data.Weeks == data.min_week]

base = base[['Patient','FVC']].copy()

base.columns = ['Patient','min_FVC']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)
# base.head() 
data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base
data.head()
COLS = ['Sex','SmokingStatus'] #,'Age'

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)

#=================
data.head()
data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','percent','week','BASE']
tr = data.loc[data.WHERE=='train']

chunk = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']

del data
tr.shape, chunk.shape, sub.shape
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

#=============================#

def score(y_true, y_pred):

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return K.mean(metric)

#============================#

def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)

#=============================#

def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss

#=================

def make_model(nh):

    z = L.Input((nh,), name="Patient")

    x = L.Dense(100, activation="relu", name="d1")(z)

    x = L.Dense(100, activation="relu", name="d2")(x)

    #x = L.Dense(100, activation="relu", name="d3")(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)

    p2 = L.Dense(3, activation="relu", name="p2")(x)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                     name="preds")([p1, p2])

    

    model = M.Model(z, preds, name="CNN")

    #model.compile(loss=qloss, optimizer="adam", metrics=[score])

    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])

    return model
y = tr['FVC'].values

z = tr[FE].values

ze = sub[FE].values

nh = z.shape[1]

pe = np.zeros((ze.shape[0], 3))

pred = np.zeros((z.shape[0], 3))
net = make_model(nh)

print(net.summary())

print(net.count_params())
NFOLD = 5

kf = KFold(n_splits=NFOLD)
tf.cast(y, tf.float32)
z = z.astype(np.float32)

y = y.astype(np.float32)
%%time

cnt = 0

EPOCHS = 800

for tr_idx, val_idx in kf.split(z):

    cnt += 1

    print(f"FOLD {cnt}")

    net = make_model(nh)

    net.fit(z[tr_idx], y[tr_idx], batch_size=128, epochs=500, 

            validation_data=(z[val_idx], y[val_idx]), verbose=0) #

    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=128))

    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=128))

    print("predict val...")

    pred[val_idx] = net.predict(z[val_idx], batch_size=128, verbose=0)

    print("predict test...")

    pe += net.predict(ze, batch_size=128, verbose=0) / NFOLD
sigma_opt = mean_absolute_error(y, pred[:, 1])

unc = pred[:,2] - pred[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)
# idxs = np.random.randint(0, y.shape[0], 100)

# plt.plot(y[idxs], label="ground truth")

# plt.plot(pred[idxs, 0], label="q25")

# plt.plot(pred[idxs, 1], label="q50")

# plt.plot(pred[idxs, 2], label="q75")

# plt.legend(loc="best")

# plt.show()
print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())
sub.head()
sub['FVC1'] = 0.996*pe[:, 1]

sub['Confidence1'] = pe[:, 2] - pe[:, 0]
subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
subm.loc[~subm.FVC1.isnull()].head(10)
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

if sigma_mean<70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']
subm.head()
subm.describe().T
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
subm[["Patient_Week","FVC","Confidence"]].to_csv("../working/submission.csv", index=False)