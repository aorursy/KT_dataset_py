import numpy as np

import pandas as pd

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go

import matplotlib.pyplot as plt



import seaborn as sns



from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier



import optuna

from optuna.samplers import TPESampler



from sklearn.cluster import KMeans

from sklearn.manifold import TSNE



!pip install pyod



from pyod.models.copod import COPOD



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense

from sklearn.model_selection import KFold

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split



from lightgbm import LGBMClassifier

from xgboost import XGBClassifier
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
train
train = train.drop(['id'], axis=1)

test = test.drop(['id'], axis=1)
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Male', 'Female'], 

        y=[

            len(train[train['Gender']=='Male']),

            len(train[train['Gender']=='Female'])

        ], 

        name='Train Gender',

        text = [

            str(round(100 * len(train[train['Gender']=='Male']) / len(train), 2)) + '%',

            str(round(100 * len(train[train['Gender']=='Female']) / len(train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['Male', 'Female'], 

        y=[

            len(test[test['Gender']=='Male']),

            len(test[test['Gender']=='Female'])

        ], 

        name='Test Gender',

        text=[

            str(round(100 * len(test[test['Gender']=='Male']) / len(test), 2)) + '%',

            str(round(100 * len(test[test['Gender']=='Female']) / len(test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test gender column',

    height=400,

    width=700

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(train[train['Driving_License']==1]),

            len(train[train['Driving_License']==0])

        ], 

        name='Train Driving_License',

        text = [

            str(round(100 * len(train[train['Driving_License']==1]) / len(train), 2)) + '%',

            str(round(100 * len(train[train['Driving_License']==0]) / len(train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(test[test['Driving_License']==1]),

            len(test[test['Driving_License']==0])

        ], 

        name='Test Driving_License',

        text=[

            str(round(100 * len(test[test['Driving_License']==1]) / len(test), 2)) + '%',

            str(round(100 * len(test[test['Driving_License']==0]) / len(test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Driving_License column',

    height=400,

    width=700

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(train[train['Previously_Insured']==1]),

            len(train[train['Previously_Insured']==0])

        ], 

        name='Train Previously_Insured',

        text = [

            str(round(100 * len(train[train['Previously_Insured']==1]) / len(train), 2)) + '%',

            str(round(100 * len(train[train['Previously_Insured']==0]) / len(train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(test[test['Previously_Insured']==1]),

            len(test[test['Previously_Insured']==0])

        ], 

        name='Test Previously_Insured',

        text = [

            str(round(100 * len(test[test['Previously_Insured']==1]) / len(test), 2)) + '%',

            str(round(100 * len(test[test['Previously_Insured']==0]) / len(test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Previously_Insured column',

    height=400,

    width=700

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(train[train['Vehicle_Damage']=='Yes']),

            len(train[train['Vehicle_Damage']=='No'])

        ], 

        name='Train Vehicle_Damage',

        text = [

            str(round(100 * len(train[train['Vehicle_Damage']=='Yes']) / len(train), 2)) + '%',

            str(round(100 * len(train[train['Vehicle_Damage']=='No']) / len(train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(test[test['Vehicle_Damage']=='Yes']),

            len(test[test['Vehicle_Damage']=='No'])

        ], 

        name='Test Vehicle_Damage',

        text = [

            str(round(100 * len(test[test['Vehicle_Damage']=='Yes']) / len(test), 2)) + '%',

            str(round(100 * len(test[test['Vehicle_Damage']=='No']) / len(test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Vehicle_Damage column',

    height=400,

    width=700

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['> 2 Years', '1-2 Year', '< 1 Year'], 

        y=[

            len(train[train['Vehicle_Age']=='> 2 Years']),

            len(train[train['Vehicle_Age']=='1-2 Year']),

            len(train[train['Vehicle_Age']=='< 1 Year'])

        ], 

        name='Train Vehicle_Age',

        text = [

            str(round(100 * len(train[train['Vehicle_Age']=='> 2 Years']) / len(train), 2)) + '%',

            str(round(100 * len(train[train['Vehicle_Age']=='1-2 Year']) / len(train), 2)) + '%',

            str(round(100 * len(train[train['Vehicle_Age']=='< 1 Year']) / len(train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['> 2 Years', '1-2 Year', '< 1 Year'], 

        y=[

            len(test[test['Vehicle_Age']=='> 2 Years']),

            len(test[test['Vehicle_Age']=='1-2 Year']),

            len(test[test['Vehicle_Age']=='< 1 Year'])

        ], 

        name='Test Vehicle_Age',

        text = [

            str(round(100 * len(test[test['Vehicle_Age']=='> 2 Years']) / len(test), 2)) + '%',

            str(round(100 * len(test[test['Vehicle_Age']=='1-2 Year']) / len(test), 2)) + '%',

            str(round(100 * len(test[test['Vehicle_Age']=='< 1 Year']) / len(test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Vehicle_Age column',

    height=400,

    width=700

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Histogram(

        x=train['Age'], 

        name='Train Age'

    ),

    go.Histogram(

        x=test['Age'], 

        name='Test Age'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Age column distribution',

    height=500,

    width=900

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Histogram(

        x=train['Annual_Premium'], 

        name='Train Annual_Premium'

    ),

    go.Histogram(

        x=test['Annual_Premium'], 

        name='Test Annual_Premium'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Annual_Premium column distribution',

    height=500,

    width=800

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Histogram(

        x=train['Policy_Sales_Channel'], 

        name='Train Policy_Sales_Channel'

    ),

    go.Histogram(

        x=test['Policy_Sales_Channel'], 

        name='Test Policy_Sales_Channel'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Policy_Sales_Channel column distribution',

    height=500,

    width=800

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Histogram(

        x=train['Vintage'], 

        name='Train Vintage'

    ),

    go.Histogram(

        x=test['Vintage'], 

        name='Test Vintage'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Vintage column distribution',

    height=500,

    width=800

)



fig.show()
tr = train['Region_Code'].value_counts().reset_index()

x_tr = tr['index'].tolist()

y_tr = tr['Region_Code'].tolist()

te = test['Region_Code'].value_counts().reset_index()

x_te = te['index'].tolist()

y_te = te['Region_Code'].tolist()



fig = make_subplots(rows=2, cols=1)



traces = [

    go.Bar(

        x=x_tr, 

        y=y_tr, 

        name='Train Region_Code'

    ),

    go.Bar(

        x=x_te, 

        y=y_te, 

        name='Test Region_Code'

    )

]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 1) + 1, (i % 1)  +1)



fig.update_layout(

    title_text='Train / test Region_Code',

    height=900,

    width=800

)



fig.show()
fig = make_subplots(rows=1, cols=1)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(train[train['Response']==1]),

            len(train[train['Response']==0])

        ], 

        name='Train Response'

    ),

]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train Response column',

    height=400,

    width=400

)



fig.show()
fig = px.histogram(

    train, 

    "Age", 

    color='Response',

    nbins=100, 

    title='Age & Response ditribution', 

    width=700,

    height=500

)



fig.show()
fig = px.histogram(

    train[train['Response'] == 1], 

    "Age", 

    nbins=100, 

    title='Age distribution for positive response', 

    width=700,

    height=500

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Declined', 'Accepted'], 

        y=[

            len(train[(train['Gender']=='Male') & (train['Response']==0)]),

            len(train[(train['Gender']=='Male') & (train['Response']==1)])

        ], 

        name='Gender: Male'

    ),

    go.Bar(

        x=['Declined', 'Accepted'],  

        y=[

            len(train[(train['Gender']=='Female') & (train['Response']==0)]),

            len(train[(train['Gender']=='Female') & (train['Response']==1)])

        ], 

        name='Gender: Female'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train gender/response dependencies',

    height=400,

    width=700

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Declined', 'Accepted'], 

        y=[

            len(train[(train['Previously_Insured']==0) & (train['Response']==0)]),

            len(train[(train['Previously_Insured']==0) & (train['Response']==1)])

        ], 

        name='Previously_Insured: Previously Not Insured'

    ),

    go.Bar(

        x=['Declined', 'Accepted'],  

        y=[

            len(train[(train['Previously_Insured']==1) & (train['Response']==0)]),

            len(train[(train['Previously_Insured']==1) & (train['Response']==1)])

        ], 

        name='Previously_Insured: Previously Insured'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train previously_insured/response dependencies',

    height=400,

    width=700

)



fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Declined', 'Accepted'], 

        y=[

            len(train[(train['Vehicle_Damage']=='No') & (train['Response']==0)]),

            len(train[(train['Vehicle_Damage']=='No') & (train['Response']==1)])

        ], 

        name='Vehicle_Damage: No'

    ),

    go.Bar(

        x=['Declined', 'Accepted'],  

        y=[

            len(train[(train['Vehicle_Damage']=='Yes') & (train['Response']==0)]),

            len(train[(train['Vehicle_Damage']=='Yes') & (train['Response']==1)])

        ], 

        name='Vehicle_Damage: Yes'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train vehicle_damage/response dependencies',

    height=400,

    width=700

)



fig.show()
fig = make_subplots(rows=1, cols=3)



traces = [

    go.Bar(

        x=['Declined', 'Accepted'], 

        y=[

            len(train[(train['Vehicle_Age']=='> 2 Years') & (train['Response']==0)]),

            len(train[(train['Vehicle_Age']=='> 2 Years') & (train['Response']==1)])

        ], 

        name='Vehicle_Age: > 2 Years'

    ),

    go.Bar(

        x=['Declined', 'Accepted'], 

        y=[

            len(train[(train['Vehicle_Age']=='1-2 Year') & (train['Response']==0)]),

            len(train[(train['Vehicle_Age']=='1-2 Year') & (train['Response']==1)])

        ], 

        name='Vehicle_Age: 1-2 Year'

    ),

    go.Bar(

        x=['Declined', 'Accepted'], 

        y=[

            len(train[(train['Vehicle_Age']=='< 1 Year') & (train['Response']==0)]),

            len(train[(train['Vehicle_Age']=='< 1 Year') & (train['Response']==1)])

        ], 

        name='Vehicle_Age: < 1 Year'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 3) + 1, (i % 3)  +1)



fig.update_layout(

    title_text='Train/test Vehicle_Age/Response dependencies',

    height=400,

    width=800

)



fig.show()
fig = px.histogram(

    train, 

    "Annual_Premium", 

    color='Response',

    nbins=100, 

    title='Annual_Premium & Response ditribution', 

    width=700,

    height=500

)

fig.show()
fig = px.histogram(

    train[train['Response'] == 1], 

    "Annual_Premium", 

    nbins=100, 

    title='Annual_Premium distribution for positive response', 

    width=700,

    height=500

)



fig.show()
fig = px.histogram(

    train, 

    "Vintage", 

    color='Response',

    nbins=100, 

    title='Vintage & Response ditribution', 

    width=700,

    height=500

)



fig.show()
fig = px.histogram(

    train[train['Response'] == 1], 

    "Vintage", 

    nbins=100, 

    title='Vintage distribution for positive response', 

    width=700,

    height=500

)

fig.show()
train.loc[train['Gender'] == 'Male', 'Gender'] = 1

train.loc[train['Gender'] == 'Female', 'Gender'] = 0

test.loc[test['Gender'] == 'Male', 'Gender'] = 1

test.loc[test['Gender'] == 'Female', 'Gender'] = 0



train.loc[train['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2

train.loc[train['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1

train.loc[train['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0

test.loc[test['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2

test.loc[test['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1

test.loc[test['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0



train.loc[train['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1

train.loc[train['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0

test.loc[test['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1

test.loc[test['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0
for col in train.columns:

    train[col] = train[col].astype(np.int32)



train
f = plt.figure(figsize=(13, 11))

plt.matshow(train.corr(), fignum=f.number)

plt.xticks(range(train.shape[1]), train.columns, fontsize=14, rotation=75)

plt.yticks(range(train.shape[1]), train.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
for col in train.columns:

    if col == 'Response':

        continue

    print(col, train[col].corr(train['Response']))
fig = px.scatter(

    train, 

    x="Annual_Premium", 

    y="Age", 

    color="Response",

    width=600,

    height=600,

    title='Annual_premium vs Age scatter'

)

fig.show()
X = train.drop(['Response'], axis=1)

y = train['Response']
kmeans = KMeans(n_clusters=2, random_state=666).fit(X)
train['cluster'] = kmeans.labels_

train
train['cluster'].value_counts()
print('Kmeans accuracy: ', accuracy_score(train['Response'], train['cluster']))

print('Kmeans f1_score: ', f1_score(train['Response'], train['cluster']))
response = train['Response']

train = train.drop(['Response', 'cluster'], axis=1)
clf = COPOD(contamination=0.15)

clf.fit(train)
cluster = clf.predict(train)

train['cluster'] = cluster

train['Response'] = response

train
train['cluster'].value_counts()
print('COPOD accuracy: ', accuracy_score(train['Response'], train['cluster']))

print('COPOD f1_score: ', f1_score(train['Response'], train['cluster']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
print('Positive cases % in validation set: ', round(100 * len(y_test[y_test == 1]) / len(y_test), 3), '%')

print('Positive cases % in train set: ', round(100 * len(y_train[y_train == 1]) / len(y_train), 3), '%')
model = LogisticRegression(random_state=666)

model.fit(X_train, y_train)
preds = model.predict(X_test)

print('Simple Logistic Regression accuracy: ', accuracy_score(y_test, preds))

print('Simple Logistic Regression f1_score: ', f1_score(y_test, preds))
def plot_confusion_matrix(y_real, y_pred):

    cm = confusion_matrix(y_real, y_pred)



    ax= plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax, fmt='g')



    ax.set_xlabel('Predicted labels')

    ax.set_ylabel('True labels')
plot_confusion_matrix(y_test, preds)
X_train = X_train.drop(['Region_Code', 'Vintage', 'Driving_License'], axis=1)

X_test = X_test.drop(['Region_Code', 'Vintage', 'Driving_License'], axis=1)
model = LogisticRegression(random_state=666)

model.fit(X_train, y_train)
preds = model.predict(X_test)

print('Simple Logistic Regression accuracy: ', accuracy_score(y_test, preds))

print('Simple Logistic Regression f1_score: ', f1_score(y_test, preds))
plot_confusion_matrix(y_test, preds)
model = LGBMClassifier(random_state=666)

model.fit(X_train, y_train)



preds = model.predict(X_test)

print('Simple LGBM accuracy: ', accuracy_score(y_test, preds))

print('Simple LGBM Regression f1_score: ', f1_score(y_test, preds))
np.random.seed(666)

sampler = TPESampler(seed=0)



def create_model(trial):

    max_depth = trial.suggest_int("max_depth", 2, 20)

    n_estimators = trial.suggest_int("n_estimators", 1, 400)

    learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)

    gamma = trial.suggest_uniform('gamma', 0.0000001, 1)

    scale_pos_weight = trial.suggest_int("scale_pos_weight", 1, 20)

    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, gamma=gamma, scale_pos_weight=scale_pos_weight, random_state=0)

    return model



def objective(trial):

    model = create_model(trial)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    score = f1_score(y_test, preds)

    return score



#study = optuna.create_study(direction="maximize", sampler=sampler)

#study.optimize(objective, n_trials=500)



#xgb_params = study.best_params

xgb_params = {'max_depth': 4, 'n_estimators': 372, 'learning_rate': 0.09345905554110154, 'gamma': 0.6641238000625036, 'scale_pos_weight': 4}

xgb_params['random_state'] = 0

xgb = XGBClassifier(**xgb_params)

xgb.fit(X_train, y_train)

preds = xgb.predict(X_test)

print('Optimized XGBClassifier accuracy: ', accuracy_score(y_test, preds))

print('Optimized XGBClassifier f1-score', f1_score(y_test, preds))
plot_confusion_matrix(y_test, preds)
def create_model(trial):

    max_depth = trial.suggest_int("max_depth", 2, 20)

    n_estimators = trial.suggest_int("n_estimators", 2, 300)

    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestClassifier(min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, max_depth=max_depth, random_state=0)

    return model



def objective(trial):

    model = create_model(trial)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    score = f1_score(y_test, preds)

    return score



# study = optuna.create_study(direction="maximize", sampler=sampler)

# study.optimize(objective, n_trials=100)

# rf_params = study.best_params

rf_params = {'max_depth': 20, 'n_estimators': 14, 'min_samples_leaf': 1}

rf_params['random_state'] = 0

rf = RandomForestClassifier(**rf_params)

rf.fit(X_train, y_train)

preds = rf.predict(X_test)

print('Optimized RF accuracy: ', accuracy_score(y_test, preds))

print('Optimized RF f1-score:', f1_score(y_test, preds))
plot_confusion_matrix(y_test, preds)
def recall_score(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_score(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def keras_f1_score(y_true, y_pred):

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(7),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(30, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(2, activation='softmax')

    ])

    model.compile(

        loss=tf.keras.losses.binary_crossentropy, 

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),

        metrics=[keras_f1_score]

    )

    return model
y_nn_train = to_categorical(y_train)
class_weight = {

    0: 1.,

    1: 8.

}
model = create_model()

model.fit(X_train, y_nn_train, validation_split=0.2, epochs=30, batch_size=128, verbose=2, class_weight=class_weight)
preds = model.predict(X_test)

preds = np.argmax(preds, axis=1)
print('NN accuracy: ', accuracy_score(y_test, preds))

print('NN f1-score', f1_score(y_test, preds))
plot_confusion_matrix(y_test, preds)