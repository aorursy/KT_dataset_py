import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/avhranalytics/train_jqd04QH.csv')

test = pd.read_csv('/kaggle/input/avhranalytics/test_KaymcHn.csv')
train.head()
train.dtypes
train.shape, test.shape
combine = train.append(test)

combine.shape
combine.isnull().sum()
combine['company_size'].value_counts()
combine['company_size'].fillna("Unknown", inplace=True)

combine['company_size'] = combine['company_size'].replace('50-99','CS_Tier3')

combine['company_size'] = combine['company_size'].replace('100-500','CS_Tier4')

combine['company_size'] = combine['company_size'].replace('10/49','CS_Tier2')

combine['company_size'] = combine['company_size'].replace('10000+','CS_Tier8')

combine['company_size'] = combine['company_size'].replace('1000-4999','CS_Tier6')

combine['company_size'] = combine['company_size'].replace('<10','CS_Tier1')

combine['company_size'] = combine['company_size'].replace('500-999','CS_Tier5')

combine['company_size'] = combine['company_size'].replace('5000-9999','CS_Tier7')

combine['company_size'].value_counts()
combine['gender'].value_counts()
combine['gender'].fillna("Unknown", inplace=True)

combine['gender'].value_counts()
combine['relevent_experience'].value_counts()
combine['relevent_experience'].fillna("Unknown", inplace=True)

combine['relevent_experience'] = combine['relevent_experience'].replace('Has relevent experience','RE_Yes')

combine['relevent_experience'] = combine['relevent_experience'].replace('No relevent experience','RE_No')

combine['relevent_experience'].value_counts()
combine['enrolled_university'].value_counts()
combine['enrolled_university'].fillna("Unknown", inplace=True)

combine['enrolled_university'] = combine['enrolled_university'].replace('no_enrollment','No')

combine['enrolled_university'] = combine['enrolled_university'].replace('Full time course','Full_Time')

combine['enrolled_university'] = combine['enrolled_university'].replace('Part time course','Part_Time')

combine['enrolled_university'].value_counts()
combine['education_level'].value_counts()
combine['education_level'].fillna("0", inplace=True)

combine['education_level'] = combine['education_level'].replace('Graduate','3')

combine['education_level'] = combine['education_level'].replace('Masters','4')

combine['education_level'] = combine['education_level'].replace('High School','2')

combine['education_level'] = combine['education_level'].replace('Phd','5')

combine['education_level'] = combine['education_level'].replace('Primary School','1')

combine['education_level'] = combine['education_level'].astype('int')

combine['education_level'].value_counts()
combine['major_discipline'].value_counts()
combine['major_discipline'].fillna("Unknown", inplace=True)

combine['major_discipline'] = combine['major_discipline'].replace('Business Degree','Business_Degree')

combine['major_discipline'] = combine['major_discipline'].replace('No Major','No_Major')

combine['major_discipline'].value_counts()
combine['experience'].value_counts()
combine['experience'].fillna("-1", inplace=True)

combine['experience'] = combine['experience'].replace('>20','21')

combine['experience'] = combine['experience'].replace('<1','0')

combine['experience'] = combine['experience'].astype('int')

combine['experience'].value_counts()
bins= [-1,0,3,6,9,12,15,18,21]

labels = ['Unknown','Exp_Tier1','Exp_Tier2','Exp_Tier3','Exp_Tier4','Exp_Tier5','Exp_Tier6','Exp_Tier7']

combine['experience'] = pd.cut(combine['experience'], bins=bins, labels=labels, right=False)

combine['experience'].value_counts()
combine['company_type'].value_counts()
combine['company_type'].fillna("Unknown", inplace=True)

combine['company_type'] = combine['company_type'].replace('Pvt Ltd','Pvt_Ltd')

combine['company_type'] = combine['company_type'].replace('Funded Startup','Funded_Startup')

combine['company_type'] = combine['company_type'].replace('Public Sector','Public_Sector')

combine['company_type'] = combine['company_type'].replace('Early Stage Startup','Early_Stage_Startup')

combine['company_type'].value_counts()
combine['last_new_job'].value_counts()
combine['last_new_job'].fillna("-1", inplace=True)

combine['last_new_job'] = combine['last_new_job'].replace('>4','5')

combine['last_new_job'] = combine['last_new_job'].replace('never','0')

combine['last_new_job'] = combine['last_new_job'].astype('int')

combine['last_new_job'].value_counts()
combine['training_hours'].describe()
combine['training_hours'] = np.log(combine['training_hours'])

combine['training_hours'].describe()
combine['city_development_index'].describe()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

combine['city'] = encoder.fit_transform(combine['city'])
combine.dtypes
train_cleaned = combine[combine['target'].isnull()!=True].drop(['enrollee_id'], axis=1)
combine = pd.get_dummies(combine)

combine.shape
X = combine[combine['target'].isnull()!=True].drop(['enrollee_id','target'], axis=1)

y = combine[combine['target'].isnull()!=True]['target']



X_test = combine[combine['target'].isnull()==True].drop(['enrollee_id','target'], axis=1)



X.shape, y.shape, X_test.shape
train_cleaned.head()
import seaborn as sns

import plotly.io as pio

import plotly.graph_objects as go

import plotly.express as px
fig = px.parallel_categories(train_cleaned[['education_level','enrolled_university','major_discipline','target']], 

                             color="target", 

                             color_continuous_scale=px.colors.sequential.Inferno)

fig.show()
fig = px.parallel_categories(train_cleaned[['gender', 'target']], 

                             color="target", 

                             color_continuous_scale=px.colors.sequential.Inferno)

fig.show()
train_cleaned.columns
CompanyType = pd.crosstab(train_cleaned['company_type'],train_cleaned['target']).reset_index().melt(id_vars='company_type')

fig = px.bar(CompanyType, x="company_type", y="value", color='target', barmode='group',

             height=400, width=900)

fig.show()
Gender = pd.crosstab(train_cleaned['gender'],train_cleaned['target']).reset_index().melt(id_vars='gender')

fig = px.bar(Gender, x="gender", y="value", color='target', barmode='group',

             height=400, width=900)

fig.show()
MajorDiscipline = pd.crosstab(train_cleaned['major_discipline'],

                              train_cleaned['target']).reset_index().melt(id_vars='major_discipline')

fig = px.bar(MajorDiscipline, x="major_discipline", y="value", color='target', barmode='group',

             height=400, width=900)

fig.show()
EducationLevel = pd.crosstab(train_cleaned['education_level'],

                              train_cleaned['target']).reset_index().melt(id_vars='education_level')

fig = px.bar(EducationLevel, x="education_level", y="value", color='target', barmode='group',

             height=400, width=900)

fig.show()
CompanySize = pd.crosstab(train_cleaned['company_size'],

                          train_cleaned['target']).reset_index().melt(id_vars='company_size')

fig = px.bar(CompanySize, x="company_size", y="value", color='target', barmode='group',

             height=400, width=900)

fig.show()
Experience = pd.crosstab(train_cleaned['experience'],

                          train_cleaned['target']).reset_index().melt(id_vars='experience')

fig = px.bar(Experience, x="experience", y="value", color='target', barmode='group',

             height=400, width=900)

fig.show()
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from lightgbm import LGBMClassifier

model = LGBMClassifier(max_depth=5,

                       learning_rate=0.4, 

                       n_estimators=100)



model.fit(x_train,y_train,

          eval_set=[(x_train,y_train),(x_val, y_val.values)],

          eval_metric='auc',

          early_stopping_rounds=100,

          verbose=200)



pred_y = model.predict_proba(x_val)[:,1]
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

print(roc_auc_score(y_val, pred_y))

confusion_matrix(y_val, pred_y>0.5)
fpr, tpr, thresholds = roc_curve(y_val, pred_y)

fig = px.line(x=fpr, y=tpr, width=400, height=400,

              labels={'x':'False Positive Rates','y':'True Positive Rates'})

fig.show()
import lightgbm

lightgbm.plot_importance(model)
err = []

y_pred_tot_lgm = []



from sklearn.model_selection import StratifiedKFold



fold = StratifiedKFold(n_splits=15)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.05,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='auc',

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,1]

    print("err_lgm: ",roc_auc_score(y_val,pred_y))

    err.append(roc_auc_score(y_val, pred_y))

    pred_test = m.predict_proba(X_test)[:,1]

    i = i + 1

    y_pred_tot_lgm.append(pred_test)
np.mean(err,0)
from xgboost import XGBClassifier



errxgb = []

y_pred_tot_xgb = []



from sklearn.model_selection import KFold,StratifiedKFold



fold = StratifiedKFold(n_splits=15)

i = 1

for train_index, test_index in fold.split(X,y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = XGBClassifier(boosting_type='gbdt',

                      max_depth=5,

                      learning_rate=0.07,

                      n_estimators=5000,

                      random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='auc',

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,-1]

    print("err_xgb: ",roc_auc_score(y_val,pred_y))

    errxgb.append(roc_auc_score(y_val, pred_y))

    pred_test = m.predict_proba(X_test)[:,-1]

    i = i + 1

    y_pred_tot_xgb.append(pred_test)
np.mean(errxgb,0)
from catboost import CatBoostClassifier,Pool, cv

errCB = []

y_pred_tot_cb = []

from sklearn.model_selection import KFold,StratifiedKFold



fold = StratifiedKFold(n_splits=15)

i = 1

for train_index, test_index in fold.split(X,y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = CatBoostClassifier(n_estimators=5000,

                           random_state=1994,

                           eval_metric='AUC',

                           learning_rate=0.03, max_depth=5)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,-1]

    print("err_cb: ",roc_auc_score(y_val,pred_y))

    errCB.append(roc_auc_score(y_val,pred_y))

    pred_test = m.predict_proba(X_test)[:,-1]

    i = i + 1

    y_pred_tot_cb.append(pred_test)
np.mean(errCB, 0)
(np.mean(errxgb, 0) + np.mean(err, 0) + np.mean(errCB, 0))/3
# Stacking the predictions

submission = pd.DataFrame()

submission['enrollee_id'] = combine[combine['target'].isnull()==True]['enrollee_id']

submission['target'] = (np.mean(y_pred_tot_lgm, 0) + np.mean(y_pred_tot_cb, 0) + np.mean(y_pred_tot_xgb, 0))/3

submission.to_csv('Stacking.csv', index=False, header=True)

submission.shape
import tensorflow as tf

print(tf.__version__)
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epochs, logs={}):

        if(logs.get('val_AUC') > 0.620):

            self.model.stop_training = True

            

callbacks = myCallback()



model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=(X.shape[1],)),

    tf.keras.layers.Dense(100, activation='relu'),

    tf.keras.layers.Dense(10, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.summary()



from tensorflow.keras.optimizers import RMSprop, SGD, Adamax, Adagrad



model.compile(optimizer = "adam", 

              loss = 'binary_crossentropy', 

              metrics = ['AUC'])



history = model.fit(

    x_train, 

    y_train, 

    epochs = 30, 

    validation_data = (x_val, y_val),

    callbacks = [callbacks]

)



score = model.evaluate(x_val, y_val, verbose=1)



print("Test Score:", score[0])

print("Test AUC:", score[1])
auc = history.history['AUC']

val_auc = history.history['val_AUC']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(auc))



plt.plot(epochs, auc, label="Training")

plt.plot(epochs, val_auc, label="Validation")

plt.legend()

plt.title('Training and validation accuracy')

plt.show()



plt.plot(epochs, loss, label="Training")

plt.plot(epochs, val_loss, label="Validation")

plt.legend()

plt.title('Training and validation loss')

plt.show()
pred_test = model.predict(X_test)

submission = pd.DataFrame()

submission['enrollee_id'] = combine[combine['target'].isnull()==True]['enrollee_id']

submission['target'] = pred_test

submission.to_csv('NN.csv', index=False, header=True)

submission.shape