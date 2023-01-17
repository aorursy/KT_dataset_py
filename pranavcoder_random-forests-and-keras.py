# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")

%matplotlib inline
def process(df):

    # Imput missing lines and drop line with problem

    from sklearn.preprocessing import Imputer

    df['lead_time'] = Imputer(strategy='median').fit_transform(

                                    df['lead_time'].values.reshape(-1, 1))

    df = df.dropna()

    for col in ['perf_6_month_avg', 'perf_12_month_avg']:

        df[col] = Imputer(missing_values=-99).fit_transform(

                                    df[col].values.reshape(-1, 1))

    # Convert to binaries

    for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',

               'stop_auto_buy', 'rev_stop', 'went_on_backorder']:

        df[col] = (df[col] == 'Yes').astype(int)

    # Normalization    

    from sklearn.preprocessing import normalize

    qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 

                   'forecast_6_month', 'forecast_9_month', 'min_bank',

                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 

                   'sales_3_month', 'sales_6_month', 'sales_9_month',]

    df[qty_related] = normalize(df[qty_related], axis=1)

    # Scale lead time

    df.lead_time = df.lead_time/df.lead_time.max().astype(np.float64)    

    return df



cols=range(1,23)

df = process(pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv',usecols=cols))

df.head()
def plot_2d(X, y, title=''):

    from sklearn.preprocessing import StandardScaler

    X_std = StandardScaler().fit_transform(X)



    from sklearn.decomposition import PCA

    dec = PCA(n_components=2)

    X_reduced = dec.fit_transform(X_std)

    

    f, ax = plt.subplots(figsize=(6,6))

    ax.scatter(X_reduced[y==0,0], X_reduced[y==0,1], alpha=0.5, 

               facecolors='none', edgecolors='cornflowerblue', label="Negative")

    ax.scatter(X_reduced[y==1,0], X_reduced[y==1,1], c='darkorange', marker='*', 

               label='Positive')

    plt.title("Explained variance ratio: %.2f%%" % (100*dec.explained_variance_ratio_.sum()))

    ax.legend(loc='lower left')

    ax.spines['right'].set_visible(False)

    ax.spines['top'].set_visible(False)

    ax.set_xlabel('PC1')

    ax.set_ylabel('PC2')

    plt.show()

    

sample = df.sample(5000, random_state=36)



X_sample = sample.drop('went_on_backorder',axis=1).values

y_sample = sample['went_on_backorder'].values



plot_2d(X_sample, y_sample)
X = df.drop('went_on_backorder', axis=1).values

y = df['went_on_backorder'].values

print('Imbalanced ratio in training set: 1:%i' % (Counter(y)[0]/Counter(y)[1]))



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
from sklearn import tree, ensemble

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import make_pipeline



cart = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5)

rus = make_pipeline(RandomUnderSampler(),tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5))

forest = ensemble.RandomForestClassifier(criterion='entropy', max_depth=15, min_samples_leaf=5)

gboost = ensemble.GradientBoostingClassifier(max_depth=15, min_samples_leaf=5)



cart.fit(X_train, y_train)

rus.fit(X_train, y_train)

forest.fit(X_train, y_train)

#gboost.fit(X_train, y_train)
n_splits = 10



from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

ub = BaggingClassifier(warm_start=True, n_estimators=0)



for split in range(n_splits):

    X_res, y_res = RandomUnderSampler(random_state=split).fit_sample(X_train,y_train) 

    ub.n_estimators += 1

    ub.fit(X_res, y_res)
def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])

    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,

            label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))



f, ax = plt.subplots(figsize=(6,6))



roc_auc_plot(y_test,ub.predict_proba(X_test),label='UB ',l='-')

roc_auc_plot(y_test,forest.predict_proba(X_test),label='FOREST ',l='--')

roc_auc_plot(y_test,cart.predict_proba(X_test),label='CART', l='-.')

roc_auc_plot(y_test,rus.predict_proba(X_test),label='RUS',l=':')



ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', 

        label='Random Classifier')    

ax.legend(loc="lower right")    

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_xlim([0, 1])

ax.set_ylim([0, 1])

ax.set_title('Receiver Operator Characteristic curves')

sns.despine()
def precision_recall_plot(y_true, y_proba, label=' ', l='-', lw=1.0):

    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_test,

                                                  y_proba[:,1])

    average_precision = average_precision_score(y_test, y_proba[:,1],

                                                     average="micro")

    ax.plot(recall, precision, label='%s (average=%.3f)'%(label,average_precision),

            linestyle=l, linewidth=lw)



f, ax = plt.subplots(figsize=(6,6))

precision_recall_plot(y_test,ub.predict_proba(X_test),label='UB ',l='-')

precision_recall_plot(y_test,forest.predict_proba(X_test),label='FOREST ',l='-')

precision_recall_plot(y_test,cart.predict_proba(X_test),label='CART',l='-.')

precision_recall_plot(y_test,rus.predict_proba(X_test),label='RUS',l=':')



ax.set_xlabel('Recall')

ax.set_ylabel('Precision')

ax.legend(loc="upper right")

ax.grid(True)

ax.set_xlim([0, 1])

ax.set_ylim([0, 1])

ax.set_title('Precision-recall curves')

sns.despine()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve

from pandas import Series

train = pd.read_csv("../input/Kaggle_Training_Dataset_v2.csv")
train.columns
train.head(5)



#there are some null values so I dropped them

train.isnull().sum().sum()

train = train.dropna()

train.isnull().sum().sum()
train = train.drop(['lead_time'], axis = 1)

#train['lead_time'] = train['lead_time'].fillna(train['lead_time'].mean())

train=train.drop(['sku'], axis=1)

#train.replace(['Yes', 'No'], [1, 0])



train['went_on_backorder']=train['went_on_backorder'].map( {'No': 0, 'Yes': 1} ).astype(int)

train['potential_issue']=train['potential_issue'].map( {'No': 0, 'Yes': 1} ).astype(int)

train['deck_risk']=train['deck_risk'].map( {'No': 0, 'Yes': 1} ).astype(int)

train['oe_constraint']=train['oe_constraint'].map( {'No': 0, 'Yes': 1} ).astype(int)

train['ppap_risk']=train['ppap_risk'].map( {'No': 0, 'Yes': 1} ).astype(int)

train['stop_auto_buy']=train['stop_auto_buy'].map( {'No': 0, 'Yes': 1} ).astype(int)

train['rev_stop']=train['rev_stop'].map( {'No': 0, 'Yes': 1} ).astype(int)
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



a=[column for column in train]

trace = go.Heatmap(z=train.corr().values,

                   x=a,

                   y=a)

data=[trace]

py.iplot(data, filename='backorders heatmap')
X = train.drop(['went_on_backorder'], axis = 1)

Y = train['went_on_backorder']
plt.figure()

train['went_on_backorder'].value_counts().plot(kind = 'bar')

plt.ylabel("Count")

plt.title('Went on backorder? (0=No, 1=Yes)')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
from imblearn.over_sampling import SMOTE

oversampler = SMOTE(random_state = 0)

X_train, Y_train = oversampler.fit_sample(X_train, Y_train)
plt.figure()

Series(Y_train).value_counts().sort_index().plot(kind = 'bar')

plt.ylabel("Count")

plt.title('Went on backorder? (0=No, 1=Yes)')
def roc_curve_acc(Y_test, Y_pred,method):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',label='%s AUC = %0.3f'%(method, roc_auc))

    plt.legend(loc='lower right')

    plt.plot([0,1],[0,1],'b--')

    plt.xlim([-0.1,1.2])

    plt.ylim([-0.1,1.2])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')
RF = RandomForestClassifier()

RF.fit(X_train, Y_train)
features_list = train.columns.values

feature_importance = RF.feature_importances_

sorted_idx = np.argsort(feature_importance)[:20]

 

plt.figure(figsize=(5,7))

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()

plt.show()
train['national_inv'].describe()
Y_pred = RF.predict(X_test)
print("Random Forest Classifier report \n", classification_report(Y_test, Y_pred))
roc_curve_acc(Y_test, Y_pred, "RF")
plt.figure(figsize = (5,4))

cm = confusion_matrix(Y_test, Y_pred)

sns.heatmap(cm, annot = True)

plt.title('Confusion matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')
precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)

area = auc(recall, precision)

plt.figure()

plt.plot(recall, precision, label = 'Area Under Curve = %0.3f'% area)

plt.legend(loc = 'lower left')

plt.title('Precision-Recall curve')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.0])

plt.xlim([0.0, 1.0])

plt.show()
test = pd.read_csv("../input/Kaggle_Test_Dataset_v2.csv")
test = test.dropna()

#Why test data includes test results?

y_test_data=test['went_on_backorder'].map( {'No': 0, 'Yes': 1} ).astype(int)

test = test.drop(['lead_time','sku','went_on_backorder'], axis = 1)



test['potential_issue']=test['potential_issue'].map( {'No': 0, 'Yes': 1} ).astype(int)

test['deck_risk']=test['deck_risk'].map( {'No': 0, 'Yes': 1} ).astype(int)

test['oe_constraint']=test['oe_constraint'].map( {'No': 0, 'Yes': 1} ).astype(int)

test['ppap_risk']=test['ppap_risk'].map( {'No': 0, 'Yes': 1} ).astype(int)

test['stop_auto_buy']=test['stop_auto_buy'].map( {'No': 0, 'Yes': 1} ).astype(int)

test['rev_stop']=test['rev_stop'].map( {'No': 0, 'Yes': 1} ).astype(int)
Y_pred_test = RF.predict(test)
print("Test data Random Forest Classifier report \n", classification_report(y_test_data, Y_pred_test))


import csv

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Activation,Dropout

X=[]

Y=[]

with open('../input/Kaggle_Training_Dataset_v2.csv') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

         S=[]

         #S.append(row['deck_risk'])

         S.append(row['forecast_3_month'])

         S.append(row['forecast_6_month'])

         S.append(row['forecast_9_month'])

         S.append(row['in_transit_qty'])

         S.append(row['lead_time'])

         S.append(row['local_bo_qty'])

         S.append(row['min_bank'])

         S.append(row['national_inv'])

         S.append(row['oe_constraint'])

         S.append(row['perf_12_month_avg'])

         S.append(row['perf_6_month_avg'])

         S.append(row['pieces_past_due'])

         S.append(row['potential_issue'])

         S.append(row['ppap_risk'])

         S.append(row['sales_1_month'])

         S.append(row['sales_3_month'])

         S.append(row['sales_6_month'])

         S.append(row['sales_9_month'])

         S.append(row['sku'])

         S.append(row['stop_auto_buy'])  

         X.append(S)

         Y.append([row['went_on_backorder']])

for i in X:

    for n,s in enumerate(i):

      if s=='No':

        i[n]=0

      if s=='Yes':

        i[n]=1

      if s=='':

          i[n]=0

for i in X:

     for n,s in enumerate(i):

        i[n]=float(s)



         

for i in Y:

    for n,s in enumerate(i):

      if s=='No':

        i[n]=0

      if s=='Yes':

        i[n]=1

'''for i in Y:

     for n,s in enumerate(i):

        i[n]=float(s)'''



        

X=np.array(X).astype(float)  

Y=np.array(Y)  

model = Sequential()

model.add(Dense(100, activation='relu', input_dim=20))

model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='nadam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])     

model.fit(X,Y,verbose=1,shuffle=True, nb_epoch=3,batch_size=100,validation_split=0.2)

X=[]

Y=[]

with open('../input/Kaggle_Test_Dataset_v2.csv') as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

         S=[]

         #S.append(row['deck_risk'])

         S.append(row['forecast_3_month'])

         S.append(row['forecast_6_month'])

         S.append(row['forecast_9_month'])

         S.append(row['in_transit_qty'])

         S.append(row['lead_time'])

         S.append(row['local_bo_qty'])

         S.append(row['min_bank'])

         S.append(row['national_inv'])

         S.append(row['oe_constraint'])

         S.append(row['perf_12_month_avg'])

         S.append(row['perf_6_month_avg'])

         S.append(row['pieces_past_due'])

         S.append(row['potential_issue'])

         S.append(row['ppap_risk'])

         S.append(row['sales_1_month'])

         S.append(row['sales_3_month'])

         S.append(row['sales_6_month'])

         S.append(row['sales_9_month'])

         S.append(row['sku'])

         S.append(row['stop_auto_buy'])  

         X.append(S)

         Y.append([row['went_on_backorder']])

for i in X:

    for n,s in enumerate(i):

      if s=='No':

        i[n]=0

      if s=='Yes':

        i[n]=1

      if s=='':

          i[n]=0

for i in X:

     for n,s in enumerate(i):

        i[n]=float(s)         

for i in Y:

    for n,s in enumerate(i):

      if s=='No':

        i[n]=0

      if s=='Yes':

        i[n]=1

 

for i in X:

     for n,s in enumerate(i):

        i[n]=float(s)

score = model.evaluate(X,Y, batch_size=16)

print("LOSS")

print(score[0])

print("precision")

print(score[1])
