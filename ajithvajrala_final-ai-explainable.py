import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier

from imblearn.under_sampling import RandomUnderSampler

from sklearn import svm

import eli5

from eli5.sklearn import PermutationImportance



from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score, auc, roc_auc_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv")
train_df.shape
train_df.head()
train_df.columns
train_df.isna().sum()
train_df.dtypes
train_df.nunique()
train_df.columns
train_df['Revenue'].value_counts()/(len(train_df))
sns.countplot(train_df['Revenue']);
train_df.columns
missing_cols =['Administrative', 'Administrative_Duration', 'Informational',

       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',

       'BounceRates', 'ExitRates']



for col in missing_cols:

    train_df[col] = train_df[col].fillna(train_df[col].mean())
train_df.describe()
sns.countplot(train_df['VisitorType']);
sns.countplot(train_df['Weekend']);
sns.countplot(train_df['Month']);
corr = train_df.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
plt.figure(figsize=(15,15))

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,  annot=True);
outliers = ['Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration','PageValues' ]

for col in outliers:

    train_df[col] = train_df[col].clip(lower=train_df[col].quantile(0.1), upper=train_df[col].quantile(0.98))
train_df.describe()
train_df.groupby(['Revenue']).mean()
for col in ['Month','VisitorType', 'Weekend']:

    train_df[col] = train_df[col].astype('category').cat.codes
train_df['Revenue'] = train_df['Revenue'].astype('category').cat.codes
train_df['Revenue'].value_counts()
#Modelling
X = train_df.drop(['Revenue'], axis=1)

y = train_df['Revenue']
X_train,X_test, y_train,  y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
vals = precision_recall_fscore_support(y_test, y_pred, average='macro')

print(vals)

precision = vals[0]

recall = vals[1]

f1 = vals[2]
rf
def feature_imporatance_plotting(vals_imp, names,model_type):



    feature_importance = np.array(vals_imp)

    feature_names = np.array(names)



    #Create a DataFrame using a Dictionary

    data={'feature_names':feature_names,'feature_importance':feature_importance}

    fi_df = pd.DataFrame(data)



    

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)



    plt.figure(figsize=(10,8))

    

    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

   

    plt.title(model_type + ' important features')

    plt.xlabel('Level of Imporatance')

    plt.ylabel('Names of variables')
feature_imporatance_plotting(rf.feature_importances_,X.columns,'Random Forest')
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)



print(confusion_matrix(y_test, y_pred_dt))

print(classification_report(y_test, y_pred_dt))

print(accuracy_score(y_test, y_pred_dt))
dt.get_params()
feature_imporatance_plotting(dt.feature_importances_,X.columns,'Decision Trees')
extra = ExtraTreesClassifier()

extra.fit(X_train, y_train)

y_pred_extra = extra.predict(X_test)

print(confusion_matrix(y_test, y_pred_extra))

print(classification_report(y_test, y_pred_extra))

print(accuracy_score(y_test, y_pred_extra))
extra.get_params()
feature_imporatance_plotting(extra.feature_importances_,X.columns,'Extra Tree Classifier')
#Logistic Regression

logit = LogisticRegression()

logit.fit(X_train, y_train)

y_pred_log_prob = logit.predict_proba(X_test)[:,1]

y_pred_log = logit.predict(X_test)

print(roc_auc_score(y_test, y_pred_log_prob))
logit.get_params()
print(confusion_matrix(y_test, y_pred_log))

print(classification_report(y_test, y_pred_log))
def logistic_plot(y_test, y_pred_proba):

    logit_roc_auc = roc_auc_score(y_test, y_pred_proba)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

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
logistic_plot(y_test, y_pred_log_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_log_prob)

#print(tpr)

#print(fpr)

#print(thresholds)

print(roc_auc_score(y_test, y_pred_log_prob))

optimal_idx = np.argmax(tpr - fpr)

optimal_threshold = thresholds[optimal_idx]

print("Threshold value is:", optimal_threshold)

#plot_roc_curve(fpr, tpr)
y_logis_pred = np.where(y_pred_log>0.16706550300190282, 1, 0)

print(classification_report(y_test, y_logis_pred))
#Permutation feature importance

X_tr_perm, X_val_perm, y_train_perm, y_val_perm = train_test_split(X_train, y_train, 

                                                                   stratify=y_train, 

                                                                   test_size=0.2, 

                                                                  random_state=42)
rf_perm = RandomForestClassifier()

rf_perm.fit(X_tr_perm, y_train_perm)
perm = PermutationImportance(rf_perm, random_state=1).fit(X_val_perm, y_val_perm)

eli5.show_weights(perm, feature_names = X.columns.tolist())
dt_perm = DecisionTreeClassifier()

dt_perm.fit(X_tr_perm, y_train_perm)



perm = PermutationImportance(dt_perm, random_state=1).fit(X_val_perm, y_val_perm)

eli5.show_weights(perm, feature_names = X.columns.tolist())
lg_perm = LogisticRegression()

lg_perm.fit(X_tr_perm, y_train_perm)



perm = PermutationImportance(lg_perm, random_state=1).fit(X_val_perm, y_val_perm)

eli5.show_weights(perm, feature_names = X.columns.tolist())
ex_perm = ExtraTreesClassifier()

ex_perm.fit(X_tr_perm, y_train_perm)



perm = PermutationImportance(ex_perm, random_state=1).fit(X_val_perm, y_val_perm)

eli5.show_weights(perm, feature_names = X.columns.tolist())
#Keras

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import to_categorical 

from sklearn.metrics import roc_auc_score

import tensorflow as tf

from keras import backend as K
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
X_train.shape, y.shape, X_test.shape, y_test.shape
model = Sequential()



model.add(Dense(16, input_dim=17, activation='elu'))

#model.add(Dense(8, activation='elu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
print(model.summary())
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.fit(X_train.values, y_train.values, epochs=10, batch_size=16)
y_pred = model.predict(X_test)

print(roc_auc_score(y_test, y_pred))
logistic_plot(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

#print(tpr)

#print(fpr)

#print(thresholds)

print(roc_auc_score(y_test, y_pred))

optimal_idx = np.argmax(tpr - fpr)

optimal_threshold = thresholds[optimal_idx]

print("Threshold value is:", optimal_threshold)

#plot_roc_curve(fpr, tpr)
y_pred_label= np.where(y_pred > 0.15409985, 1, 0)
confusion_matrix(y_test, y_pred_label)
print(classification_report(y_test, y_pred_label))