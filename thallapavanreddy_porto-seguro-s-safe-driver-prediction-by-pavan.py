import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline





import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 65000)



from scipy import stats



from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split



##Synthetic Minority Over-sampling Technique to overcome imbalanced dataset

#from imblearn.over_sampling import SMOTE 



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold



from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, make_scorer

score_fun = make_scorer('roc_auc')
df = pd.read_csv("/kaggle/input/porto-seguro-safe-driver-prediction/train.csv")

test_data = pd.read_csv("/kaggle/input/porto-seguro-safe-driver-prediction/test.csv")
df['target'].value_counts(normalize=True)
df['target'].value_counts()
df = df.replace(-1, np.NAN)

test_data = test_data.replace(-1, np.NAN)
test_id = test_data['id']

test_data.drop(columns='id', inplace=True)
missing_values_per = df.isna().sum()/df.shape[0]*100

missing_values_per[missing_values_per>40]
df.drop(columns='ps_car_03_cat', inplace=True)
df_id = df['id']

y = df['target']

df.drop(columns=['id', 'target'], inplace=True)
test_data.shape
df.shape
columns = df.columns.to_list()
cat = []

reg = []

for i in columns:

    if 'cat' in i:

        cat.append(i)

    elif 'bin' in i:

        cat.append(i)        

    elif 'reg' in i:

        reg.append(i)

    elif 'ind' in i:

        cat.append(i)

    elif df[i].dtype=='float64':

        reg.append(i)

    else:

        cat.append(i)
df[reg] = df[reg].astype('float64')

df[cat] = df[cat].astype('O')
df[reg].head()
df[cat].describe()
rows, columns = df.shape

drop_nunique_col = []

for col in df.columns:

    if df[col].nunique == rows or df[col].nunique == 1:

        drop_nunique_col.append(col)

drop_nunique_col    
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0)
for col in X_train.columns:

    if X_train[col].dtype == 'object':

        X_train[col] = X_train[col].fillna(X_train[col].mode()[0])

        X_test[col] = X_test[col].fillna(X_train[col].mode()[0])

        test_data[col] = test_data[col].fillna(X_train[col].mode()[0])



    else:

        X_train[col] = X_train[col].fillna(X_train[col].mean())

        X_test[col] = X_test[col].fillna(X_train[col].mean())

        test_data[col] = test_data[col].fillna(X_train[col].mean())
'''

print(y_train.value_counts())

print(y_train.value_counts(normalize=True))

print(y_train.shape)



smort = SMOTE(sampling_strategy=0.3, k_neighbors=8)  ## SMOTE Parameters

X_train, y_train = smort.fit_resample(X_train, y_train)



print(y_train.value_counts())

print(y_train.value_counts(normalize=True))

print(y_train.shape)

'''
class LabelEncoderExt(object):

    def __init__(self):

        """

        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]

        Unknown will be added in fit and transform will take care of new item. It gives unknown class id

        """

        self.label_encoder = LabelEncoder()

        # self.classes_ = self.label_encoder.classes_



    def fit(self, data_list):

        """

        This will fit the encoder for all the unique values and introduce unknown value

        :param data_list: A list of string

        :return: self

        """

        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])

        self.classes_ = self.label_encoder.classes_



        return self



    def transform(self, data_list):

        """

        This will transform the data_list to id list where the new values get assigned to Unknown class

        :param data_list:

        :return:

        """

        new_data_list = list(data_list)

        for unique_item in np.unique(data_list):

            if unique_item not in self.label_encoder.classes_:

                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]



        return self.label_encoder.transform(new_data_list)
Le = LabelEncoderExt()

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        Le.fit(X_train[col])

        X_train[col] = Le.transform(X_train[col]).asdtype('int64')

        X_test[col] = Le.transform(X_test[col]).asdtype('int64')

        test_data[col] = Le.transform(test_data[col]).asdtype('int64')
mMs = MinMaxScaler()

for col in reg:

        #mMs.fit([X_train[col]])

        X_train[col] = mMs.fit_transform(np.array(X_train[col]).reshape(-1, 1))

        X_test[col] = mMs.transform(np.array(X_test[col]).reshape(-1, 1))

        test_data[col] = mMs.transform(np.array(test_data[col]).reshape(-1, 1))
ch2_Value = []

pValue = []

for col in cat:

    ct = pd.crosstab(X_train[col], y_train)   

    ch2_Value.append(stats.chi2_contingency(ct)[0])

    pValue.append(stats.chi2_contingency(ct)[1])

    

ch2_df = pd.DataFrame()

ch2_df['cat_columns'] = cat

ch2_df['ch2_value'] = ch2_Value

ch2_df['pValue'] = pValue







print("Before Feature Selection[Ch2_Test] No of Categorical Columns: =======>", len(cat))

ch2_test_af_col = ch2_df[ch2_df['pValue']<0.06]['cat_columns'].tolist()

print("After Feature Selection[Ch2_Test] No of Categorical Columns: ========>", len(ch2_test_af_col))



final_col = reg + ch2_test_af_col     ## combing continous and ch2 test outcome columns 



X_train = X_train[final_col]

X_test = X_test[final_col]

test_data = test_data[final_col]





print("Final Number of Columns : ========>", len(final_col))
'''

features_importance = rf.feature_importances_

features_importance[::-1].sort()



feature_imp_df = pd.DataFrame()

feature_imp = []

for i, col in enumerate(X_train.columns):

    feature_imp.append(features_importance[i])

    #print("{}. {} ({})".format(i + 1, col, features_importance[i]))

feature_imp_df['cName'] = X_train.columns

feature_imp_df['feature_imp_Val'] = feature_imp



#feature_imp_df



plt.figure(figsize=(15, 15))

plt.plot(np.arange(1, features_importance.shape[0]+1), np.cumsum(features_importance))

'''
rf_hyp = RandomForestClassifier(n_jobs=-1)
rf_params = {

    'n_estimators': [50, 75, 100],

    'criterion' : ['gini', 'entropy'], 

    'max_depth' : [4, 5, 6, 7, 8], 

    'max_leaf_nodes': [20, 30, 40, 50],

    'min_samples_leaf': [5, 10, 15, 20]

}
rsCV = RandomizedSearchCV(estimator=rf_hyp, param_distributions=rf_params, scoring='roc_auc')
rsCV.fit(X_train, y_train)
best_params = rsCV.best_params_
best_params
rf_best_Model = rsCV.best_estimator_ ## Choosing Best estimator with best parameters
rf_best_Model.fit(X_train, y_train)
def cross_val_model(X,y, model, n_splits=3, n_folds=3):

    'Do split dataset and calculate cross_score'

    X = np.array(X)

    y = np.array(y)

    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y))



    for j, (train_idx, test_idx) in enumerate(folds):

        X_train = X[train_idx]

        y_train = y[train_idx]

        X_holdout = X[test_idx]

        y_holdout = y[test_idx]



        print ("Fit %s Split %d" % (str(model).split('(')[0], j+1))

        model.fit(X_train, y_train)

        cross_score = cross_val_score(model, X_holdout, y_holdout, cv=n_folds, scoring='roc_auc')

        print("         Mean cross_score of ",n_folds," Folds : =========:>", cross_score.mean())
cross_val_model(X_train, y_train, rf_best_Model, n_splits=5, n_folds=5)
y_predict = rf_best_Model.predict(X_test)

y_predict_proba = rf_best_Model.predict_proba(X_test)[::,1]
train_acc = rf_best_Model.score(X_train, y_train)    # Model Evaluation

test_acc =  rf_best_Model.score(X_test, y_test)



recallScore = recall_score(y_test, y_predict)

precisionScore = precision_score(y_test, y_predict)



f1Score = f1_score(y_test, y_predict)

auc = roc_auc_score(y_test, y_predict_proba)

fpr, tpr, thrshould = roc_curve(y_test, y_predict_proba)
print("\n\n")

print("Model Name: ", str(rf_best_Model).split("(")[0])

print("ConfusionMatrix: \n", confusion_matrix(y_test, y_predict))    

print("TrainAcc: ====> {}".format(train_acc))

print("TestAccuracy : ====> {}".format(test_acc))

print("recall: ====> {}".format(recallScore))

print("Precision: ====> {}".format(precisionScore))

print("F1Score: ====> {}".format(f1Score))

print("AUC: ====> {}".format(auc))
plt.figure(figsize=(8,8))

plt.plot(fpr, tpr, label="Model Name: "+str(rf_best_Model).split("(")[0]+"\n"+"auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Area Under The Curve AUC-ROC')



plt.legend(loc= 7)

plt.show()
## Submission 
result_df = pd.DataFrame(columns=['id', 'target'])

result_df['id'] = test_id

result_df['target'] = rf_best_Model.predict_proba(test_data)[::, 1]
result_df.to_csv("Porto Seguro_Submission.csv", index=False, sep = ',', encoding = 'utf8')
test_data.shape