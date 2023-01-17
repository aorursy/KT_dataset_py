import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data = pd.read_excel('../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')

#preparing to be easier later
data['NEW ICU'] = 0

data

#same code from the https://www.kaggle.com/felipeveiga/starter-covid-19-sirio-libanes-icu-admission

comorb_lst = [i for i in data.columns if "DISEASE" in i]
comorb_lst.extend(["HTN", "IMMUNOCOMPROMISED", "OTHER"])

demo_lst = [i for i in data.columns if "AGE_" in i]
demo_lst.append("GENDER")

vitalSigns_lst = data.iloc[:,193:-2].columns.tolist()

lab_lst = data.iloc[:,13:193].columns.tolist()
#dataframe where each patient is a row
data_groupby = data.groupby("PATIENT_VISIT_IDENTIFIER", as_index = False).agg({"ICU":(list), "WINDOW":list, 'NEW ICU':(list)})
#so we need to create this new dataset

data_groupby['USEFUL WINDOWS'] = 99

data_groupby['NEW ICU'] = 0

data_groupby


#for each row of the dataset
for index1, row in data_groupby.iterrows():

  #count = 0 means that we did not encounter 1
  count= 0

  #for each element of the ICU list of that row
  for i in range(len(data_groupby['ICU'][index1])):

    #it means it is our first 1. If is the first element, we dont want it
    if count == 0 and data_groupby['ICU'][index1][i] == 1 and i ==0:

      data_groupby['USEFUL WINDOWS'][index1] = int(-99)

    #it is our first one and we want the previous value
    if count == 0 and data_groupby['ICU'][index1][i] == 1 and i !=0:

      data_groupby['USEFUL WINDOWS'][index1] = int(i - 1)

      data_groupby['NEW ICU'][index1] = 1

    #count gett updated
    if data_groupby['ICU'][index1][i] == 1:

      count =+ 1

data_groupby.head(20)
#creating the new dataframe
df = pd.DataFrame(columns=data.columns)

lst_new_icu = []
#for each row in our groupby dataset
for index1, row in data_groupby.iterrows():

    #list where we are going to put the index that we want we have to see if we need to change ICU target or not
    lst_row_number = []

    #if -99, we ignore
    if data_groupby['USEFUL WINDOWS'][index1] == -99:
      pass

    #if 99, we want all rows of that patient
    if data_groupby['USEFUL WINDOWS'][index1] == 99:

      lst_row_number = [index1*5, (index1*5) + 1, (index1*5) + 2, (index1*5) + 3, (index1*5) + 4]
      lst_new_icu.append(0)
      lst_new_icu.append(0)
      lst_new_icu.append(0)
      lst_new_icu.append(0)
      lst_new_icu.append(0)

    if data_groupby['USEFUL WINDOWS'][index1] > -1 and data_groupby['USEFUL WINDOWS'][index1] < 6:
      #we only want the row before the first 1
      lst_row_number = [(index1*5) + (data_groupby['USEFUL WINDOWS'][index1])]
      lst_new_icu.append(1)


    if len(lst_row_number) > 0:

      for i in range(len(lst_row_number)):

        df = df.append(data.iloc[int(lst_row_number[i])], ignore_index=False, verify_integrity=False, sort=None)
df['NEW ICU 2'] = lst_new_icu
df.head(20)
#removing column window and using one hot enconding
df = df.drop(['WINDOW', 'PATIENT_VISIT_IDENTIFIER'], axis = 1)
df = pd.get_dummies(df, drop_first=True)

df.rename(columns={'NEW ICU 2': 'ICU'}, inplace=True)
df
#target class
sns.countplot('ICU', data=df, palette = 'GnBu')
plt.title('Does not need ICU vs Needs ICU', fontsize=14);
#we changed the names, so we have to create the list again
demo_lst = ['GENDER_1', 'AGE_ABOVE65_1', 'AGE_PERCENTIL_20th', 'AGE_PERCENTIL_30th', 'AGE_PERCENTIL_40th', 'AGE_PERCENTIL_50th', 'AGE_PERCENTIL_60th', 'AGE_PERCENTIL_70th', 'AGE_PERCENTIL_80th', 'AGE_PERCENTIL_90th', 'AGE_PERCENTIL_Above 90th' ]

#we do not have the column window anymore
vitalSigns_lst = vitalSigns_lst[:-1]

#using demographics + comorbities
X = df[comorb_lst + demo_lst]

y = df['ICU']
#xgboost with hyperopt
import xgboost as xgb
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score

data_dmatrix = xgb.DMatrix(data=X,label=y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

y_test.value_counts()
y_train.value_counts()

def objective(space):
    print(space)
    clf = xgb.XGBRegressor(n_estimators =int(space['n_estimators']),
                           n_fold = 3,
                           num_parallel_tree = int(space['num_parallel_tree']),
                           colsample_bytree=space['colsample_bytree'],
                           learning_rate = space['learning_rate'],
                           max_depth = int(space['max_depth']),
                           min_child_weight = int(space['min_child_weight']),
                           subsample = space['subsample'],
                           gamma = space['gamma'],
                           reg_alpha = space['reg_alpha'],
                           reg_lambda = space['reg_lambda'],
                           scale_pos_weight = space['scale_pos_weight'],
                           objective = 'binary:logistic')

    eval_set  = [( X_train, y_train), ( X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,verbose=False)

    pred = clf.predict(X_test)

    
    for i in range(len(pred)):

      if pred[i] >= 0.5:

        pred[i] = 1

      else:

        pred[i] = 0
    


    #score = roc_auc_score(y_test, pred)
    #score = f1_score(y_test, pred)
    score = recall_score(y_test, pred)
    #score = accuracy_score(y_test, pred)
    print ("SCORE:", score)
    return {'loss':1 - score, 'status': STATUS_OK }


space ={'max_depth': hp.uniform("x_max_depth", 1, 10),
        'min_child_weight': hp.uniform ('x_min_child', 1, 10),
        'subsample': hp.uniform ('x_subsample', 0.5, 1),
        'gamma' : hp.uniform ('x_gamma', 0.1, 1.0),
        'colsample_bytree' : hp.uniform ('x_colsample_bytree', 0.1 ,1),
        'reg_alpha' : hp.uniform('x_reg_alpha', 5, 7),
        'n_estimators' :hp.uniform('x_n_estimators', 100, 500),
        'learning_rate' :hp.uniform('x_learning_rate', 0.01 , 0.05),
        'scale_pos_weight' :hp.uniform('x_scale_pos_weight', 4.5, 6.5),
        'reg_lambda' : hp.uniform ('x_reg_lambda',1, 1.1),
        'num_parallel_tree' :hp.uniform('x_num_parallel_tree', 100, 200)
    }


trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print(best)
import matplotlib.pyplot as plt

xg_reg = xgb.XGBRegressor(n_estimators = int(best['x_n_estimators']),
                          n_fold = 3,
                          num_parallel_tree = int(best['x_num_parallel_tree']),
                          colsample_bytree=best['x_colsample_bytree'],
                          learning_rate = best['x_learning_rate'],
                          max_depth = int(best['x_max_depth']),
                          min_child_weight = int(best['x_min_child']),
                          subsample = best['x_subsample'],
                          gamma = best['x_gamma'],
                          reg_alpha = best['x_reg_alpha'],
                          reg_lambda = best['x_reg_lambda'],
                          scale_pos_weight = best['x_scale_pos_weight'],
                          objective = 'binary:logistic')


xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)



for i in range(len(preds)):

  if preds[i] >= 0.5:

    preds[i] = round(1)

  else:

    preds[i] = round(0)


lst_y_test = []

for i in range(len(y_test)):

  lst_y_test.append(y_test.iloc[i])

lst_y_test


xgb.plot_importance(xg_reg)


plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

conf_matrix = confusion_matrix(lst_y_test, preds)

#tn, fp, fn, tp = conf_matrix.ravel()

tp = conf_matrix[0,0]
fp = conf_matrix[1,0]
tn = conf_matrix[1,1]
fn = conf_matrix[0,1]


print('True Positives = ', tp)
print('True Negatives = ', tn)
print('False Positives = ', fp)
print('False Negatives = ', fn)

print("correct ICU predictions = %.2f%%" % (100*tn/(tn + fp)))

print("correct no ICU predictions = %.2f%%" % (100*tp/(tp + fn)))

plt.subplots(figsize=(8,8))
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax);

ax.set_xlabel('Prediction');ax.set_ylabel('Label'); 
ax.set_title('Confusion matrix xgboost'); 
ax.xaxis.set_ticklabels(['No ICU', 'ICU']); ax.yaxis.set_ticklabels(['No ICU', 'ICU']);


print("Accuracy: %.2f" % metrics.accuracy_score(y_test, preds))
print("Precision: %.2f" % metrics.precision_score(y_test, preds))
print("Recall: %.2f" % metrics.recall_score(y_test, preds))
print("f1_score: %.2f" % metrics.f1_score(y_test, preds))