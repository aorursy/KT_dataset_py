import pandas as pd
import sklearn
import numpy as np

file_path = '../input/minor-project-2020/train.csv'
train = pd.read_csv(file_path) 
train.columns

file_path = '../input/minor-project-2020/test.csv'
test = pd.read_csv(file_path) 

columns = ['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6',
       'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_13',
       'col_14', 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'col_20',
       'col_21', 'col_22', 'col_23', 'col_24', 'col_25', 'col_26', 'col_27',
       'col_28', 'col_29', 'col_30', 'col_31', 'col_32', 'col_33', 'col_34',
       'col_35', 'col_36', 'col_37', 'col_38', 'col_39', 'col_40', 'col_41',
       'col_42', 'col_43', 'col_44', 'col_45', 'col_46', 'col_47', 'col_48',
       'col_49', 'col_50', 'col_51', 'col_52', 'col_53', 'col_54', 'col_55',
       'col_56', 'col_57', 'col_58', 'col_59', 'col_60', 'col_61', 'col_62',
       'col_63', 'col_64', 'col_65', 'col_66', 'col_67', 'col_68', 'col_69',
       'col_70', 'col_71', 'col_72', 'col_73', 'col_74', 'col_75', 'col_76',
       'col_77', 'col_78', 'col_79', 'col_80', 'col_81', 'col_82', 'col_83',
       'col_84', 'col_85', 'col_86', 'col_87']
X = train[columns]
X_t = test[columns]
columns = ['target']
Y = train[columns]
Y = np.where((Y == 0), 0, 1)
class0 = np.where(Y == 0)[0]
class1 = np.where(Y == 1)[0]


# from sklearn import preprocessing

# scaler = preprocessing.MinMaxScaler()              #Instantiate the scaler
# X = scaler.fit_transform(X_train)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
sample =  RandomOverSampler(random_state=42)
X,Y = sample.fit_sample(X,Y)
len(X)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
Model = LogisticRegression()
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    }
]
#model = GridSearchCV(Model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
model = LogisticRegression(C=0.0018329807108324356, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)


param = model.fit(X, Y)
#param.best_estimator
final = model.predict_proba(X_t)    
final
#param.best_estimator_
import csv

with open('classify.csv', 'w') as csvfile:
    fieldnames = ['id', 'target']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)  
        
    writer.writeheader()
    
    for i in range(len(test)):
      writer.writerow({'id':(int)(test.loc[[i],'id']), 'target': final[i][1] }) 

    
