import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline





# pd.set_option('display.max_columns', 100)
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')
df.describe()
df.head()
# df.info()

120 -df.astype(bool).sum(axis=0)

# df['chem1'] = df
df['chem_1'].replace(0,np.nan,inplace= True)

df['chem_2'].replace(0,np.nan,inplace= True)
df.fillna(value=df.mean(),inplace=True)

# df["chem_1"].fillna(value=df.mean(),inplace=True)
# df['chem_5'].replace(0,np.nan,inplace= True)

# df['chem_6'].replace(0,np.nan,inplace= True)
df.isnull().any()
# missing_columns = ['chem_1','chem_2']
    # def random_imputation(df, feature):



    #     number_missing = df[feature].isnull().sum()

    #     observed_values = df.loc[df[feature].notnull(), feature]

    #     df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)



    #     return df
# for feature in missing_columns:

#     df[feature + '_imp'] = df[feature]

#     df = random_imputation(df, feature)
df.isnull().any()
# from sklearn import linear_model

# random_data = pd.DataFrame(columns = ["Ran" + name for name in missing_columns])



# for feature in missing_columns:

        

#     random_data["Ran" + feature] = df[feature + '_imp']

#     parameters = list(set(df.columns) - set(missing_columns) - {feature + '_imp'})

    

#     model = linear_model.LinearRegression()

#     model.fit(X = df[parameters], y = df[feature + '_imp'])

    

#     #Standard Error of the regression estimates is equal to std() of the errors of each estimates

#     predict = model.predict(df[parameters])

#     std_error = (predict[df[feature].notnull()] - df.loc[df[feature].notnull(), feature + '_imp']).std()

    

#     #observe that I preserve the index of the missing data from the original dataframe

#     random_predict = np.random.normal(size = df[feature].shape[0], 

#                                       loc = predict, 

#                                       scale = std_error)

#     random_data.loc[(df[feature].isnull()) & (random_predict > 0), "Ran" + feature] = random_predict[(df[feature].isnull()) & 

#                                                                             (random_predict > 0)]
df.describe()
df.corr()
import xgboost as xgb
X = df[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute"]].copy()

# X = df[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5_imp","chem_6_imp","chem_7","attribute"]].copy()



y = df["class"].copy()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.10,random_state=42)
# D_train = xgb.DMatrix(X_train, label=y_train)

# D_test = xgb.DMatrix(X_val, label=y_val)



# # param = {

# #     'eta': 0.3, 

# #     'max_depth': 3,  

# #     'objective': 'multi:softprob',  

# #     'num_class': 8} 



# steps = 20  # The number of training iterations

# model = xgb.train(param, D_train, steps)

# # preds = model

# # import numpy as np

# from sklearn.metrics import precision_score, recall_score, accuracy_score



# preds = model.predict(D_test)

# best_preds = np.asarray([np.argmax(line) for line in preds])



# # print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))

# # print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))

# print("Accuracy = {}".format(accuracy_score(y_val, best_preds)))

from sklearn.model_selection import GridSearchCV



clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree',

                                     colsample_bylevel=1, colsample_bynode=1,

                                     colsample_bytree=1, gamma=0,

                                     learning_rate=0.1, max_delta_step=0,

                                     max_depth=3, min_child_weight=1,

                                     missing=None, n_estimators=100, n_jobs=1,

                                     nthread=None, objective='binary:logistic',

                                     random_state=0, reg_alpha=0,

                                     scale_pos_weight=1, seed=None, silent=None,

                                     subsample=1, verbosity=1)




#TODO

# clf = RandomForestClassifier()        #Initialize the classifier object



# parameters = {'n_estimators':[2,3,4,5,6,7,8,9,10,11,12,14,15]}    #Dictionary of parameters



# scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



# grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



# grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



# best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



# unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

# optimized_predictions = best_clf.predict(X_val)        #Same, but use the best estimator



# acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

# acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model

# grid_fit.best_params_



# print("Accuracy score on unoptimized model:{}".format(acc_unop))

# print("Accuracy score on optimized model:{}".format(acc_op))
clf.fit(X_train,y_train)

predict = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

predict.isnull().any()

# predit[]

# predict.rename({'chem_1': 'chem_1_imp', 'chem_2': 'chem_2_imp'}, axis=1, inplace=True)

# predict = pd.get_dummies(predict, columns=["type"])
X_test_predict = predict[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute"]].copy()
y_pred_lr_test = clf.predict(X_test_predict)
predict['class'] = y_pred_lr_test
predict.head()
ans = predict[["id","class"]].copy()

ans.head()
ans.to_csv('ans.csv',index=False,encoding ='utf-8' )