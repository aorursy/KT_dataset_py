import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
#READ DATASET

df=pd.read_csv("../input/bank-customer-churn-modeling/Churn_Modelling.csv")

df.head()
#SOME FEATURE ENGINEERING

df["EstimatedSalary/Tenure"]=df["EstimatedSalary"]/(df["Tenure"]+1)

df["EstimatedSalary/Age"]=df["EstimatedSalary"]/df["Age"]

df["EstimatedSalary/Numberofproducts"]=df["EstimatedSalary"]/df["NumOfProducts"]



df["Balance/Tenure"]=df["Balance"]/(df["Tenure"]+1)

df["Balance/Age"]=df["Balance"]/df["Age"]

df["Balance/Numberofproducts"]=df["Balance"]/df["NumOfProducts"]



df["Numberofproducts/Age"]=df["NumOfProducts"]/df["Age"]

df["Numberofproducts/Tenure"]=df["NumOfProducts"]/df["Tenure"]

#CONVERT 1-0 FEATURES TO OBJECT

df= df.astype({"HasCrCard": str})

df= df.astype({"IsActiveMember": str})
#SEPERATE AS TRAIN AND TEST

tr=df.head(8000)

te=df.tail(2000)
#SEPERATE TRAIN(DATAFRAME WITH THE NAME "TR") DATAFRAME AS TRAIN AND TEST FOR MODEL

X = tr.drop(['RowNumber','CustomerId','Exited'], axis=1)

y = tr['Exited']





categorical_features_indices = np.where((X.dtypes != np.int64)&(X.dtypes != np.float64))[0]



X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

#CATBOOST

cb_model = CatBoostClassifier(iterations=500, learning_rate=0.04, 

                              depth=7,eval_metric='AUC',verbose=50,

                              random_seed = 42,task_type='GPU')



cb_model.fit(X_train, y_train, eval_set=(X_validation,y_validation), 

             cat_features=categorical_features_indices, use_best_model=True)



#FEATURE IMPORTANCE

fea_imp = pd.DataFrame({'imp': cb_model.feature_importances_, 'col': X.columns})

fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-100:]

fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 25), legend=None)

plt.title('CatBoost - Feature Importance')

plt.ylabel('Features')

plt.xlabel('Importance')
#SCORING THE TEST(DATAFRAME WITH THE NAME "TR")

te4scor=te.drop(['RowNumber','CustomerId','Exited'], axis=1)

preds = cb_model.predict_proba(te4scor)[:,1]



#PUTTING THE PREDICTIONS IN TO DATAFRAME NAMED "df2"

df2 = pd.DataFrame()

df2['Exited_Pred'] = preds.tolist()

df2.head()
#MERGING THE RESULTS

te.reset_index(inplace=True,drop=True)



df2 = pd.merge(df2, te, left_index=True, right_index=True)

df2.head()
#LOOKING THE RESULTS

print(df2["Exited_Pred"].mean())

print(df2["Exited"].mean())



#AUC

from sklearn.metrics import roc_auc_score

y_true = np.array(df2["Exited"])

print(y_true)

y_scores = np.array(df2["Exited_Pred"])

print(y_scores)

roc_auc_score(y_true, y_scores)