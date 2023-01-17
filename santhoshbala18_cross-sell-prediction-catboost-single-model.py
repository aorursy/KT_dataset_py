import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/av-janatahack-crosssell-prediction/train.csv")

test = pd.read_csv("../input/av-janatahack-crosssell-prediction/test.csv")
train.shape
test.shape
sns.countplot(x='Previously_Insured',hue='Response',data=train)
sns.countplot(x='Previously_Insured',hue='Response',data=train[train['Previously_Insured']==0])
train_without_insured = train[train['Previously_Insured']==0].reset_index(drop=True)
train_without_insured.head()
train_without_insured.Response.value_counts()
sns.countplot(x='Gender',hue='Response',data=train_without_insured)
sns.countplot(x='Driving_License',hue='Response',data=train_without_insured)

print(train.Driving_License.value_counts())
sns.countplot(x='Vehicle_Damage',hue='Response',data=train_without_insured)
sns.countplot(x='Vehicle_Age',hue='Response',data=train_without_insured)
train_without_insured.describe()
plt.hist(train_without_insured['Annual_Premium'])

plt.show()
binary_cols = ['Gender','Vehicle_Damage']

cat_cols=['Region_Code','Vehicle_Age','Policy_Sales_Channel']

cont_cols = ['Annual_Premium','Age']

drop_cols = ['Driving_License','id','Vintage']
#Combine both train and test data for processing

combined_df = pd.concat([train,test], sort=False)
#Policy channel with less than 100 records are combined to form "Others"

channel_count = combined_df.Policy_Sales_Channel.value_counts()

combined_df.loc[combined_df.Policy_Sales_Channel.isin(channel_count[channel_count<100].index),'Policy_Sales_Channel'] = 0
for each in drop_cols:

    combined_df.drop([each], axis=1, inplace=True)
# Convert Categorical columns to cat codes

for each in binary_cols:

    combined_df[each] = combined_df[each].astype('category').cat.codes



for each in cat_cols:

    combined_df[each] = combined_df[each].astype('category').cat.codes
combined_df.head()
train_df = combined_df[combined_df['Response'].notnull()]

test_df = combined_df[combined_df['Response'].isnull()]

train_df['Response'] = train_df['Response'].astype('int')

test_df.drop('Response',axis=1,inplace=True)
train_df = train_df[train_df['Previously_Insured']==0].reset_index(drop=True)

test_df_without_insured = test_df[test_df['Previously_Insured']==0].reset_index(drop=True)

train_df.drop('Previously_Insured',axis=1,inplace=True)

test_df_without_insured.drop('Previously_Insured',axis=1,inplace=True)
#Normalizing continuous columns

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



for each_col in cont_cols:

    train_df[each_col] = scaler.fit_transform(train_df[each_col].values.reshape(-1,1))

    test_df_without_insured[each_col] = scaler.transform(test_df_without_insured[each_col].values.reshape(-1,1))
X = train_df.drop('Response',axis=1)

y =  train_df['Response']
from sklearn.model_selection import KFold, StratifiedKFold

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score



fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

overall_preds = np.zeros((test_df_without_insured.shape[0]))

for train_index, test_index in fold.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    catboost_model = CatBoostClassifier(iterations=1000,border_count=103, learning_rate=0.05, l2_leaf_reg=30, depth=6, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42)

    catboost_model.fit(X_train,y_train,eval_set=(X_test, y_test), cat_features= binary_cols+cat_cols,early_stopping_rounds=200,verbose=200)

    pred_prob = catboost_model.predict_proba(X_test)[:,1]

    print("Validation ROC AUC:"+str(roc_auc_score(y_test,pred_prob)))

    ypreds = catboost_model.predict_proba(test_df_without_insured)[:,1]

    overall_preds += ypreds

overall_preds = overall_preds / 3
test_df.loc[test_df['Previously_Insured']==0,'Response'] = overall_preds

test_df.loc[test_df['Previously_Insured']==1,'Response'] = 0
subs = pd.read_csv("../input/av-janatahack-crosssell-prediction/sample_submission_iA3afxn.csv")
subs['Response']=test_df.Response
subs.to_csv("catboost_prediction.csv", index=False)