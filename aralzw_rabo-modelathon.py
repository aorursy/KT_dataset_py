import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier



%matplotlib inline
# data set which you can split into train / validation

df = pd.read_csv("/kaggle/input/rabo-modelathon/train.csv")



# test set excluding the Default column

df_test = pd.read_csv("/kaggle/input/rabo-modelathon/test.csv")
### Boxplot of features by default (0 or 1) 

feature_names = df.drop(['Id', 'Default'], axis=1).columns

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 12))

for idx, feat in enumerate(feature_names): 

    ax = axes[int(idx / 5), idx % 5]

    sns.stripplot(x='Default', y=feat, data=df, ax=ax, jitter = False, size=1.5, color='black', edgecolor='gray')

    sns.boxplot(x='Default', y=feat, data=df, ax=ax)  

    ax.set_ylabel(feat) 

fig.tight_layout()

plt.show()
### Data preprocessing



X = df.drop(columns=['Default', 'Id'])

y = df['Default']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
### Build your best model and evaluate performance



model = XGBClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_val)

print('ROC score: {:.2f}'.format(roc_auc_score(y_val, y_pred))) 
### Create a list of predictions 

X_test = df_test.drop(['Id'], axis = 1)

predictions = model.predict(X_test)
#save it to csv 

pd.DataFrame({'Id': df_test['Id'], 'Outcome': predictions}).to_csv('submissions.csv', index = False)



print("Created submission file, please go to the notebook overview and go to the output tab to submit")