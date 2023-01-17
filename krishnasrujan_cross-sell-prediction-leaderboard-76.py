import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling  import RandomUnderSampler
data = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
data
data.info()
data.Response.value_counts()

print(data.Response.value_counts()/data.shape[0] *100)
data.drop('id',axis=1,inplace=True)
ID = test['id']
test.drop('id',axis=1,inplace=True)
print(data['Vehicle_Age'].value_counts(),test['Vehicle_Age'].value_counts())
print(data['Region_Code'].value_counts(),test['Region_Code'].value_counts())
region_code = data.groupby(['Region_Code'])['Response'].sum().sort_values().to_dict()
policy_channel = data.groupby(['Policy_Sales_Channel'])['Response'].sum().sort_values().to_dict()
from sklearn.preprocessing import LabelEncoder 
l = LabelEncoder()
data['Gender'] = l.fit_transform(data['Gender'])
data['Vehicle_Age'] = l.fit_transform(data['Vehicle_Age'])
data['Vehicle_Damage'] = l.fit_transform(data['Vehicle_Damage'])
#data['Policy_Sales_Channel'] = l.fit_transform(data['Policy_Sales_Channel'])

test['Gender'] = l.fit_transform(test['Gender'])
test['Vehicle_Age'] = l.fit_transform(test['Vehicle_Age'])
test['Vehicle_Damage'] = l.fit_transform(test['Vehicle_Damage'])
test['Vehicle_Age'] = l.fit_transform(test['Vehicle_Age'])
data
sns.pairplot(data)
data.corr()
data['Region_Code'].replace(region_code,inplace=True)
data['Policy_Sales_Channel'].replace(policy_channel,inplace=True)

test['Region_Code'].replace(region_code,inplace=True)
test['Policy_Sales_Channel'].replace(policy_channel,inplace=True)
data
# as you can see the correlation of region code with response has increased
data.corr()
plt.figure(figsize=(15, 15))

for i, col in enumerate(data.columns,1):
    plt.subplot(5,4, i)
    sns.boxplot(y=col,data=data)
    plt.xlabel(col)
plt.tight_layout()
data['Annual_Premium'].hist(bins=50)
## Annual Premium has many ouliers so applying log transformation
data['Annual_Premium'] = np.log(data['Annual_Premium'])
test['Annual_Premium'] = np.log(test['Annual_Premium'])
data[['Annual_Premium']].boxplot()
data['Annual_Premium'].hist(bins=50)
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix
import xgboost
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

oversample = RandomOverSampler(sampling_strategy=0.5)
X, y = oversample.fit_resample(X, y)

print(X.shape)
scale = StandardScaler()
X = scale.fit_transform(X)
test = scale.transform(test)
from catboost import CatBoostClassifier, Pool
model = CatBoostClassifier(learning_rate=0.03,iterations=800,depth=6,
                           eval_metric='AUC',task_type="GPU",devices='0:1')
model.fit(X,
          y,
          eval_set=None,
          verbose=True)
model.get_feature_importance()
y_probs = model.predict_proba(test)
y_probs
y_probs = y_probs[:,1]
sub_cat = pd.DataFrame({'id':ID,'Response':y_probs})
sub_cat.set_index('id',inplace=True)
sub_cat