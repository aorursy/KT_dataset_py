import pandas as pd

data = pd.read_csv("../input/wns-inno/train_LZdllcl.csv")
data.head()
#import pixiedust
#display(data)
data = data.drop(['region','employee_id'],axis =1)
data.columns
data.shape
data.isna().sum()
data["education"].fillna( method ='ffill', inplace = True)

data = data.fillna(data.mean())

data.isna().sum()
#data = data.dropna()
data.shape
data.dtypes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Categorical boolean mask

categorical_feature_mask = data.dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = data.columns[categorical_feature_mask].tolist()

categorical_cols
#['department', 'region', 'education', 'gender', 'recruitment_channel']

print(data['recruitment_channel'].nunique())

print(data['department'].nunique())

print(data['education'].nunique())

print(data['gender'].nunique())
print(data['department'].unique())

print(data['recruitment_channel'].unique())

print(data['education'].unique())

print(data['gender'].unique())
# instantiate labelencoder object

le = LabelEncoder()

# apply le on column gender

data['gender'] = le.fit_transform(data['gender'])

data.head(2)
from sklearn.preprocessing import OneHotEncoder

# instantiate OneHotEncoder

features = ['department', 'education', 'recruitment_channel']

ohe = OneHotEncoder(categorical_features = features, sparse=False ) 

# categorical_features = boolean mask for categorical columns

# sparse = False output an array not sparse matrix
# apply OneHotEncoder on categorical feature columns

#X_ohe = ohe.fit_transform(data) # It returns an numpy array

ohe = OneHotEncoder(sparse=False)

X_ohe = ohe.fit_transform(data[['department', 'education', 'recruitment_channel']])
X_ohe.shape
type(X_ohe)
X_ohe
df= pd.get_dummies(data['department'], prefix=['department'],drop_first=True)

df1 =  pd.get_dummies(data['education'], prefix=['education'],drop_first=True)

df2 =  pd.get_dummies(data['recruitment_channel'], prefix=['RC'],drop_first=True)

data = pd.concat([data, df, df1,df2],axis=1)

data = data.drop(['department', 'education', 'recruitment_channel'],axis=1)
data.head(2)
data.shape
from sklearn.model_selection import train_test_split
X = data.drop(['is_promoted'],axis=1)

Y = data['is_promoted']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state = 42, test_size = 0.2)
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()

logit.fit(X_train,Y_train)
predict1 = logit.predict(X_test)
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

print(classification_report(Y_test,predict1))
fpr,tpr,threshold = roc_curve(Y_test,logit.predict_proba(X_test)[:,1])
logit_roc_auc_1 = roc_auc_score(Y_test,logit.predict(X_test))

logit_roc_auc_1
from sklearn.metrics import f1_score

# f1 score

score = f1_score(predict1, Y_test)

score
import matplotlib.pyplot as plt 

plt.plot(fpr,tpr,label = 'Logistic Regression (Sensitivity = %0.3f)'%logit_roc_auc_1)

plt.plot([0,1],[0,1],'r--')

plt.title('ROC Curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc = "lower right")
pd.value_counts(data['is_promoted'])

from imblearn.over_sampling import SMOTE

from collections import Counter

# applying SMOTE to our data and checking the class counts

X_resampled, y_resampled = SMOTE().fit_resample(X, Y)

print(sorted(Counter(y_resampled).items()))
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_resampled,y_resampled,test_size = 0.2, random_state =2 )
logit.fit(X1_train, Y1_train)
pred1 = logit.predict(X1_test)
print(classification_report(Y1_test, pred1))
from sklearn.metrics import f1_score

# f1 score

score = f1_score(pred1, Y1_test)

score
from imblearn.over_sampling import ADASYN

from collections import Counter

# applying SMOTE to our data and checking the class counts

X_resampled1, y_resampled1 = ADASYN().fit_resample(X, Y)

print(sorted(Counter(y_resampled1).items()))
X2_train,X2_test,Y2_train,Y2_test = train_test_split(X_resampled1,y_resampled1,test_size = 0.2, random_state =2 )
logit.fit(X2_train, Y2_train)
pred2 = logit.predict(X2_test)
print(classification_report(Y2_test, pred2))
from sklearn.metrics import f1_score

# f1 score

score = f1_score(pred2, Y2_test)

score
test = pd.read_csv("../input/wns-inno/test_2umaH9m.csv")
test.isna().sum()
test1 = pd.read_csv("../input/wns-inno/test_2umaH9m.csv")
test.head()
test.keys()
test = test.drop(['region','employee_id'],axis =1)

test["education"].fillna( method ='ffill', inplace = True)

test = test.fillna(test.mean())

test.isna().sum()
#['department', 'region', 'education', 'gender', 'recruitment_channel']

print(test['recruitment_channel'].nunique())

print(test['department'].nunique())

print(test['education'].nunique())

print(test['gender'].nunique())
# instantiate labelencoder object

le = LabelEncoder()

# apply le on column gender

test['gender'] = le.fit_transform(test['gender'])

test.head(2)
df4= pd.get_dummies(test['department'], prefix=['department'],drop_first=True)

df5 =  pd.get_dummies(test['education'], prefix=['education'],drop_first=True)

df6 =  pd.get_dummies(test['recruitment_channel'], prefix=['RC'],drop_first=True)
test = pd.concat([test, df4, df5,df6],axis=1)

test = test.drop(['department','education', 'recruitment_channel'],axis=1)
test.shape
test_pred = logit.predict(test)

len(test_pred)
import numpy as np

employee_id=np.array(test1['employee_id'])

len(employee_id)
submission = pd.DataFrame({'employee_id': employee_id, 'is_promoted': list(test_pred)}, columns=['employee_id', 'is_promoted'])
submission.head()
submission.shape
# Install `XlsxWriter` 

#!pip install XlsxWriter



# Specify a writer

#writer = pd.ExcelWriter('submission.xlsx', engine='xlsxwriter')



# Write your DataFrame to a file     

#submission.to_excel(writer, 'Sheet1')



# Save the result 

#writer.save()
from sklearn.ensemble import RandomForestClassifier

random_forest1 = RandomForestClassifier( max_depth=15)

random_forest1.fit(X2_train, Y2_train)
pred_forest = random_forest1.predict(X2_test)

#X2_train,X2_test,Y2_train,Y2_test
print(classification_report(Y2_test, pred_forest))
score = f1_score(pred_forest, Y2_test)

score
test_pred_forest = random_forest1.predict(test)

len(test_pred_forest)
submission = pd.DataFrame({'employee_id': employee_id, 'is_promoted': list(test_pred_forest)}, columns=['employee_id', 'is_promoted'])
submission.head()
# Specify a writer

#writer = pd.ExcelWriter('submission.xlsx', engine='xlsxwriter')



# Write your DataFrame to a file     

#submission.to_excel(writer, 'Sheet1')



# Save the result 

#writer.save()