# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing libraries



import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn import metrics as ms

from sklearn.metrics import confusion_matrix

from xgboost.sklearn import XGBClassifier

from scipy.stats import uniform, randint

from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier
#Train Data



train_data = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

train_data.head(5)
# Test Data



test_data = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')

test_data.head(5)
print(train_data.shape)

print(test_data.shape)
train_data.info()
test_data.info()
train_data.isnull().sum()
test_data.isnull().sum()
Categorical_Feature = ['Loan_ID', 'Gender' , 'Married' , 'Dependents' , 'Education' , 'Self_Employed' , 'Loan_Amount_Term' , 'Property_Area', 'Credit_History']

Numerical_Feature = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']



for i in Categorical_Feature:

  train_data[i].fillna(train_data[i].mode()[0], inplace=True)

  test_data[i].fillna(test_data[i].mode()[0], inplace=True)



for j in Numerical_Feature:

  train_data[j] = train_data[j].replace(np.nan , train_data[j].median())

  test_data[j] = test_data[j].replace(np.nan , test_data[j].median())



print("count of null values for train_data")

print(train_data.isnull().sum())

print(" ")

print("Count of null values for test_data")

print(test_data.isnull().sum())
Objecttype_feature_list = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']



print("Unique value in train_data")

for i in Objecttype_feature_list:

  print("Count of unique values in column ", i, "is", train_data[i].nunique(), "which are", train_data[i].unique())



print(" ")

print(" ")



print("Unique value in test_data")

for i in Objecttype_feature_list:

  print("count of unique values in column ", i, "is", test_data[i].nunique(), "which are", test_data[i].unique())

train_data_LoanID = train_data['Loan_ID']

train_data = train_data.drop(columns=['Loan_ID'])

train_data_encoded = pd.get_dummies(train_data,drop_first=True)



test_data_LoanID = test_data['Loan_ID']

test_data = test_data.drop(columns=['Loan_ID'])

test_data_encoded = pd.get_dummies(test_data,drop_first=True)

train_data_encoded.columns
test_data_encoded.columns
# saving loan status and dropping it from train dataset.

train_Loan_status=train_data_encoded['Loan_Status_Y']

train_data_encoded=train_data_encoded.drop('Loan_Status_Y',axis=1)
eda_train_data_encoded = train_data_encoded

eda_test_data_encoded = test_data_encoded
eda_train_data_encoded.head(2)
eda_test_data_encoded.head(2)
eda_train_data_encoded.describe()
# checking skewness



for i in eda_train_data_encoded:

  print('skewness of',i,'is',eda_train_data_encoded[i].skew())
eda_train_data_encoded['Loan_Status_Y'] = train_Loan_status
eda_train_data_encoded['Loan_Status_Y'].value_counts()
eda_train_data_encoded['Loan_Status_Y'].hist(grid = False)
eda_train_data_encoded['Education_Not Graduate'].hist(grid = False)
counts, bin_edges = np.histogram(eda_train_data_encoded['ApplicantIncome'], bins=20, density = True)

pdf = counts/(sum(counts))

plt.plot(bin_edges[1:],pdf)



counts_1, bin_edges_1 = np.histogram(eda_test_data_encoded['ApplicantIncome'], bins=20, density = True)

pdf_1 = counts_1/(sum(counts_1))

plt.plot(bin_edges_1[1:],pdf_1)



plt.title('pdf of ApplicantIncome')

plt.legend(['eda_train_data_encoded_pdf', 'eda_test_data_encoded_pdf'])

plt.show()
counts, bin_edges = np.histogram(eda_train_data_encoded['LoanAmount'], bins=20, density = True)

pdf = counts/(sum(counts))

plt.plot(bin_edges[1:],pdf)



counts_1, bin_edges_1 = np.histogram(eda_test_data_encoded['LoanAmount'], bins=20, density = True)

pdf_1 = counts_1/(sum(counts_1))

plt.plot(bin_edges_1[1:],pdf_1)



plt.title('pdf of LoanAmount')

plt.legend(['eda_train_data_encoded_pdf', 'eda_test_data_encoded_pdf'])

plt.show()
counts, bin_edges = np.histogram(eda_train_data_encoded['Loan_Amount_Term'], bins=20, density = True)

pdf = counts/(sum(counts))

plt.plot(bin_edges[1:],pdf)



counts_1, bin_edges_1 = np.histogram(eda_test_data_encoded['Loan_Amount_Term'], bins=20, density = True)

pdf_1 = counts_1/(sum(counts_1))

plt.plot(bin_edges_1[1:],pdf_1)



plt.title('pdf of Loan_Amount_Term')

plt.legend(['eda_train_data_encoded_pdf', 'eda_test_data_encoded_pdf'])

plt.show()
eda_train_data_encoded['Loan_Amount_Term'].value_counts().sort_values().plot(kind = 'barh')

plt.show()
counts, bin_edges = np.histogram(eda_train_data_encoded['CoapplicantIncome'], bins=20, density = True)

pdf = counts/(sum(counts))

plt.plot(bin_edges[1:],pdf)



counts_1, bin_edges_1 = np.histogram(eda_test_data_encoded['CoapplicantIncome'], bins=20, density = True)

pdf_1 = counts_1/(sum(counts_1))

plt.plot(bin_edges_1[1:],pdf_1)



plt.title('pdf of CoapplicantIncome')

plt.legend(['eda_train_data_encoded_pdf', 'eda_test_data_encoded_pdf'])

plt.show()
sns.countplot('Loan_Status_Y', hue='Education_Not Graduate', data=eda_train_data_encoded)

plt.xlabel("Education_Not Graduate and Loan_Status_Y")

plt.ylabel("Count")



plt.title("count plot for Education_Not Graduate w.r.t. Loan_Status_Y")

plt.show()
sns.countplot('Loan_Status_Y', hue='Gender_Male', data=eda_train_data_encoded)

plt.xlabel("Gender_Male and Loan_Status_Y")

plt.ylabel("Count")

plt.title("count plot for Gender_Male w.r.t. Loan_Status_Y")

plt.show()
sns.countplot('Loan_Status_Y', hue='Self_Employed_Yes', data=eda_train_data_encoded)

plt.xlabel("Self_Employed_Yes and Loan_Status_Y")

plt.ylabel("Count")

plt.title("count plot for Self_Employed_Yes w.r.t. Loan_Status_Y")

plt.show()
sns.violinplot(x="Loan_Status_Y", y="ApplicantIncome", data=eda_train_data_encoded, size=8)

plt.show()
np.mean(eda_train_data_encoded['ApplicantIncome'])
plt.title('box plot')

sns.boxplot(x='Loan_Status_Y',y='LoanAmount', data=eda_train_data_encoded)

plt.show()
sns.violinplot(x="Loan_Status_Y", y="LoanAmount", data=eda_train_data_encoded, size=8)

plt.show()
sns.countplot('Loan_Status_Y', hue='Married_Yes', data=eda_train_data_encoded)

plt.xlabel("Married_Yes and Loan_Status_Y")

plt.ylabel("Count")

plt.title("count plot for Married_Yes w.r.t. Loan_Status_Y")

plt.show()
sns.countplot('Loan_Status_Y', hue='Credit_History', data=eda_train_data_encoded)

plt.xlabel("Credit_History and Loan_Status_Y")

plt.ylabel("Count")

plt.title("count plot for Credit_History w.r.t. Loan_Status_Y")

plt.show()
plt.scatter(eda_train_data_encoded['LoanAmount'], eda_train_data_encoded['Loan_Amount_Term'])

plt.show()
plt.scatter(eda_train_data_encoded['LoanAmount'], eda_train_data_encoded['ApplicantIncome'])

plt.show()
Norm_train_data = train_data_encoded

Norm_test_data = test_data_encoded


lst = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']

for i in lst:

  Norm_train_data[i]=((Norm_train_data[i]-Norm_train_data[i].min())/(Norm_train_data[i].max()-Norm_train_data[i].min()))

  Norm_test_data[i]=((Norm_test_data[i]-Norm_test_data[i].min())/(Norm_test_data[i].max()-Norm_test_data[i].min()))

Norm_train_data.head(5)
Norm_test_data.head(2)
after_eda_train_data = Norm_train_data

after_eda_test_data = Norm_test_data
plt.title('box plot')

after_eda_train_data.boxplot(column='ApplicantIncome')

plt.show()
IQR = (np.percentile(after_eda_train_data['ApplicantIncome'], 75)) - (np.percentile(after_eda_train_data['ApplicantIncome'], 25))

min = ((np.percentile(after_eda_train_data['ApplicantIncome'], 25)) - 1.5 * IQR)

max = ((np.percentile(after_eda_train_data['ApplicantIncome'], 75)) + 1.5 * IQR)

  

for j in range(len(after_eda_train_data['ApplicantIncome'])):

  if (after_eda_train_data['ApplicantIncome'][j]>=max):

    after_eda_train_data['ApplicantIncome'][j]=max

  elif (after_eda_train_data['ApplicantIncome'][j]<=min):

    after_eda_train_data['ApplicantIncome'][j]=min

plt.title('box plot')

after_eda_train_data.boxplot(column='ApplicantIncome')

plt.show()
plt.title('box plot')

after_eda_train_data.boxplot(column='LoanAmount')

plt.show()
IQR = (np.percentile(after_eda_train_data['LoanAmount'], 75)) - (np.percentile(after_eda_train_data['LoanAmount'], 25))

min = ((np.percentile(after_eda_train_data['LoanAmount'], 25)) - 1.5 * IQR)

max = ((np.percentile(after_eda_train_data['LoanAmount'], 75)) + 1.5 * IQR)

  

for j in range(len(after_eda_train_data['LoanAmount'])):

  if (after_eda_train_data['LoanAmount'][j]>=max):

    after_eda_train_data['LoanAmount'][j]=max

  elif (after_eda_train_data['LoanAmount'][j]<=min):

    after_eda_train_data['LoanAmount'][j]=min

plt.title('box plot')

after_eda_train_data.boxplot(column='LoanAmount')

plt.show()
plt.title('box plot')

after_eda_train_data.boxplot(column='CoapplicantIncome')

plt.show()
IQR = (np.percentile(after_eda_train_data['CoapplicantIncome'], 75)) - (np.percentile(after_eda_train_data['CoapplicantIncome'], 25))

min = ((np.percentile(after_eda_train_data['CoapplicantIncome'], 25)) - 1.5 * IQR)

max = ((np.percentile(after_eda_train_data['CoapplicantIncome'], 75)) + 1.5 * IQR)

  

for j in range(len(after_eda_train_data['CoapplicantIncome'])):

  if (after_eda_train_data['CoapplicantIncome'][j]>=max):

    after_eda_train_data['CoapplicantIncome'][j]=max

  elif (after_eda_train_data['CoapplicantIncome'][j]<=min):

    after_eda_train_data['CoapplicantIncome'][j]=min

plt.title('box plot')

after_eda_train_data.boxplot(column='CoapplicantIncome')

plt.show()
after_eda_train_data.corr()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["var"] = after_eda_train_data.columns

vif["VIF"] = [variance_inflation_factor(after_eda_train_data.values, i) for i in range(after_eda_train_data.shape[1])]

vif2=vif.sort_values(by=['VIF'], ascending=False)

vif2.reset_index(inplace = True)

print(vif2)
m_train_data = after_eda_train_data

m_test_data = after_eda_test_data
y = m_train_data['Loan_Status_Y']
m_train_data = m_train_data.drop('Loan_Status_Y', axis=1)
m_train_data.head()
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(m_train_data, y, stratify=y, test_size=0.2, random_state=1)

print(train_X.shape, val_X.shape, train_y.shape, val_y.shape )
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier
clf = CatBoostClassifier(loss_function='MultiClass', depth=5, iterations= 350, l2_leaf_reg= 1, learning_rate= 0.02)

clf.fit(train_X,train_y)



tr_pred=clf.predict(train_X)

val_pred=clf.predict(val_X)

print ("Accuracy for train is for ",ms.accuracy_score(train_y,tr_pred))

print ("Accuracy for val is for ",ms.accuracy_score(val_y,val_pred))



catboost_val_acc = ms.accuracy_score(val_y,val_pred)
confusion_matrix(val_y,val_pred)
# predicting test data



y_test_pred=clf.predict(m_test_data)

test_sub=pd.DataFrame(y_test_pred,columns=['Loan_Status'])

test_sub['Loan_ID']=test_data_LoanID
test_sub['Loan_Status']= test_sub['Loan_Status'].map({0: 'N' , 1: 'Y'})

test_sub=test_sub[['Loan_ID','Loan_Status']]



test_sub.head(2)
#test_sub.to_csv('catbboost_submission.csv',index=False)

#The test score for catboost classifier is : 0.7847222222222222.

# Always scale the input. The most convenient way is to use a pipeline.



alph=[0.001,0.01,0.1,1,10,100]

acc=[]



for i in alph:

  clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log', max_iter=1000, tol=1e-3, class_weight="balanced", alpha=i ))

  clf.fit(train_X, train_y)

  val_pred=clf.predict(val_X)

  acc.append(ms.accuracy_score(val_y,val_pred))

  print ("Accuracy for ",i, ms.accuracy_score(val_y,val_pred))

  

sgd_val_acc = np.max(acc)

print(sgd_val_acc)
r_cfl=RandomForestClassifier(random_state=42,n_jobs=-1)

r_cfl=RandomForestClassifier(n_estimators=40,random_state=42,n_jobs=-1, max_depth=4)

r_cfl.fit(train_X, train_y)

tr_pred=r_cfl.predict(train_X)

val_pred=r_cfl.predict(val_X)

print ("Accuracy for train is for ",i, ms.accuracy_score(train_y,tr_pred))

print ("Accuracy for val is for ",i, ms.accuracy_score(val_y,val_pred))



random_forest_val_acc = ms.accuracy_score(val_y,val_pred)
df=pd.DataFrame(data=[['CatBoostClassifier Model',catboost_val_acc], ['SGDClassifier',sgd_val_acc], ['RamdomForestClassifier',random_forest_val_acc]], columns=['Model','Validation accuracy'])

df