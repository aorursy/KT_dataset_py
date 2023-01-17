!pip install pycaret
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


telcom = pd.read_csv(r'/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
telcom.head()
print ("Rows     : " ,telcom.shape[0])
print ("Columns  : " ,telcom.shape[1])
print ("\nMissing values :  ", telcom.isnull().sum().values.sum())
print ("\nUnique values :  \n",telcom.nunique())
telcom.info()
#Data Manipulation

#Replacing spaces with null values in total charges column
telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan)

#Dropping null values from total charges column which contain .15% missing data 
telcom = telcom[telcom["TotalCharges"].notnull()]
telcom = telcom.reset_index()[telcom.columns]

#convert to float type
telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

#replace 'No internet service' to No for the following columns
replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols : 
    telcom[i]  = telcom[i].replace({'No internet service' : 'No'})
    
#replace values
telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes",0:"No"})

#Tenure to categorical column
def tenure_lab(telcom) :
    
    if telcom["tenure"] <= 12 :
        return "Tenure_0-12"
    elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24 ):
        return "Tenure_12-24"
    elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48) :
        return "Tenure_24-48"
    elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60) :
        return "Tenure_48-60"
    elif telcom["tenure"] > 60 :
        return "Tenure_gt_60"
telcom["tenure_group"] = telcom.apply(lambda telcom:tenure_lab(telcom),
                                      axis = 1)

#Separating churn and non churn customers
churn     = telcom[telcom["Churn"] == "Yes"]
not_churn = telcom[telcom["Churn"] == "No"]

#Separating catagorical and numerical columns
Id_col     = ['customerID']
target_col = ["Churn"]
cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
import seaborn as sns
import matplotlib.pyplot  as plt
val_counts = telcom["Churn"].value_counts()

#labels
lab = val_counts.keys().tolist()
#values
val = val_counts.values.tolist()

fig1, ax1 = plt.subplots()
ax1.pie(val, labels=lab, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
def plotpie(col):
  val_counts_churn = churn[col].value_counts()
  #labels
  lab_churn = val_counts_churn.keys().tolist()
  #values
  val_churn = val_counts_churn.values.tolist()

  val_counts_notchurn = not_churn[col].value_counts()
  #labels
  lab_notchurn = val_counts_churn.keys().tolist()
  #values
  val_notchurn = val_counts_churn.values.tolist()

  fig1, (ax1,ax2) = plt.subplots(1,2)
  fig1.suptitle(col)
  ax1.pie(val_churn, labels=lab_churn, autopct='%1.1f%%', shadow=True)
  ax1.axis('equal')
  ax2.pie(val_notchurn, labels=lab_notchurn, autopct='%1.1f%%', shadow=True)
  ax2.axis('equal')


for col in cat_cols:
  plotpie(col)
def plothist(col):
  plt.figure()
  ax = sns.boxplot(x="Churn", y=col, data=telcom)


for col in num_cols:
  plothist(col)
sns.pairplot(telcom[num_cols + ['Churn']], hue="Churn")
tenures = sorted(telcom["tenure_group"].unique())

ax = sns.countplot(x="tenure_group", hue="Churn", data=telcom,order=tenures)
plt.xticks(rotation=90)

plt.figure()


ax = sns.barplot(x="tenure_group", y="MonthlyCharges", hue='Churn', data=telcom, estimator=np.mean,order=tenures)
plt.xticks(rotation=90)

plt.figure()
ax = sns.barplot(x="tenure_group", y="TotalCharges", hue='Churn', data=telcom, estimator=np.mean,order=tenures)
plt.xticks(rotation=90)
#
#tmp_df = pd.DataFrame(telcom.groupby(['Churn'])['TotalCharges'].mean()).reset_index()
#tmp_df.columns = ['Churn','Avg Total Charges']
#ax = sns.countplot(x="Avg Total Charges", hue="Churn", data=tmp_df)
#plt.xticks(rotation=90)
#
sns.relplot(x="MonthlyCharges", y="TotalCharges", hue="tenure_group", alpha=.5, palette="muted",
            height=6, data=telcom)
plt.figure()
sns.relplot(x="MonthlyCharges", y="TotalCharges", hue="Churn", alpha=.5, palette="muted",
            height=6, data=telcom)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#customer id col
Id_col     = ['customerID']
#Target columns
target_col = ["Churn"]
#categorical columns
cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
#numerical columns
num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
#Binary columns with 2 values
bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    telcom[i] = le.fit_transform(telcom[i])
    
#Duplicating columns for multi value columns
telcom = pd.get_dummies(data = telcom,columns = multi_cols )

#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(telcom[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

#dropping original values merging scaled values for numerical columns
df_telcom_og = telcom.copy()
telcom = telcom.drop(columns = num_cols,axis = 1)
telcom = telcom.merge(scaled,left_index=True,right_index=True,how = "left")

telcom.info()
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X = telcom[[i for i in telcom.columns if i not in Id_col + target_col]]
Y = telcom[target_col + Id_col]

principal_components = pca.fit_transform(X)
pca_data = pd.DataFrame(principal_components,columns = ["PC1","PC2"])
pca_data = pca_data.merge(Y,left_index=True,right_index=True,how="left")
pca_data["Churn"] = pca_data["Churn"].replace({1:"Churn",0:"Not Churn"})

sns.relplot(x="PC1", y="PC2", hue="Churn", alpha=.5, palette="muted",
            height=6, data=pca_data)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score
#splitting train and test data 
telcom = telcom.drop(columns = Id_col,axis = 1)
train,test = train_test_split(telcom,test_size = .25 ,random_state = 111)
    
##seperating dependent and independent variables
#cols    = [i for i in telcom.columns if i not in Id_col + target_col]
#train_X = train[cols]
#train_Y = train[target_col]
#test_X  = test[cols]
#test_Y  = test[target_col]
train.info()
from pycaret.utils import enable_colab
enable_colab()
features = train.columns.tolist()
features.remove('Churn')
from pycaret.classification import *
exp_clf101 = setup(data = train, target = 'Churn',numeric_features=features, session_id=123) 
compare_models()
xboost = create_model('xgboost')
tuned_xgboost= tune_model(xboost)
plot_model(tuned_xgboost, plot = 'auc')
plot_model(tuned_xgboost, plot = 'pr')
plot_model(tuned_xgboost, plot='feature')
plot_model(tuned_xgboost, plot = 'confusion_matrix')
predict_model(tuned_xgboost);
final_xgboost = finalize_model(tuned_xgboost)
print(final_xgboost)
unseen_predictions = predict_model(final_xgboost, data=test)
unseen_predictions.head()
save_model(final_xgboost,'Final xgboost Model 13_08_2020')
saved_final_xgboost = load_model('Final xgboost Model 13_08_2020')
new_prediction = predict_model(saved_final_xgboost, data=test)
new_prediction.head()