import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
PCOS_inf = pd.read_csv("../input/polycystic-ovary-syndrome-pcos/PCOS_infertility.csv")

PCOS_data = pd.read_csv("../input/polycystic-ovary-syndrome-pcos/data without infertility _final.csv")
PCOS_data.head().T
PCOS_data[~ PCOS_data['Unnamed: 42'].isna()].T
PCOS_data.info()
PCOS_data[PCOS_data['Marraige Status (Yrs)'].isnull()].T
#lets assign the median to the missing data

PCOS_data['Marraige Status (Yrs)'].fillna(PCOS_data['Marraige Status (Yrs)'].median(),inplace=True)
PCOS_data['Fast food (Y/N)'].fillna(PCOS_data['Fast food (Y/N)'].median(),inplace=True)
PCOS_data.drop('Unnamed: 42',axis=1,inplace=True)
PCOS_inf.head()
PCOS_inf.info()
data = pd.merge(PCOS_data,PCOS_inf, on='Patient File No.', suffixes={'','_y'},how='left')
data.columns = ['SNo', 'Patient_File_No.', 'PCOS_(Y/N)', 'Age_(yrs)', 'Weight_(Kg)',

       'Height(Cm)', 'BMI', 'Blood_Group', 'Pulse_rate(bpm)',

       'RR_(breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle_length(days)',

       'Marriage_Status_(Yrs)', 'Pregnant(Y/N)', 'No_of_aborptions',

       'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)',

       'Waist:Hip_Ratio', 'TSH_(mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)',

       'Vit_D3_(ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight_gain(Y/N)',

       'hair_growth(Y/N)', 'Skin_darkening (Y/N)', 'Hair_loss(Y/N)',

       'Pimples(Y/N)', 'Fast_food_(Y/N)', 'Reg_Exercise(Y/N)',

       'BP_Systolic(mmHg)', 'BP_Diastolic(mmHg)', 'Follicle_No.(L)',

       'Follicle_No.(R)', 'Avg.Fsize(L)(mm)', 'Avg.Fsize(R)(mm)',

       'Endometrium(mm)', 'Sl.No_y', 'PCOS(Y/N)_y',

       'I_beta-HCG(mIU/mL)', 'II_beta-HCG(mIU/mL)', 'AMH(ng/mL)_y']
data.drop(['Sl.No_y', 'PCOS(Y/N)_y','AMH(ng/mL)_y'],axis=1,inplace=True)
data.info()
data.describe().T
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

target = data['PCOS_(Y/N)']

data.drop('PCOS_(Y/N)',axis=1,inplace=True)
plt.figure(figsize=(8,7))

sns.countplot(target)

plt.title('Data imbalance')

plt.show()
X_train,X_test, y_train, y_test = train_test_split(data, target, test_size=0.15, random_state=1, stratify = target)

X_train,X_valid, y_train, y_valid =  train_test_split(X_train, y_train, test_size=0.3, random_state=1, stratify=y_train)
from sklearn.metrics import roc_auc_score

def print_scores(m):

    res = [roc_auc_score(y_train,m.predict_proba(X_train)[:,1]),roc_auc_score(y_valid,m.predict_proba(X_valid)[:,1])]

    for r in res:

        print(r)
rf = RandomForestClassifier(n_jobs=-1,n_estimators=150,max_features='sqrt',min_samples_leaf=10)

rf.fit(X_train,y_train)

print_scores(rf)
from sklearn.metrics import roc_curve

y_pred_proba = rf.predict_proba(X_valid)[:,1]

fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)
plt.figure(figsize=(8,7))

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=11) ROC curve')

plt.show()
def get_fi(m, df):

    return pd.DataFrame({'col': df.columns, 'imp': m.feature_importances_}).sort_values('imp',ascending=False)



#lets get the feature importances for training set

fi = get_fi(rf,X_train)
def plot_fi(df):

    df.plot('col','imp','barh',figsize=(10,10))

    

plot_fi(fi)