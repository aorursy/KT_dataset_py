# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV
session_df= pd.read_csv('../input/session_related.csv')

#sessionData.head()
delivery_df= pd.read_csv('../input/generic_outliers_data.csv')

#genericData.head()
data_df = pd.read_csv('../input/delivery_related.csv')

#deliveryData.head()
firstmerge = pd.merge(deliveryData,sessionData,on='OrderId')
MergedDataset = pd.merge(firstmerge,genericData,on='CustId')

#MergedDataset.head()
IndependentAttributes = pd.DataFrame()
CustId = MergedDataset['CustId']

OrderId = MergedDataset['OrderId']

EmailId = MergedDataset['EmailId']

MobileNo = MergedDataset['MobileNo']

MacAddress = MergedDataset['MacAddress']

AvgPurchase = MergedDataset['AvgPurchase']

City = MergedDataset['City']

IsValidGeo = MergedDataset['IsValidGeo']

IsValidAddress = MergedDataset['IsValidAddress']

IsDeliveryRejected = MergedDataset['IsDeliveryRejected']

ReplacementDate = MergedDataset['ReplacementDate']

IsOneTimeUseProduct = MergedDataset['IsOneTimeUseProduct']

Session_Pincode = MergedDataset['Session_Pincode']

DeliveryDate = MergedDataset['DeliveryDate']

OrderDate = MergedDataset['OrderDate']

Fraud = MergedDataset['Fraud']
IndependentAttributes['CustId'] = CustId

IndependentAttributes['OrderId'] = OrderId

IndependentAttributes['EmailId'] = EmailId

IndependentAttributes['MobileNo'] = MobileNo

IndependentAttributes['MacAddress'] = MacAddress

IndependentAttributes['Session_Pincode'] = Session_Pincode

IndependentAttributes['AvgPurchase'] = AvgPurchase

IndependentAttributes['City'] = City

IndependentAttributes['OrderDate'] = OrderDate

IndependentAttributes['DeliveryDate'] = DeliveryDate
df1 = pd.DataFrame(IndependentAttributes['OrderDate'])

df1['DeliveryDate'] = IndependentAttributes['DeliveryDate']

df1['OrderDate'] = pd.to_datetime(df1['OrderDate'], format='%d/%m/%Y')

df1['DeliveryDate'] = pd.to_datetime(df1['DeliveryDate'], format='%d/%m/%Y')

DaysDifference = (df1['DeliveryDate'] - df1['OrderDate'])

df1.drop(['DeliveryDate', 'OrderDate'], axis='columns', inplace=True)



IndependentAttributes['DaysDifference'] = DaysDifference
IndependentAttributes['ReplacementDate'] = ReplacementDate

IndependentAttributes['IsDeliveryRejected'] = IsDeliveryRejected

IndependentAttributes['IsOneTimeUseProduct'] = IsOneTimeUseProduct

IndependentAttributes['IsValidAddress'] = IsValidAddress

IndependentAttributes['IsValidGeo'] = IsValidGeo

IndependentAttributes['Fraud'] = Fraud
IndependentAttributes.IsDeliveryRejected.replace(('yes', 'no'), (1, 0), inplace=True)

IndependentAttributes.IsOneTimeUseProduct.replace(('yes', 'no'), (1, 0), inplace=True)

IndependentAttributes.IsValidGeo.replace(('YES', 'NO'), (1, 0), inplace=True)

IndependentAttributes.IsValidAddress.replace(('yes', 'no'), (1, 0), inplace=True)

IndependentAttributes.Fraud.replace(('normal', 'suspicious', 'fraudulent'), (0, 1, 1), inplace=True)
IndependentAttributes.MacAddress.replace(regex=r'-', value='', inplace=True)
IndependentAttributes.DeliveryDate.replace(regex=r'/', value='-', inplace=True)

IndependentAttributes.OrderDate.replace(regex=r'/', value='-', inplace=True)

IndependentAttributes.ReplacementDate.replace(regex=r'/', value='-', inplace=True)
IndependentAttributes['DaysDifference'] = IndependentAttributes.DaysDifference.dt.days

IndependentAttributes.head()
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
df = pd.DataFrame() 

df = reduce_mem_usage(IndependentAttributes)
Fraud_txn = df[df['Fraud']== 1]

normal_txn = df[df['Fraud']== 0]



# print("---------------------------")

# print("From the training dataset:")

# print("---------------------------")

# print("  Total Customers : %i"\

#       %(len(df)))

# print("")

# print("  Total Normal transactions  : %i"\

#       %(len(normal_txn)))



# print("  Normal transactions Rate   : %i %% "\

#      % (1.*len(normal_txn)/len(df)*100.0))

# print("-------------------------")



# print("  Fraudulent transactions         : %i"\

#       %(len(Fraud_txn)))



# print("  Fraudulent transactions Rate    : %i %% "\

#      % (1.*len(Fraud_txn)/len(df)*100.0))

# print("-------------------------")
from sklearn.preprocessing import LabelEncoder

categorical = ['IsValidAddress','IsDeliveryRejected','IsOneTimeUseProduct','IsValidGeo','CustId','City']



label_encoder = LabelEncoder()

for col in categorical:

    df[col] = label_encoder.fit_transform(df[col].astype(str))



df=df.iloc[:df.shape[0]]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

features = pd.DataFrame(df)

labels = pd.DataFrame(features['Fraud'])

features = features.drop(['Fraud','EmailId','MacAddress','City'], axis=1)

features = features.apply(LabelEncoder().fit_transform)

#features.head()
#labels.head()
# Scaling the Train and Test feature set 

#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()

#features = scaler.fit_transform(features)

features.head()
labels.head()
from sklearn.model_selection import train_test_split

train, test, labels_train, labels_test = train_test_split(features, labels, test_size=0.60, random_state = 42)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, labels_train, test_size=0.20, random_state = 42)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0)

x_train_std_os,y_train_os = sm.fit_sample(X_train,y_train)


from sklearn.ensemble.forest import RandomForestClassifier

rf = RandomForestClassifier(random_state=0).fit(x_train_std_os,y_train_os)

rf_pred = rf.predict(X_val)
from sklearn import metrics
###Just gave the output of few algo with smote which classifies both classes and without smote it doesnt classify both classes
temp1 = metrics.f1_score(y_val,rf_pred,average=None)[1]

temp2 = metrics.f1_score(y_val,rf_pred,average=None)[0]



# F1Fraud= temp1*100

# F1Normal= temp2*100





F1Fraud = str(round(temp1, 2)*100)

F1Normal = str(round(temp2, 2)*100)





#print(F1Fraud,F1Normal,sep="\n")
# import json

# data = {"F1 Fraud":F1Fraud,"F1 Normal":F1Normal, "model": "SVM"}

# jstr = json.dumps(data)



# with open('F1.json', 'w') as outfile: 

#     jstr

# import json

# with open('data.json', 'w') as outfile:

#     data = {"F1":metrics.f1_score(y_val,pred,average=None), "model": "SVM"}

#     json.dump(data, outfile)



# data = {"F1":metrics.f1_score(y_val,pred,average=None), "model": "SVM"}    

# import json

# with open('data.txt', 'w') as f:

#     json.dump(data, f, ensure_ascii=False)



import json



data = {

    'f1_fraud': F1Fraud,

    'f1_normal': F1Normal,

    'model': 'SVM'

}





with open("data_file.json", "w") as write_file:

    json.dump(data, write_file)

    

with open("data_file.json", "r") as read_file:

    data = json.load(read_file)

    

print(data)
features.head()
# y_hats2 = model.predict(X)



# features['y_hats'] = y_hats2



rf_pred
rf_pred
res = pd.DataFrame(rf_pred)

results = res.rename(columns={0:'Fraud'})
results.head()
final_export = pd.DataFrame()
OrderId = features['OrderId']

#OrderDate = features['OrderDate']

#City = features['City']

AvgPurchase = features['AvgPurchase']

PredictedLabel = results['Fraud']

#EmailId = features['EmailId']

final_export['OrderId'] = OrderId 

final_export['CustId'] = CustId 

final_export['City'] = City 

final_export['PredictedLabel'] = PredictedLabel 

final_export['AvgPurchase'] = AvgPurchase

final_export['EmailId'] = EmailId

final_export['DeliveryDate']=DeliveryDate

final_export['OrderDate'] = OrderDate 

final_export.head()
export_csv = final_export.to_csv (r'abc.csv', index = None, header=True)