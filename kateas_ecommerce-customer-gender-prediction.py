#Load necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from collections import Counter
#scikit learn  library
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
#Use LogisticRegression for classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#oversampling
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading dataset ( without header)
header_str =  ['user_id','startTime','endTime','ProductList']
dataset = pd.read_csv("../input/trainingData.csv", header=None,names = header_str) 
dataset_test = pd.read_csv("../input/trainingLabel.csv", header=None,names = ['gender']) 

#Custom function for extract infor from product list
def extract_product(str):
    if ";"  in str:
        prd_lst = str.split(";")
        count_item = len(prd_lst)
        first_lv1 = prd_lst[0].split("/")[0]
        first_lv2 = prd_lst[0].split("/")[1]
        lv1_lst =[]
        lv2_lst =[]
        for item in prd_lst:
            lv1_lst.append(item.split("/")[0])
            lv2_lst.append(item.split("/")[1])
        unique_lv1 = len(set(lv1_lst))
        unique_lv2 = len(set(lv2_lst))
        most_freq_lv1 =  max(lv1_lst, key=Counter(lv1_lst).get)        
    else:
        lv_lst = str.split("/")
        first_lv1 = lv_lst[0]
        first_lv2 = lv_lst[1]
        count_item = 1
        unique_lv1 = 1
        unique_lv2 = 1
        most_freq_lv1 = first_lv1
    return (count_item,first_lv1,first_lv2,unique_lv1,unique_lv2,most_freq_lv1)    

#Feature Extraction :
new_col = ('NumProduct','FirstA','FirstB','UniqueA','UniqueB','MostA')      
new_col_lst = dataset['ProductList'].apply(lambda x: extract_product(x))    
new_col_df = pd.DataFrame(new_col_lst.tolist(),columns =new_col)

data = pd.concat([dataset, new_col_df], axis=1)
# Time feature extraction
data['startTime'] = pd.to_datetime(data['startTime'])
data['endTime'] = pd.to_datetime(data['endTime'])
data['duration'] = data['endTime'] - data['startTime']
data['duration'] = data['duration'].astype('timedelta64[m]')
data['weekday'] = data['startTime'].dt.dayofweek
data['hour_24h'] = data['startTime'].dt.hour

drop_lst = ['user_id', 'startTime', 'endTime', 'ProductList']

data = data.drop(drop_lst,axis =1 )
print(data.head())
#Encoding for categories features
data = pd.get_dummies(data)

# Devide data set into train set , test set
X = data
Y = dataset_test['gender'].map({'female':1,'male':0})

val_size = 0.25
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size)
print(X.shape)
print(X_train.shape)
print(X_val.shape)
#sm = SMOTE(random_state=12, ratio = "minority")
#X_train, Y_train = sm.fit_sample(X_train, Y_train)
# Using Logistic Regression for classification
#clf = LogisticRegression(class_weight = {1:.1, 0:.35})
#Version 2: Using Random Forest instead Logistric Regression
clf = RandomForestClassifier(class_weight = {1:.1, 0:.35})
clf.fit(X_train,Y_train)
print("Evalute based on validation set")
print("f1 : " + " %s" % f1_score(Y_val, clf.predict(X_val)))
print("accuracy score" + " %s" % accuracy_score(Y_val, clf.predict(X_val)))
print("recall score micro: " + " %s" % recall_score(Y_val, clf.predict(X_val), average='micro'))
print("recall score macro: " + " %s" % recall_score(Y_val, clf.predict(X_val), average='macro'))
final_score = Counter(zip(Y, clf.predict(X)))
tp= final_score[1,1]
tn= final_score[0,0]
fp= final_score[0,1]
fn= final_score[1,0]
acc = (tp+tn)/len(Y)
recall1 = tp/(tp+fn)
recall2 = tn/(tn+fp)
recall_n = (recall1 + recall2) / 2
print("Performance on total data set:")
print("Accuracy " + "%.2f" % acc)
print("RECALL AVG FINAL  : " + "%.2f" % recall_n)   

print(X.shape)
#pd.DataFrame(clf.predict(X), columns=["predict"])
#pd.DataFrame(clf.predict(X), columns=["predict"]).to_csv("predict_result.csv")
pd.DataFrame(clf.predict(X)).to_csv("predict_result3.csv", index=False)
result_df = pd.concat([X,Y,pd.DataFrame(clf.predict(X), columns=["predict"])], axis =1)
result_df.to_csv('result.csv')