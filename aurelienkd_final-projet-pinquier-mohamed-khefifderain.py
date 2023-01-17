import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

ROOT_FILENAME = "../input/"
#ROOT_FILENAME = ""
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
RESULT_FILENAME = 'res2.cv'
train = pd.read_csv(ROOT_FILENAME+TRAIN_FILENAME, parse_dates=['Dates'], index_col=False)
print(train.shape)
train.head(3)
test = pd.read_csv(ROOT_FILENAME+TEST_FILENAME, parse_dates=['Dates'], index_col=False)
print(test.shape)
test.head(3)
train.info()
categories = {c:i for i,c in enumerate(train['Category'].unique())}
cat_rev = {i:c for i,c in enumerate(train['Category'].unique())}
districts = {c:i for i,c in enumerate(train['PdDistrict'].unique())}
weekdays = {'Monday':0., 'Tuesday':1., 'Wednesday':2., 'Thursday': 3., 'Friday':4., 'Saturday':5., 'Sunday':6.}
weekdays2 = {'Monday':0., 'Tuesday':0., 'Wednesday':0., 'Thursday': 0., 'Friday':0., 'Saturday':1., 'Sunday':1}
def getHourZn(hour):
    if(hour >= 2 and hour < 8): return 1;
    if(hour >= 8 and hour < 12): return 2;
    if(hour >= 12 and hour < 14): return 3;
    if(hour >= 14 and hour < 18): return 4;
    if(hour >= 18 and hour < 22): return 5;
    if(hour < 2 or hour >= 22): return 6;
def define_address(addr):
    addr_type = 0.
    # Address types:
    #  Intersection: 1
    #  Residence: 0categories
    if '/' in addr and 'of' not in addr:
        addr_type = 1.
    else:
        add_type = 0.
    return addr_type
def feature_engineering(data):
    data['Day'] = data['Dates'].dt.day
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year-2003
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['Day_Num'] = [float(weekdays[w]) for w in data.DayOfWeek]
    data['WeekOfYear'] = data['Dates'].dt.weekofyear
    data['District_Num'] = [float(districts[t]) for t in data.PdDistrict]
    data['HourZn'] = preprocessing.scale(list(map(getHourZn, data['Dates'].dt.hour)))
    data['isWeekday'] = [float(weekdays2[w]) for w in data.DayOfWeek]
    data['X'] = preprocessing.scale(list(map(lambda x: x+122.4194, data.X)))
    data['Y'] = preprocessing.scale(list(map(lambda x: x-37.7749, data.Y)))
    data['Address_Type'] = list(map(define_address, data.Address))
#   data['HourZn'] = getHourZn(data['Dates'].dt.hour);
    return data
X_loc = ['X', 'Y', 'District_Num', 'Address_Type']
X_time = ['Minute', 'Hour']
X_date = ['Year','Month', 'Day', 'Day_Num', 'HourZn']
X_all = X_loc + X_time + X_date

train = feature_engineering(train)
train['Category_Num'] = [float(categories[t]) for t in train.Category]

X_train, X_test, y_train, y_test = train_test_split(train[X_all], train['Category_Num'], test_size = 0.2, random_state = 0)

test = feature_engineering(test)


#clf = RandomForestClassifier(max_features="log2", max_depth=11, n_estimators=24,
#                             min_samples_split=1000, oob_score=True).fit(X_train,y_train)
#y = clf.predict(X_test)

clf = RandomForestClassifier(max_features="log2", max_depth=14, n_estimators=25,
                             min_samples_split=300, oob_score=False).fit(X_train,y_train)
y = clf.predict(X_test)


recall_score(y,y_test, average='micro')
# recall_score_micro : 0.29731792039177724
recall_score(y,y_test, average='macro')
# recall_score_macro : 0.09619918771835194
recall_score(y,y_test, average='weighted')
# recall_score_weighted : 0.29731792039177724

precision_score(y,y_test, average='micro')
# precision_score_micro : 0.29731792039177724
precision_score(y,y_test, average='macro')
# precision_score_macro : 0.07442702033391868
precision_score(y,y_test, average='weighted')
# precision_score_weighted : 0.567191672388691

accuracy_score(y,y_test)
# accuracy_score : 0.29731792039177724


f1_score(y,y_test, average='micro')
# F1_micro : 0.29731792039177724
f1_score(y,y_test, average='macro')
# F1_macro : 0.0683830234838435
f1_score(y,y_test, average='weighted')
# F1_weighted : 0.3704176711997292

#clf = RandomForestClassifier(max_features="log2", max_depth=14, n_estimators=25,
#                             min_samples_split=300, oob_score=False).fit(train[X_all],train['Category_Num'])
#y = clf.predict_proba(test[X_all])
#submission = pd.DataFrame({cat_rev[p] : [y[i][p] for i in range(len(y))] for p in range(len(y[0]))})

#submission['Id'] = [i for i in range(len(submission))]
#submission.to_csv('result_RN.csv', index=False)
# # 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y = knn.predict(X_test)


recall_score(y,y_test, average='micro')
# recall_score_micro : 0.1996640282444052
recall_score(y,y_test, average='macro')
# recall_score_macro : 0.07300143475941939
recall_score(y,y_test, average='weighted')
# recall_score_weighted : 0.1996640282444052

precision_score(y,y_test, average='micro')
# precision_score_micro : 0.1996640282444052
precision_score(y,y_test, average='macro')
# precision_score_macro : 0.052448254827175196
precision_score(y,y_test, average='weighted')
# precision_score_weighted : 0.2742525234494359

accuracy_score(y,y_test)
# accuracy_score : 0.1996640282444052


f1_score(y,y_test, average='micro')
# F1_micro : 0.19966402824440524
f1_score(y,y_test, average='macro')
# F1_macro : 0.05236657213517695
f1_score(y,y_test, average='weighted')
# F1_weighted : 0.22499597882768185


#y = knn.predict_proba(X_test)

#submission = pd.DataFrame({cat_rev[p] : [y[i][p] for i in range(len(y))] for p in range(len(y[0]))})
#submission['Id'] = [i for i in range(len(submission))]

#submission.to_csv("submission_knn.csv", index=False) 


from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train,y_train)
y = lgr.predict(X_test)


recall_score(y,y_test, average='micro')
# recall_score_micro : 0.21990775012812483
recall_score(y,y_test, average='macro')
# recall_score_macro : 0.028791943715668795
recall_score(y,y_test, average='weighted')
# recall_score_weighted : 0.21990775012812483

precision_score(y,y_test, average='micro')
# precision_score_micro : 0.21990775012812483
precision_score(y,y_test, average='macro')
# precision_score_macro : 0.03294464738579727
precision_score(y,y_test, average='weighted')
# precision_score_weighted : 0.6483392941859246

accuracy_score(y,y_test)
# accuracy_score : 0.21990775012812483


f1_score(y,y_test, average='micro')
# F1_micro : 0.21990775012812483
f1_score(y,y_test, average='macro')
# F1_macro : 0.0207335350195717
f1_score(y,y_test, average='weighted')
# F1_weighted : 0.3166197259646911


#y = lgr.predict_proba(test[X_all])



#submission = pd.DataFrame({cat_rev[p] : [y[i][p] for i in range(len(y))] for p in range(len(y[0]))})
#submission['Id'] = [i for i in range(len(submission))]
#submission.to_csv('result_logistic.csv', index=False)


#seed = 37

#model = xgb.XGBClassifier(objective='multi:softprob',
#                          n_estimators=45,
#                          learning_rate=1.0,
#                          max_depth=1,
#                          max_delta_step=1,
#                          nthread=-1,
#                          seed=seed)
#model.fit(X_train, y_train)
#predictions = model.predict_proba(X_test)


#y = model.predict_proba(test[X_all])

#submission = pd.DataFrame({cat_rev[p] : [y[i][p] for i in range(len(y))] for p in range(len(y[0]))})
#submission['Id'] = [i for i in range(len(submission))]
#submission.to_csv('result_logistic.csv', index=False)

