import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
#reading data
df = pd.read_csv("../input/modified_census_data.csv")
df.sample()
#spliting data, I understood columns starting with LabelP are the modified ones.
df_original = pd.DataFrame(df,columns=[ 'LABEL', 'age', 'capital_gain', 'capital_loss',
       'education', 'education_num', 'fnlwgt', 'hours_per_week',
       'marital_status', 'native_country', 'occupation', 'race',
       'relationship', 'sex', 'workLABEL'])
df_discretised = pd.DataFrame(df,columns = ['LabelPage', 'LabelPcapital_gain',
       'LabelPcapital_loss', 'LabelPeducation', 'LabelPeducation_num',
       'LabelPhours_per_week', 'LabelPmarital_status', 'LabelPnative_country',
       'LabelPoccupation', 'LabelPrace', 'LabelPrelationship', 'LabelPsex',
       'LabelPworkLABEL','LABEL'])
df.dtypes.unique()
#Making all non numeric variables numeric.
def convert(data):
    col_to_transform = data.dtypes[data.dtypes!=int].index
    for i in col_to_transform:
        if data[i].unique().shape[0]>20:
            data[i] = LabelEncoder().fit_transform(data[i])
        else :
            data = pd.concat([data,pd.get_dummies(data[i])],axis = 1)
            del data[i]
    return data
df_original = convert(df_original)
df_discretised = convert(df_discretised)
#Cross validation
def cross_val(df,ratio = 0.5):
    train = df.sample(frac = ratio)
    test = df[~df.index.isin(train.index)].dropna()
    return train,test
#original data benchmark
rf = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=10,n_jobs = -1)
m = LogisticRegression()
score_lr = []
score_rf = []
for i in range(5):
    train,test= cross_val(df_original)
    Ytrain=train['LABEL'];del train['LABEL']
    Ytest=test['LABEL'];del test['LABEL']
    
    m.fit(train,Ytrain)
    rf.fit(train,Ytrain)
    temp_lr = m.predict(test)
    temp_rf = rf.predict(test)
    score_rf.append(sum(temp_rf == Ytest)/len(Ytest))
    score_lr.append(sum(temp_lr == Ytest)/len(Ytest))
print('Random forest : '+ str(np.mean(score_rf)))
print('Logistic Regression : '+ str(np.mean(score_lr)))
#discretised data benchmark
rf = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=10,n_jobs = -1)
m = LogisticRegression()
score_lr = []
score_rf = []
for i in range(5):
    train,test= cross_val(df_discretised)
    Ytrain=train['LABEL'];del train['LABEL']
    Ytest=test['LABEL'];del test['LABEL']
    
    m.fit(train,Ytrain)
    rf.fit(train,Ytrain)
    temp_lr = m.predict(test)
    temp_rf = rf.predict(test)
    score_rf.append(sum(temp_rf == Ytest)/len(Ytest))
    score_lr.append(sum(temp_lr == Ytest)/len(Ytest))
print('Random forest : '+ str(np.mean(score_rf)))
print('Logistic Regression : '+ str(np.mean(score_lr)))