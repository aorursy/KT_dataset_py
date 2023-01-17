
# coding: utf-8

# In[108]:

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# In[187]:

df = pd.read_csv('../input/train.csv')
tdf = pd.read_csv('../input/test.csv')

# In[191]:

catdf = df.select_dtypes(['object'])
tcatdf = tdf.select_dtypes(['object'])
catdf.fillna("NA",inplace=True)
tcatdf.fillna("NA",inplace=True)

v22df = catdf.loc[:,['v22']]
tv22df = tcatdf.loc[:,['v22']]
v22df.fillna("NA",inplace=True)
tv22df.fillna("NA",inplace=True)
del catdf['v22']
del tcatdf['v22']

# In[195]:

vec = DictVectorizer()
vectorized_catdf = vec.fit_transform(catdf.to_dict('records')).toarray()
tvectorized_catdf = vec.transform(tcatdf.to_dict('records')).toarray()

# In[197]:

numtrainrows = v22df['v22'].shape[0]
numtestrows = tv22df['v22'].shape[0]

totalv22 = pd.concat([v22df,tv22df])
x = pd.Categorical(totalv22['v22'])

# In[198]:

v22columncodes = x.codes
v22column = v22columncodes[:numtrainrows]
tv22column = v22columncodes[numtrainrows:]
# In[199]:

v22column = np.matrix(v22column).T
tv22column = np.matrix(tv22column).T

# In[200]:

vectorized_catdf = np.hstack((vectorized_catdf,v22column))
tvectorized_catdf = np.hstack((tvectorized_catdf,tv22column))

# In[202]:

df_withoutid_target = df.drop(['ID'],axis=1)
tdf_withoutid_target = tdf.drop(['ID'],axis=1)


# In[205]:

categorical_columns = df.select_dtypes(['object']).columns
df_without_categ_also = df_withoutid_target.drop(categorical_columns,axis=1)
tdf_without_categ_also = tdf_withoutid_target.drop(categorical_columns,axis=1)

df_without_categ_also.fillna(df.mean(axis=0),inplace=True)
tdf_without_categ_also.fillna(df.mean(axis=0),inplace=True)
# In[209]:

noncat_columns = df_without_categ_also.as_matrix()
tnoncat_columns = tdf_without_categ_also.as_matrix()
print(noncat_columns.shape,vectorized_catdf.shape,tnoncat_columns.shape,tvectorized_catdf.shape)
final_data_matrix = np.hstack((noncat_columns,vectorized_catdf))
tfinal_data_matrix = np.hstack((tnoncat_columns,tvectorized_catdf))
header = range(final_data_matrix.shape[1])
theader = range(tfinal_data_matrix.shape[1])

index = range(final_data_matrix.shape[0])
tindex = range(tfinal_data_matrix.shape[0])

df = pd.DataFrame(final_data_matrix,index=index,columns=header)
testdf = pd.DataFrame(tfinal_data_matrix,index=tindex,columns=theader)
testdfforid =pd.read_csv("../input/test.csv")
labeldf = df.loc[:,[0]]
iddf = testdfforid.loc[:,['ID']]
df.drop([0],axis=1,inplace=True)
trainx = df.as_matrix()
trainy = labeldf.as_matrix()
testx = testdf.as_matrix()
testids = iddf.as_matrix()
def testandsubmit(clf,test):
    predictions = clf.predict_proba(test)
    predictions = [p[1] for p in predictions]
    f = open('../working/submission.csv','wb')
    f.write("ID,PredictedProb\n")
    for i in range(len(predictions)):
        f.write(str(testids[i][0])+","+str(predictions[i]) + "\n")
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    global processedtrainx,processedtrainy,trainx,trainy,testx
    pca = PCA(n_components=15)
    scaler = StandardScaler()
    print(trainx.shape,testx.shape)
    processedtrainx = scaler.fit_transform(trainx)
    #processedtrainx = pca.fit_transform(processedtrainx)
    #print(pca.explained_variance_ratio_)
    processedtrainy = trainy
   
    testx = scaler.transform(testx)
    #testx = pca.transform(testx)
    
    X_train, X_test, y_train, y_test = train_test_split(processedtrainx, processedtrainy, test_size=0.1, random_state=0)
    clf = RandomForestClassifier(n_estimators=80,max_features=60,max_depth=40,criterion='entropy',class_weight='balanced_subsample')
    clf.fit(X_train,y_train)
    probabs = clf.predict_proba(X_train)
    probabs = [p[1] for p in probabs]
    print(log_loss(y_train, np.array(probabs)  ))
    validatepredict = clf.predict_proba(X_test)
    validatepredict = [p[1] for p in validatepredict]
    print(log_loss( y_test,validatepredict))
main()
