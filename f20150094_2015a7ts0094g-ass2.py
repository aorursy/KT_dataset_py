import pandas as pd

import numpy as np



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.head()



train_df.describe()



print(len(train_df))



def process(df):

    dft = df.copy(deep=True)

    for col in dft.columns:

        dft[col] = dft[col].apply(lambda x : None if x=='?' else x)

    for column in dft.columns:

        dft[column].fillna(dft[column].mode()[0], inplace=True)

    dfd = pd.get_dummies(dft)

    if 'Class' in dfd.columns:

        return dfd.drop(['Class'],axis=1),dfd['Class']

    else:

        return dfd,None



from sklearn.ensemble import RandomForestClassifier

X,y = process(train_df)

Xtest,yTest = process(test_df)



fin_cols = set(list(X.columns)).intersection(set(list(Xtest.columns)))

X = X[list(fin_cols)]

Xtest = Xtest[list(fin_cols)]



rfc = RandomForestClassifier(class_weight={0:1,1:10000000}).fit(X,y)



from sklearn.cross_validation import cross_val_score, cross_val_predict

from sklearn import metrics

scores = cross_val_score(rfc, X, y, cv=6)

print ("Cross-validated scores:",scores)



yPredict = rfc.predict(Xtest)



print(len(yPredict))



submit_df = pd.DataFrame({'ID':list(test_df['ID']),'Class':yPredict})



submit_df.describe()



submit_df.to_csv('submit2.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64 

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(submit_df)