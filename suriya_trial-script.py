import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn import preprocessing

df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

df['Date']=pd.to_datetime(pd.Series(df['Original_Quote_Date']))
df['Year']=df['Date'].apply(lambda x: int(str(x)[:4]))
df['Month']=df['Date'].apply(lambda x: int(str(x)[5:7]))
df['Date']=df['Date'].apply(lambda x: int(str(x)[8:10]))
df['Field10'].apply(lambda x : int(x.replace(',','')) )

test['Date']=pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test['Year']=test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month']=test['Date'].apply(lambda x: int(str(x)[5:7]))
test['Date']=test['Date'].apply(lambda x: int(str(x)[8:10]))
test['Field10'].apply(lambda x : int(x.replace(',','')))

label=df['QuoteConversion_Flag']
df.drop('QuoteConversion_Flag',axis=1,inplace=True)
number=test['QuoteNumber']
drop_columns=['Original_Quote_Date','QuoteNumber']
for names in drop_columns:
        df.drop(names,axis=1,inplace=True)
        test.drop(names,axis=1,inplace=True)
clf=xgb.XGBClassifier(max_depth=7,learning_rate=0.03,n_estimators=650,subsample=0.86,seed=50)
for f in df.columns:
    if df[f].dtypes=='object':
        encoder=preprocessing.LabelEncoder()
        encoder.fit( list(df[f])+list(test[f]) )
        df[f]=encoder.transform(list(df[f].values))
        test[f]=encoder.transform(list(test[f].values))


df.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)
clf.fit(df,label)
output=clf.predict_proba(test)[:,1]
sample=pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag=output
sample.to_csv('final.csv',index=False)