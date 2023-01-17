import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from plotly import graph_objs as go
from PIL import Image
dftrain = pd.read_csv('C:/Users/user/Desktop/IVY WORK BOOK/Kaggle submission/Tweet Sentiment Extraction/train.csv')
dftest = pd.read_csv('C:/Users/user/Desktop/IVY WORK BOOK/Kaggle submission/Tweet Sentiment Extraction/test.csv')
dftrain.head(30)
dftrain=dftrain.dropna()
dftrain=dftrain.drop_duplicates()
dftest
dftest=dftest.dropna()
dftest=dftest.drop_duplicates()
sns.countplot(x='sentiment',data=dftrain)
def positive_neutral(inpdata):
    if(inpdata=='positive'):
        return(1)
    elif(inpdata=='negative'):
        return(0)
    else:
        return(2)  
dftrain['sentiment_value']=dftrain['sentiment'].apply(positive_neutral)
dftest['sentiment_value']=dftest['sentiment'].apply(positive_neutral)
def text_preprocess(inpdata1,inpdata2):
    import re
    if(inpdata2==2):
        return(inpdata1)
    else:
        cleanedArticle1=re.sub(r'[?|$|(),"".@#=><|!]Â&*/',r' ',inpdata1)
        cleanedArticle2=re.sub(r'[^a-z A-Z 0-9]',r' ',cleanedArticle1)
        cleanedArticle3=cleanedArticle2.lower()
        #cleanedArticle4=re.sub(r'\b\w{1,1}\b', ' ',cleanedArticle3)
        cleanedArticle4=re.sub(r'https?://\S+|www\.\S+',r' ',cleanedArticle3)
        cleanedArticle5=re.sub(r' +', ' ',cleanedArticle4)
        return(cleanedArticle5)
for i in range (0, len(dftrain)):
    dftrain['process_text'].iloc[i]=text_preprocess(dftrain['text'].iloc[i],dftrain['sentiment_value'].iloc[i])
for j in range (0, len(dftest)):
      dftest['process_text'].iloc[j]=text_preprocess(dftest['text'].iloc[j],dftest['sentiment_value'].iloc[j])

dftest.head(60)
dftest['selected_text']=dftest['textID']
def simple_text_blob(inpdata1,inpdata2):
    from textblob import TextBlob
    if(inpdata2==2):
        return(inpdata1)
    elif(inpdata2==1):
        blob=TextBlob(inpdata1)
        inp=blob.tokenize()
        for word in inp:
            if(TextBlob(word).polarity>0):
                return(word)
    else:
        blob=TextBlob(inpdata1)
        inp=blob.tokenize()
        for word in inp:
              if(TextBlob(word).polarity<0):
                return(word) 
for j in range(0,len(dftest)):
    dftest['selected_text'].iloc[j]=simple_text_blob(dftest['process_text'].iloc[j],dftest['sentiment_value'].iloc[j])

#for j in range(0,len(dftrain)):
    #dftrain['selected_text'].iloc[j]=simple_text_blob(dftrain['text'].iloc[j],dftrain['sentiment_value'].iloc[j])
dftrain[['process_text','Sel_text','selected_text','sentiment_value']].tail(50)
dftrain=dftrain.drop('Sel_text', axis=1)
dftrain.shape
sample_df=pd.DataFrame(dftest,columns=['textID','selected_text'])
sample_df.to_csv('sample_df.csv')
