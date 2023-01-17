!ls -lrt /kaggle/input

!java -version

!javac -d /kaggle/working/  /kaggle/input/code/java/CORD19/src/DTBean.java

!javac -d /kaggle/working/  /kaggle/input/code/java/CORD19/src/Segment.java

!javac -d /kaggle/working/  /kaggle/input/code/java/CORD19/src/Text.java

!javac -d /kaggle/working/  /kaggle/input/code/java/CORD19/src/Word2Num.java

!javac -d /kaggle/working/ -cp /kaggle/working/ /kaggle/input/code/java/CORD19/src/SentenceSplitter.java

!javac -d /kaggle/working/ -cp /kaggle/working/ /kaggle/input/code/java/CORD19/src/NovelTherapeuticsPipeline.java

!javac -d /kaggle/working/ -cp /kaggle/working/ /kaggle/input/code/java/CORD19/src/BERTInputPreprocess.java

!javac -d /kaggle/working/ -cp /kaggle/working/ /kaggle/input/code/java/CORD19/src/BERTPostProcessing.java

!javac -d /kaggle/working/ -cp /kaggle/working/:/kaggle/input/code/java/CORD19/lib/* /kaggle/input/code/java/CORD19/src/BERTOutputProcessor.java



!java -cp /kaggle/working/ NovelTherapeuticsPipeline

!java -cp /kaggle/working/ BERTInputPreprocess

!ls -lrt /kaggle/input

!java -cp /kaggle/working/:/kaggle/input/code/java/CORD19/lib/* BERTOutputProcessor

!java -cp /kaggle/working/:/kaggle/input/code/java/CORD19/lib/* BERTPostProcessing
import nltk

import re

nltk.downloader.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd



regex=r"\b(low|reduce|stop)(.*)(infection|fatal|mortal|risk|cytokine storm|concentration|death|adverse)+"

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



def checkNegFP(text):

    m=re.search(regex,text)

    if(m!=None):

        return True

    else:

        return False



input_file = '/kaggle/working/novel_th_ab_wbert_processed.tsv'

tdf = pd.read_csv(input_file,sep='\t',converters={"Clinical Improvement (Y/N)":str})

for i, r in tdf.iterrows():

    r[1]=r[1].title() 

    score_dict = SentimentIntensityAnalyzer().polarity_scores(r[9]);

    if(score_dict['neg']>score_dict['pos']):

        if checkNegFP(r[9]):

            r[9]='Y'

        else:

            r[9]='N'

    elif(score_dict['neg']<score_dict['pos']):

        r[9]='Y'

    else:

        r[9]='-'

tdf.to_csv("/kaggle/working/novel_th.csv")
tdf
!java -cp /kaggle/working/ NovelTherapeuticsPipeline q1

!java -cp /kaggle/working/ BERTInputPreprocess q1
!java -cp /kaggle/working/:/kaggle/input/code/java/CORD19/lib/* BERTOutputProcessor q1

!java -cp /kaggle/working/:/kaggle/input/code/java/CORD19/lib/* BERTPostProcessing q1
input_file = '/kaggle/working/hgs_ab_wbert_processed.tsv'

df = pd.read_csv(input_file,sep='\t',converters={"Clinical Improvement (Y/N)":str})

for i, r in df.iterrows():

    r[1]=r[1].title() 

    score_dict = SentimentIntensityAnalyzer().polarity_scores(r[9]);

    if(score_dict['neg']>score_dict['pos']):

        if checkNegFP(r[9]):

            r[9]='Y'

        else:

            r[9]='N'

    elif(score_dict['neg']<score_dict['pos']):

        r[9]='Y'

    else:

        r[9]='-'

df.to_csv("/kaggle/working/hgs.csv")
df