import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
list_csv =[]



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if('.csv' in os.path.join(dirname, filename)):

            list_csv.append(os.path.join(dirname, filename))



            

list_csv = list(set(list_csv) - set(['/kaggle/input/tweet-sentiment-extraction/train.csv',

 '/kaggle/input/tweet-sentiment-extraction/test.csv',

 '/kaggle/input/tweet-sentiment-extraction/sample_submission.csv']))
list_csv
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")



list_df =[]



for i in list_csv:

    df = pd.read_csv(i)

    df_test = pd.merge(df_test, df, on = 'textID')
df_test.head()
df_sentiment = df_test.copy()

df_sentiment.columns = ['textID', 'text', 'sentiment'] + list('selected_text_' + pd.Series(range(len(list_csv))).astype('str'))

df_sentiment.head()
count = 0



dictionary = {}





for row in df_sentiment.iterrows():

    #print(row)

    list_elements = [str(i).lower().strip() for i in row[1].to_list()[3:]]

    cleaned_text = ' '.join(list_elements)

    cleaned_text = cleaned_text.split(' ')



    df_idf = pd.Series(cleaned_text).value_counts()/(len(list_csv)-1)



    dictionary[row[1].textID] = ' '.join(df_idf[df_idf > .7].index)

    

    count = count+1

df = pd.DataFrame.from_dict(dictionary, columns=['selected_text'], orient = 'index').reset_index()



df.columns = ['textID', 'selected_text']



df.head()
df_sample = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
df_submission = pd.merge(df_sample,df, on = 'textID', how = 'left' )
df_submission  = df_submission.fillna('')

df_submission['selected_text']= df_submission.selected_text_x + df_submission.selected_text_y

df_submission.head()
df_submission[['textID', 'selected_text']].to_csv("submission.csv", index = False)