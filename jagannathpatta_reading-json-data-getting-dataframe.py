# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json
def json_to_dataframe(file):



    f = open ( file , "r") 

    data = json.loads(f.read())               #loading the json file.

    iid = []                                  

    tit = []                                  #Creating empty lists to store values.

    con = []

    Que = []

    Ans_st = []

    Txt = []

    

    for i in range(len(data['data'])):       #Root tag of the json file contains 'title' tag & 'paragraphs' list.

        

        title = data['data'][i]['title']

        for p in range(len(data['data'][i]['paragraphs'])):  # 'paragraphs' list contains 'context' tag & 'qas' list.

            

            context = data['data'][i]['paragraphs'][p]['context']

            for q in range(len(data['data'][i]['paragraphs'][p]['qas'])):  # 'qas' list contains 'question', 'Id' tag & 'answers' list.

                

                question = data['data'][i]['paragraphs'][p]['qas'][q]['question']

                Id = data['data'][i]['paragraphs'][p]['qas'][q]['id']

                for a in range(len(data['data'][i]['paragraphs'][p]['qas'][q]['answers'])): # 'answers' list contains 'ans_start', 'text' tags. 

                    

                    ans_start = data['data'][i]['paragraphs'][p]['qas'][q]['answers'][a]['answer_start']

                    text = data['data'][i]['paragraphs'][p]['qas'][q]['answers'][a]['text']

                    

                    tit.append(title)

                    con.append(context)

                    Que.append(question)                    # Appending values to lists

                    iid.append(Id)

                    Ans_st.append(ans_start)

                    Txt.append(text)



    print('Done')      # for indication perpose.

    new_df = pd.DataFrame(columns=['Id','title','context','question','ans_start','text']) # Creating empty DataFrame.

    new_df.Id = iid

    new_df.title = tit           #intializing list values to the DataFrame.

    new_df.context = con

    new_df.question = Que

    new_df.ans_start = Ans_st

    new_df.text = Txt

    print('Done')      # for indication perpose.

    final_df = new_df.drop_duplicates(keep='first')  # Dropping duplicate rows from the create Dataframe.

    return final_df
dev_data = json_to_dataframe('/kaggle/input/stanford-question-answering-dataset/dev-v1.1.json')

dev_data
dev_data.info()
train_data = json_to_dataframe('/kaggle/input/stanford-question-answering-dataset/train-v1.1.json')

train_data
train_data.info()