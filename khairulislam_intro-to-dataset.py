# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.display import HTML

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



PATH = '/kaggle/input/tensorflow2-question-answering/'

!ls {PATH}
train_head = []

nrows = 5



with open(PATH+'simplified-nq-train.jsonl', 'rt') as f:

    for i in range(nrows):

        train_head.append(json.loads(f.readline()))



train_df = pd.DataFrame(train_head)
train_df
index = 0
ans = train_df.loc[index,'document_text'].split()

' '.join(ans[:30])
train_df.loc[index,'question_text']
#  this function taken from this kernel

def show_example(example_id):

#     example_id = 5655493461695504401



    example = train_df[train_df['example_id']==example_id]

    document_text = example['document_text'].values[0]

    question = example['question_text'].values[0]



    annotations = example['annotations'].values[0]

    la_start_token = annotations[0]['long_answer']['start_token']

    la_end_token = annotations[0]['long_answer']['end_token']

    long_answer = " ".join(document_text.split(" ")[la_start_token:la_end_token])

    short_answers = annotations[0]['short_answers']

    sa_list = []

    for sa in short_answers:

        sa_start_token = sa['start_token']

        sa_end_token = sa['end_token']

        short_answer = " ".join(document_text.split(" ")[sa_start_token:sa_end_token])

        sa_list.append(short_answer)

    

    document_text = document_text.replace(long_answer,'<LALALALA>')

    sa=False

    la=''

    for sa in short_answers:

        sa_start_token = sa['start_token']

        sa_end_token = sa['end_token']

        for i,laword in enumerate(long_answer.split(" ")):

            ind = i+la_start_token

            if ind==sa_start_token:

                la = la+' SASASASA'+laword

            elif ind==sa_end_token-1:

                la = la+' '+laword+'SESESESE'

            else:

                la = la+' '+laword

    #print(la)

    html = '<div style="font-weight: bold;font-size: 20px;color:#00239CFF">Example Id</div><br/>'

    html = html + '<div>' + str(example_id) + '</div><hr>'

    html = html + '<div style="font-weight: bold;font-size: 20px;color:#00239CFF">Question</div><br/>'

    html = html + '<div>' + question + ' ?</div><hr>'

    html = html + '<div style="font-weight: bold;font-size: 20px;color:#00239CFF">Document Text</div><br/>'

    

    if la_start_token==-1:

        html = html + '<div>There are no answers found in the document</div><hr>'

    else:

        la = la.replace('SASASASA','<span style="background-color:#C7D3D4FF; padding:5px"><font color="#000">')

        la = la.replace('SESESESE','</font></span>')

        document_text = document_text.replace('<LALALALA>','<div style="background-color:#603F83FF; padding:5px"><font color="#fff">'+la+'</font></div>')



        #for simplicity, trim words from end of the document

        html = html + '<div>' + " ".join(document_text.split(" ")[:la_end_token+200]) + ' </div>'

    display(HTML(html))
show_example(train_df.loc[index, "example_id"])
train_df.loc[index,'long_answer_candidates']
train_df.loc[index,'long_answer_candidates'][:6]
# as this is train dataset, only one annotation

annotation = train_df.loc[index, "annotations"][0]

annotation
long_candidate = annotation['long_answer']

long_candidate


' '.join(ans[long_candidate['start_token']: long_candidate['end_token']])
for short_candidate in annotation['short_answers']:

    print(' '.join(ans[short_candidate['start_token']: short_candidate['end_token']]))
annotation['yes_no_answer']
train_df.loc[index,'document_url']