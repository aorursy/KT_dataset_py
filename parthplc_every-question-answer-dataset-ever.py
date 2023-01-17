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
Trainhp = pd.read_json("/kaggle/input/hotpotqa-question-answering-dataset/hotpot_train_v1.1.json")
Trainhp.head()
TrainQuac = pd.read_csv("../input/quaccoqa/dataquest/Quac_train.csv")
TrainQuac.head()
len(TrainQuac)
TrainQuac["context"][0]

TrainCoqa = pd.read_csv("../input/quaccoqa/dataquest/CoQA_train.csv")
TrainCoqa.head()
TrainSquad = pd.read_csv("../input/squad-20-csv-file/squad_csv/train-squad.csv")
TrainSquad.head()
len(TrainSquad)
TrainSquad["context"][0]
TrainQasc = pd.read_json("../input/question-answering-via-sentence-composition-qasc/QASC_Dataset/train.jsonl",lines = True)
TrainQasc.head()
len(TrainQasc)
TrainBoolq = pd.read_json("../input/boolq-dataset/train.jsonl",lines = True)
len(TrainBoolq)
TrainMultirc = pd.read_json("../input/multirc-dataset/splitv2/train_456-fixedIds.json")
TrainMultirc["data"][0]
TrainDrop = pd.read_json("../input/dropdataset/drop_dataset/drop_dataset_train.json").T
TrainDrop
TrainDrop["qa_pairs"][100][0]["answer"]["date"]
TrainDropping = pd.DataFrame(columns=['passage', 'question', 'number','date'])

TrainDrop["qa_pairs"][35][1]["answer"]["date"]
TrainDropping
for i in range(len(TrainDrop)):
    print(i)
    for j in range(len(TrainDrop["qa_pairs"][i])):
        TrainDropping = TrainDropping.append({"passage": TrainDrop["passage"][i], "question": TrainDrop["qa_pairs"][i][j]["question"], 
                              "number": TrainDrop["qa_pairs"][i][j]["answer"]["number"],"date" : TrainDrop["qa_pairs"][i][j]["answer"]["date"] }, ignore_index=True)
       
    
    
TrainDropping.to_csv("train-drop.csv",index = False)
TrainDropping["question"][5]
TrainDropping["number"].value_counts()
TestDrop = pd.read_json("../input/dropdataset/drop_dataset/drop_dataset_dev.json").T
TestDropping = pd.DataFrame(columns=['passage', 'question', 'number','date'])

for i in range(len(TestDrop)):
    print(i)
    for j in range(len(TestDrop["qa_pairs"][i])):
        TestDropping = TestDropping.append({"passage": TestDrop["passage"][i], "question": TestDrop["qa_pairs"][i][j]["question"], 
                              "number": TestDrop["qa_pairs"][i][j]["answer"]["number"],"date" : TestDrop["qa_pairs"][i][j]["answer"]["date"] }, ignore_index=True)
       
    
    
TestDropping.head()
TestDropping["number"][0]
TrainDropping = TrainDropping[TrainDropping.number != '']
TestDropping = TestDropping[TestDropping.number != '']

TrainDropping.shape


TestDropping.shape
TrainDropping.to_csv("droptrain.csv",index = False)
TestDropping.to_csv("droptest.csv",index = False)
