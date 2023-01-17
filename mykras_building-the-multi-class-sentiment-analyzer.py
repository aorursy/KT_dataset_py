# For the current version: 

!pip install --upgrade tensorflow

!pip install ktrain
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, shutil

import ktrain

from ktrain import text

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!wget http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
!ls -GFlash --color ./
!unzip bbc-fulltext.zip

!ls ./bbc
!ls ../base_dir
# if need deleting folder

!rm -rf ../base_dir



# 'business',  'entertainment',  'politics',  'sport',  'tech'

b_original_dataset_dir = './bbc/business'

e_original_dataset_dir = './bbc/entertainment'

p_original_dataset_dir = './bbc/politics'

s_original_dataset_dir = './bbc/sport'

t_original_dataset_dir = './bbc/tech'



base_dir = '../base_dir'

os.mkdir(base_dir)



train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



train_b_dir = os.path.join(train_dir, 'business')

os.mkdir(train_b_dir)

train_e_dir = os.path.join(train_dir, 'entertainment')

os.mkdir(train_e_dir)

train_p_dir = os.path.join(train_dir, 'politics')

os.mkdir(train_p_dir)

train_s_dir = os.path.join(train_dir, 'sport')

os.mkdir(train_s_dir)

train_t_dir = os.path.join(train_dir, 'tech')

os.mkdir(train_t_dir)



test_b_dir = os.path.join(test_dir, 'business')

os.mkdir(test_b_dir)

test_e_dir = os.path.join(test_dir, 'entertainment')

os.mkdir(test_e_dir)

test_p_dir = os.path.join(test_dir, 'politics')

os.mkdir(test_p_dir)

test_s_dir = os.path.join(test_dir, 'sport')

os.mkdir(test_s_dir)

test_t_dir = os.path.join(test_dir, 'tech')

os.mkdir(test_t_dir)



!ls ../base_dir
b_100 = ['{}.txt'.format(i) for i in range(100, 300)]

print(b_100[:5])

b_300 = ['{}.txt'.format(i) for i in range(300, 350)]

print(b_300[:5])



for fname in b_100:

    src = os.path.join(b_original_dataset_dir, fname)

    dst = os.path.join(train_b_dir, fname)

    shutil.copyfile(src, dst)



for fname in b_300:

    src = os.path.join(b_original_dataset_dir, fname)

    dst = os.path.join(test_b_dir, fname)

    shutil.copyfile(src, dst)



#

!head ./base_dir/test/business/300.txt
e_100 = ['{}.txt'.format(i) for i in range(100, 300)]

print(e_100[:5])

e_300 = ['{}.txt'.format(i) for i in range(300, 350)]

print(e_300[:5])



for fname in e_100:

    src = os.path.join(e_original_dataset_dir, fname)

    dst = os.path.join(train_e_dir, fname)

    shutil.copyfile(src, dst)



for fname in e_300:

    src = os.path.join(e_original_dataset_dir, fname)

    dst = os.path.join(test_e_dir, fname)

    shutil.copyfile(src, dst)



#

!head ./base_dir/test/entertainment/300.txt
p_100 = ['{}.txt'.format(i) for i in range(100, 300)]

print(p_100[:5])

p_300 = ['{}.txt'.format(i) for i in range(300, 350)]

print(p_300[:5])



for fname in p_100:

    src = os.path.join(p_original_dataset_dir, fname)

    dst = os.path.join(train_p_dir, fname)

    shutil.copyfile(src, dst)



for fname in p_300:

    src = os.path.join(p_original_dataset_dir, fname)

    dst = os.path.join(test_p_dir, fname)

    shutil.copyfile(src, dst)



#

!head ./base_dir/test/politics/300.txt
s_100 = ['{}.txt'.format(i) for i in range(100, 300)]

print(s_100[:5])

s_300 = ['{}.txt'.format(i) for i in range(300, 350)]

print(s_300[:5])



for fname in s_100:

    src = os.path.join(s_original_dataset_dir, fname)

    dst = os.path.join(train_s_dir, fname)

    shutil.copyfile(src, dst)



for fname in s_300:

    src = os.path.join(s_original_dataset_dir, fname)

    dst = os.path.join(test_s_dir, fname)

    shutil.copyfile(src, dst)



#

!head ./base_dir/test/sport/300.txt
t_100 = ['{}.txt'.format(i) for i in range(100, 300)]

print(t_100[:5])

t_300 = ['{}.txt'.format(i) for i in range(300, 350)]

print(t_300[:5])



for fname in t_100:

    src = os.path.join(t_original_dataset_dir, fname)

    dst = os.path.join(train_t_dir, fname)

    shutil.copyfile(src, dst)



for fname in t_300:

    src = os.path.join(t_original_dataset_dir, fname)

    dst = os.path.join(test_t_dir, fname)

    shutil.copyfile(src, dst)



#

!head ./base_dir/test/tech/300.txt
import ktrain

from ktrain import text as txt

ktrain.__version__
#

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder('../base_dir', 

                                                                       maxlen=75, 

                                                                       max_features=10000,

                                                                       preprocess_mode='bert',

                                                                       #train_test_names=['train', 'validation'],

                                                                       val_pct=0.1,

                                                                       classes=['business',  'entertainment',  'politics',  'sport',  'tech'])
# STEP 2: Create a Model and Wrap in Learner Object

model = text.text_classifier('bert', (x_train, y_train) , preproc=preproc)

learner = ktrain.get_learner(model, 

                             train_data=(x_train, y_train), 

                             val_data=(x_test, y_test), 

                             batch_size=32)
# STEP 3: Train the Model

learner.fit_onecycle(2e-5, 3, checkpoint_folder='../saved_weights')
learner.validate(val_data=(x_test, y_test), class_names=['business',  'entertainment',

                                                         'politics',  'sport ', 'tech'])
# Inspecting Misclassifications

learner.view_top_losses(n=3, preproc=preproc)
# Making Predictions on New Data

p = ktrain.get_predictor(learner.model, preproc)
p.get_classes()
# Predicting label for the text

p.predict("Today crisis is very hight.")
# Predicting label for the text

p.predict("Today people use a lot of devices.")
# Predicting label for the text

p.predict('Todays wordwide the crisis is the biggest in the last forty years.')
p.save('../mypred')
fin_bert_model = ktrain.load_predictor('../mypred')
# still works

fin_bert_model.predict('Todays wordwide the crisis is the biggest in the last forty years.')
# still works

fin_bert_model.predict('Todays wordwide crysis is the biggest in the last forty years.')
# still works

fin_bert_model.predict("Bob Dylan has released a song about the Kennedy assassination -- and it's 17 minutes long.")