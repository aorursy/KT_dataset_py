# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

sys.path.extend(['../input/drqa-reader'])
import warnings

warnings.simplefilter("ignore")



from drqa.reader import Predictor



import torch



predictor = Predictor(num_workers=0, normalize=True)



if torch.cuda.is_available():

    predictor.cuda()

else:

    predictor.cpu()
import prettytable

import time



def process(document, question, candidates=None, top_n=1):

    print(question)

    t0 = time.time()

    predictions = predictor.predict(document, question, candidates, top_n)

    table = prettytable.PrettyTable(['Rank', 'Span', 'Score', 'Start Token Index', 'End Token Index'])

    for i, p in enumerate(predictions, 1):

        table.add_row([i, p[0], p[1], p[2][0], p[2][1]])

    print(table)

    print('Time: %.4f' % (time.time() - t0))

    print('\n')
text = 'Mary had a little lamb, whose fleece was white as snow. And everywhere that Mary went the lamb was sure to go.'

process(text, 'What color is Mary\'s lamb?')
text = '''

  Bob needs a job.

  There is a job for Bob.

  Rob can use Bob on his farm. 

  Bob can pick corn on the cob. 

  Rob and Bob each pick the cobs. 

  Bob likes his job with Rob. 

  There is a mob at the farm. 

  They want the cobs of corn. 

  Rob and Bob sell the cobs. 

  They like their jobs. 

  They like to sell the cobs. 

  The mob likes the cobs of corn.

  '''



questions = [

  'What does Bob need?',

  'What is Bob’s job?',

  'Who loves the cobs of corn?',

  'Who likes to sell the cobs of corn?',

  'What does Bob and Rob like to sell?'

]



for question in questions:

  process(text, question)