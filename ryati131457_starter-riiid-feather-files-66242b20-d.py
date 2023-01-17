import pandas as pd

import glob
for f in glob.glob('/kaggle/input/*.feather'):

    print(f)
%%time

questions = pd.read_feather('/kaggle/input/questions.feather')

lectures = pd.read_feather('/kaggle/input/lectures.feather')

train = pd.read_feather('/kaggle/input/train.feather')

train.info()
train.head()
train.prior_question_had_explanation = train.prior_question_had_explanation.astype('Int8')

train