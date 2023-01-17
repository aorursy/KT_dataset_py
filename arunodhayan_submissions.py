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
import pandas as pd

import os

test_file_path = "/kaggle/input/birdsong-recognition/test_audio"

test_df = pd.read_csv("/kaggle/input/birdsong-recognition/test.csv")

submission_df = pd.read_csv("/kaggle/input/birdsong-recognition/sample_submission.csv")



if os.path.exists(test_file_path):

    submission_df = predict_submission(test_df, test_file_path)



submission_df[["row_id","birds"]].to_csv('submission.csv', index=False)

submission_df.head()