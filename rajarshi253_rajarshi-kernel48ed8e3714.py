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
%ls '/kaggle/input'
import pandas as pd

data = pd.read_csv('/kaggle/input/submission/prediction_stg2.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
import pandas as pd

data = pd.read_csv('/kaggle/input/linreg/lin_reg_prediction_stg2.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
import pandas as pd

data = pd.read_csv('/kaggle/input/supportvector/svm_prediction_stg2.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/randomforest/random_forest_prediction_stg2.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/decisiontree/decision_tree_prediction_stg2.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/lstmcnn/lstm_cnn_prediction_stg2.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/ridgereg/ridge_reg_prediction_stg2.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/stacking/stacking_prediction_stg2.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/ridgeregmodif/ridge_reg_prediction_stg2_modif.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/svmmodif/svm_prediction_stg2_modif.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/lgbmmodif/lgbm_prediction_stg2_modif.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()
import pandas as pd

data = pd.read_csv('/kaggle/input/stackingmodif/stacking_prediction_stg2_modif.csv')

data.to_csv('/kaggle/working/submission.csv', index=False)
result = pd.read_csv('/kaggle/working/submission.csv')

result.head()