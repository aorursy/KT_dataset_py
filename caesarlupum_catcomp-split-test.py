import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
submission = pd.read_csv('../input/catcomp/sub_2019-10-18_11-51-06.csv')

possible_public = int(len(submission)*0.80)

submission.iloc[possible_public:,1] = 0

from datetime import datetime

submission.to_csv(

    'sub_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', 

    index=False)
aristocats  = pd.read_csv('../input/aristocat-data/submission (11) (1).csv')

possible_public_ = int(len(aristocats)*0.80)

aristocats.iloc[possible_public_:,1] = 0

from datetime import datetime

aristocats.to_csv(

    'aristocats_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.csv', 

    index=False)