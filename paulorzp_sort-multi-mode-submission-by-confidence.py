import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!ls ../input
multisub = pd.read_csv('../input/lyft-prediction-with-multi-mode-confidence/submission.csv', index_col=[0,1])
cols = list(multisub.columns)



conf = ['','','']

cn = cols[0:3]

conf[0] = cols[3:103]

conf[1] = cols[103:203]

conf[2] = cols[203:303]



def sort_by_conf(x):

    o = x[cn].argsort()[-3:][::-1]

    x[cols] = np.array(list(x[o])+

                       list(x[conf[o[0]]])+

                       list(x[conf[o[1]]])+

                       list(x[conf[o[2]]]))

    return x
multisub
multisub.apply(sort_by_conf, axis=1)
multisub.to_csv('submission.csv', float_format='%.5g')