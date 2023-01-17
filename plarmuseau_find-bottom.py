import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        if filename.find("rain")>0:

            train=pd.read_csv(os.path.join(dirname, filename) )

        if filename.find("est")>0:

            test=pd.read_csv(os.path.join(dirname, filename) )

            

        if filename.find("ubm")>0:

            subm=pd.read_csv(os.path.join(dirname, filename) )            
subm['citation_influence_label']=1
subm.to_csv('fill_1',index=False)

subm['citation_influence_label']=0

subm.to_csv('fill_0',index=False)