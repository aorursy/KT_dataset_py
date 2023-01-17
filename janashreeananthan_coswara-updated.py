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
!cat /kaggle/input/coswara/20200814/*.tar.gz.* > 20200814.tar.gz

!cat /kaggle/input/coswara/20200820/*.tar.gz.* > 20200820.tar.gz

!cat /kaggle/input/coswara/20200824/*.tar.gz.* > 20200824.tar.gz

!tar xvzf 20200814.tar.gz

!tar xvzf 20200820.tar.gz

!tar xvzf 20200824.tar.gz
import os

folders = ['20200814','20200820','20200824']

for dirname in folders:

    num = 1

    for subdirname in os.listdir(dirname):

        #print(dirname+"/"+"sample_"+ str(num))

        os.rename(dirname+"/"+subdirname,dirname+"/"+"sample_"+ str(num))

        num = num + 1

    print("No of samples in {} is {}".format(subdirname,num))
!rm -rf 20200814.tar.gz

!rm -rf 20200820.tar.gz

!rm -rf 20200824.tar.gz
folders = ['20200814','20200820','20200824']

for dirname in folders:

    for subdirname in os.listdir(dirname):

        for file in os.listdir("{}/{}".format(dirname,subdirname)):

            if ".wav" in file:

                if "cough" not in file:

                    os.remove("{}/{}/{}".format(dirname,subdirname,file))