# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import json #Used to Open Json File
data = json.load(open('../input/train.json'))
#Saved On Data Frame
data_url = pd.DataFrame.from_dict(data['images'])
del data
import urllib3 #Url Opener
http = urllib3.PoolManager(50)
failed = []
#Function To open Image if Url exsist and then saving to File
def download_img(id_,url_):
    try:
        req = http.request('GET',url_)
        f = open("train/"+str(id_)+".jpg",'wb')
        f.write(req.data)
        f.close()
    except:
        failed.append(str(id_))
from multiprocessing.pool import ThreadPool as Pool # from multiprocessing import Pool
pool_size = 500  # your "parallelness"
pool = Pool(pool_size)
#Main Function to Run for Downloading and saving Images
for index,x in data_url.iterrows():
    pool.apply_async(download_img, (x['image_id'],x['url'][0],))
    print(x['image_id'])
    
pool.close()
pool.join()
#Saving Failed Images
f = open("failed_list",'wb')
f.write(failed)
f.close
