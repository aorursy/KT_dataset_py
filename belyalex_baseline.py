import os
import pandas as pd
import numpy as np
data_dir = "data"
train_dir = data_dir + "/train"
test_dir = data_dir + "/test"
for dirname, _, filenames in os.walk(test_dir):
    df = pd.DataFrame(filenames,columns=['file_name'])
    for filename in filenames:
        print(os.path.join(dirname, filename))
df['file_name']='test/'+df['file_name']        
letters = ['A','B','E','K','M','H','O','P','C','T','Y','X']
df['plates_string']=''
for i, r in df.iterrows():
    r['plates_string']=np.random.choice(letters)+"{0:0=3d}".format(np.random.randint(0,1000))+np.random.choice(letters)+np.random.choice(letters)+"35"
df.head()
df.to_csv('baseline.csv',index=False)
