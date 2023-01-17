!pwd
import os
os.listdir("/")
os.listdir('/kaggle')
os.listdir('/kaggle/working')
import pandas as pd
pd.DataFrame({1:[2,3],4:[5,6]}).to_csv('whatever.csv')
os.listdir('/kaggle/working')
pd.read_csv('/kaggle/working/whatever.csv')
os.listdir('/kaggle/config')
os.listdir('/kaggle/lib')
os.listdir('/kaggle/lib/kaggle')
os.listdir('/kaggle/lib/kaggle/competitions')
os.listdir('/kaggle/lib/kaggle/competitions/twosigmanews')
os.listdir('/home')
os.listdir('/mnt')
os.listdir('/src')
os.listdir('/root')
!cat /root/.profile
!cat /root/.bashrc
!ls /root/.keras
!ls /usr
!ls /usr/include
!ls /usr/bin
!ls /usr/sbin
!ls -al /root
!ls -al /root/.keras
