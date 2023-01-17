from fastai.text import *
untar_data(URLs.WT103_FWD , data=False, dest=".")
!ls
untar_data(URLs.WT103_BWD , data=False, dest=".")