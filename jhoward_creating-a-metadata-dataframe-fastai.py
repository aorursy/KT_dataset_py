!pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null

!pip install git+https://github.com/fastai/fastai_dev             > /dev/null
from fastai2.basics import *

from fastai2.medical.imaging import *
path = Path('../input/rsna-intracranial-hemorrhage-detection/')
path_trn = path/'stage_1_train_images'

fns_trn = path_trn.ls()

fns_trn[:5].attrgot('name')
path_tst = path/'stage_1_test_images'

fns_tst = path_tst.ls()

len(fns_trn),len(fns_tst)
fn = fns_trn[0]

dcm = fn.dcmread()

dcm
def save_lbls():

    path_lbls = path/'stage_1_train.csv'

    lbls = pd.read_csv(path_lbls)

    lbls[["ID","htype"]] = lbls.ID.str.rsplit("_", n=1, expand=True)

    lbls.drop_duplicates(['ID','htype'], inplace=True)

    pvt = lbls.pivot('ID', 'htype', 'Label')

    pvt.reset_index(inplace=True)    

    pvt.to_feather('labels.fth')
save_lbls()
df_lbls = pd.read_feather('labels.fth').set_index('ID')

df_lbls.head(8)
df_lbls.mean()
del(df_lbls)

import gc; gc.collect();
%time df_tst = pd.DataFrame.from_dicoms(fns_tst, px_summ=True)

df_tst.to_feather('df_tst.fth')

df_tst.head()
del(df_tst)

gc.collect();
%time df_trn = pd.DataFrame.from_dicoms(fns_trn, px_summ=True)

df_trn.to_feather('df_trn.fth')