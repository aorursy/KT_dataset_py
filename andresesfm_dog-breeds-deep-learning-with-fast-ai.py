%reload_ext autoreload
%autoreload 2
%matplotlib inline
#import sys
#sys.path.append("../fastai") 
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
PATH = "data/"
sz=224
arch=resnext101_64
bs=58
label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv)))-1
val_idx = get_cv_idxs(n)
!pwd
!ls {PATH}
label_df= pd.read_csv(label_csv)
label_df.head()
label_df.pivot_table(index='breed',aggfunc=len).sort_values('id',ascending=False).head()
transformations = tfms_from_model(arch,sz,aug_tfms=transforms_side_on,max_zoom=1.2)
data = ImageClassifierData.from_csv(PATH,'train', f'{PATH}labels.csv',test_name='test',val_idxs=val_idx,suffix='.jpg',tfms=transformations,bs=bs)
fn=PATH+data.trn_ds.fnames[0]; fn
n
val_idx
len(val_idx)
img = PIL.Image.open(fn); img
img.size
size_d = {k:PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}
row_sz,col_sz = list(zip(*size_d.values()))
row_sz = np.array(row_sz);col_sz= np.array(col_sz)
row_sz[:5]
plt.hist(row_sz);
plt.hist(row_sz[row_sz<1000]);
def get_data(sz,bs):
    tfms = tfms_from_model(arch,sz,aug_tfms=transforms_side_on,max_zoom=1.2)
    data = ImageClassifierData.from_csv(PATH,'train',f'{PATH}labels.csv',test_name='test',num_workers=4, 
                                        val_idxs=val_idx,suffix='.jpg',tfms=tfms,bs=bs)
    return data if sz >300 else data.resize(340,'tmp')
data = get_data(sz,bs)
learn = ConvLearner.pretrained(arch,data,precompute=True)
learn.fit(1e-2,5)
from sklearn import metrics
data = get_data(sz,bs)
learn = ConvLearner.pretrained(arch,data,precompute=True,ps=0.5)
learn.fit(1e-2,2)
learn.precompute=False
learn.fit(1e-2, 5, cycle_len=1)
learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2)
learn.save('224_pre')
learn.load('224_pre')
learn.set_data(get_data(299,bs))
learn.freeze()
learn.fit(1e-2, 3, cycle_len=1)
learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2)
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds), axis=0)
accuracy_np(probs,y), metrics.log_loss(y, probs)
learn.save('299_pre')
learn.load('299_pre')
learn.fit(1e-2,1,cycle_len=2)
log_preds,y = learn.TTA(is_test=True)
probs = np.exp(log_preds)
probs.shape
probs[0,:,:].shape
data.classes
df = pd.DataFrame(probs[0,:,:], columns = data.classes)
df.columns
df.insert(0,'id',[o[5:-4] for o in data.test_ds.fnames])
df.columns
df.head()
SUBM = f'{PATH}subm/'
os.makedirs(SUBM,exist_ok=True)
df.to_csv(f'{SUBM}subm.gz', compression='gzip',index=False)
FileLink(f'{SUBM}subm.gz')
