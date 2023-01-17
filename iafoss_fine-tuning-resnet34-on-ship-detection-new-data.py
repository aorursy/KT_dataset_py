from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
PATH = './'
TRAIN = '../input/airbus-ship-detection/train_v2/'
TEST = '../input/airbus-ship-detection/test_v2/'
SEGMENTATION = '../input/airbus-ship-detection/train_ship_segmentations_v2.csv'
LOAD_PATH = '../input/fine-tuning-resnet34-on-ship-detection/models/'
nw = 2   #number of workers for data loader
arch = resnet34 #specify target architecture
train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
#5% of data in the validation set is sufficient for model evaluation
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)
class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        if(self.path == TEST): return 0
        masks = self.segmentation_df.loc[self.fnames[i]]['EncodedPixels']
        if(type(masks) == float): return 0 #NAN - no ship 
        else: return 1
    
    def get_c(self): return 2 #number of classes
def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    return md
sz = 256 #image size
bs = 64  #batch size

md = get_data(sz,bs)
learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learn.opt_fn = optim.Adam
#learn.lr_find()
#learn.sched.plot()
#learn.fit(2e-3, 1)
#learn.unfreeze()
#lr=np.array([1e-4,5e-4,2e-3])
#learn.fit(lr, 1, cycle_len=2, use_clr=(20,8))
#learn.sched.plot_lr()
#learn.save('Resnet34_lable_256_1')
path = learn.models_path
learn.models_path = LOAD_PATH
learn.load('Resnet34_lable_256_1')
learn.models_path = path
log_preds,y = learn.predict_with_targs(is_test=True)
probs = np.exp(log_preds)[:,1]
pred = (probs > 0.5).astype(int)
df = pd.DataFrame({'id':test_names, 'p_ship':probs})
df.to_csv('ship_detection.csv', header=True, index=False)
#sz = 384 #image size
#bs = 32  #batch size

#md = get_data(sz,bs)
#learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
#learn.opt_fn = optim.Adam
#learn.unfreeze()
#lr=np.array([1e-4,5e-4,2e-3])
#learn.load('Resnet34_lable_256_1')
#learn.fit(lr/2, 1, cycle_len=2, use_clr=(20,8)) #lr is smaller since bs is only 32
#learn.save('Resnet34_lable_384_1')