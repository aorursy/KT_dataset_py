from fastai.basics import *
from fastai.vision.all import *
path = Path('/kaggle/input/test77886699/Data/')
imgs_path = path/'images'
lbls_path = path/'labels'


print(f'Checking number of files - images:{len([f for f in imgs_path.iterdir()])}\
      masks:{len([f for f in lbls_path.iterdir()])}')


# Checking file shapes 
idx = 22
img_path = [f for f in imgs_path.iterdir()][idx]
msk_path = [f for f in lbls_path.iterdir()][idx]

img = np.load(str(img_path))
msk = np.load(str(msk_path))

print(f'Checking shapes - image: {img.shape} mask: {msk.shape}')
# Plotting a sample
_, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img.transpose((1, 2, 0))[..., [3, 2, 1]]*3.0)
ax[1].imshow(msk)

def open_npy(fn, chnls=None, cls=torch.Tensor):
    im = torch.from_numpy(np.load(str(fn))).type(torch.float32)
    if chnls is not None: im = im[chnls]
    return cls(im)

class MSTensorImage(TensorImage):
    
    def __init__(self, x, chnls_first=False):
        self.chnls_first = chnls_first

    @classmethod
    def create(cls, data:(Path,str,ndarray), chnls=None, chnls_first=True):
        
        if isinstance(data, Path) or isinstance(data, str):
            if str(data).endswith('npy'): im = open_npy(fn=data, chnls=chnls, cls=torch.Tensor)

        elif isinstance(data, ndarray): 
            im = torch.from_numpy(data)
        else:
            im = data
        
        return cls(im, chnls_first=chnls_first)

    
    def show(self, chnls=[3, 2, 1], bright=1., ctx=None):
        
        if self.ndim > 2:
            visu_img = self[..., chnls] if not self.chnls_first else self.permute([1, 2, 0])[..., chnls]
        else:
            visu_img = self
        
        visu_img = visu_img.squeeze()
        
        visu_img *= bright
        visu_img = np.where(visu_img > 1, 1, visu_img)
        visu_img = np.where(visu_img < 0, 0, visu_img)
        
        plt.imshow(visu_img) if ctx is None else ctx.imshow(visu_img)
        
        return ctx
    
    def __repr__(self):
        
        return (f'MSTensorImage: {self.shape}')
img = MSTensorImage.create(img_path)
print(img)

_, ax = plt.subplots(1, 3, figsize=(12, 4))
img.show(bright=3., ctx=ax[0])
img.show(chnls=[2, 7, 10], ctx=ax[1])
img.show(chnls=[11], ctx=ax[2])
mask = TensorMask(open_npy(msk_path))
print(mask.shape)

_, ax = plt.subplots(1, 2, figsize=(10, 5))
img.show(bright=3., ctx=ax[0])
mask.show(ctx=ax[1])
imgs_path
get_files(imgs_path,extensions='.npy')
get_lbl_fn(Path('/kaggle/input/test77886699/Data/images/Oros_1_29.npy'))

def get_lbl_fn(img_fn: Path):
    lbl_path = img_fn.parent.parent/'labels'
    lbl_name = img_fn.name
    return (lbl_path/lbl_name)

db = DataBlock(blocks=(TransformBlock(type_tfms=partial(MSTensorImage.create, chnls_first=True),
                                       item_tfms=Resize(100)),
                       TransformBlock(type_tfms=[get_lbl_fn, partial(open_npy, cls=TensorMask)], 
                                      item_tfms=[Resize(100), AddMaskCodes(codes=['clear', 'water', 'shadow'])]),
                      ),
               get_items=partial(get_files, extensions='.npy'),
               splitter=RandomSplitter(valid_pct=0.1)
              )

db.summary(source=imgs_path)

