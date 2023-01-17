!pip install git+https://github.com/fastai/fastai_dev 
from fastai2.torch_basics import *

from fastai2.test import *

from fastai2.data.all import *

from fastai2.vision.core import *

from fastai2.notebook.showdoc import *
source = Path('../input/the-oxfordiiit-pet-dataset/images')
items = get_image_files(source)

split_idx = RandomSplitter()(items)
def resized_image(fn:Path, sz=128):

    x = PILImage.create(fn).convert('RGB').resize((sz,sz))

    return tensor(array(x)).permute(2,0,1).float()/255.
class TitledImage(Tuple):

    def show(self, ctx=None, **kwargs): show_titled_image(self, ctx=ctx, **kwargs)
img = resized_image(items[0])
TitledImage(img,'test').show()
class PetTfm(Transform):

    def __init__(self, items, train_idx):

        self.items,self.train_idx = items,train_idx

        self.labeller = RegexLabeller(pat = r'/([^/]+)_\d+.jpg$')

        vals = map(self.labeller, items[train_idx])

        self.vocab,self.o2i = uniqueify(vals, sort=True, bidir=True)



    def encodes(self, i):

        o = self.items[i]

        return resized_image(o), self.o2i[self.labeller(o)]

    

    def decodes(self, x): return TitledImage(x[0],self.vocab[x[1]])
pets = PetTfm(items, split_idx[0])
x,y = pets(0)

x.shape,y
dec = pets.decode((x,y))

dec.show()
class SiameseImage(Tuple):

    def show(self, ctx=None, **kwargs): 

        img1,img2,same_breed = self

        return show_image(torch.cat([img1,img2], dim=2), title=same_breed, ctx=ctx)
SiameseImage(img,img,True).show();
class SiamesePair(Transform):

    def __init__(self,items,labels):

        self.items,self.labels,self.assoc = items,labels,self

        sortlbl = sorted(enumerate(labels), key=itemgetter(1))

        # dict of (each unique label) -- (list of indices with that label)

        self.clsmap = {k:L(v).itemgot(0) for k,v in itertools.groupby(sortlbl, key=itemgetter(1))}

        self.idxs = range_of(self.items)

        

    def encodes(self,i):

        "x: tuple of `i`th image and a random image from same or different class; y: True if same class"

        othercls = self.clsmap[self.labels[i]] if random.random()>0.5 else self.idxs

        otherit = random.choice(othercls)

        return SiameseImage(self.items[i], self.items[otherit], self.labels[otherit]==self.labels[i])
OpenAndResize = TupleTransform(resized_image)

labeller = RegexLabeller(pat = r'/([^/]+)_\d+.jpg$')

sp = SiamesePair(items, items.map(labeller))

pipe = Pipeline([sp, OpenAndResize], as_item=True)

x,y,z = t = pipe(0)

x.shape,y.shape,z
for _ in range(3): pipe.show(pipe(0))
class ImageResizer(Transform):

    order=10

    "Resize image to `size` using `resample"

    def __init__(self, size, resample=Image.BILINEAR):

        if not is_listy(size): size=(size,size)

        self.size,self.resample = (size[1],size[0]),resample



    def encodes(self, o:PILImage): return o.resize(size=self.size, resample=self.resample)

    def encodes(self, o:PILMask):  return o.resize(size=self.size, resample=Image.NEAREST)
tfms = [[PILImage.create, ImageResizer(128), ToTensor(), ByteToFloatTensor()],

        [labeller, Categorize()]]

dsrc = DataSource(items, tfms)
t = dsrc[0]

type(t[0]),type(t[1])
x,y = dsrc.decode(t)

x.shape,y
dsrc.show(t);
tfms = [[PILImage.create], [labeller, Categorize()]]

dsrc = DataSource(items, tfms)

tdl = TfmdDL(dsrc, bs=1, after_item=[ImageResizer(128), ToTensor(), ByteToFloatTensor()])
t = tdl.one_batch()

x,y = tdl.decode_batch(t)[0]

x.shape,y
dsrc.show((x,y));
pets = DataSource(items, tfms, splits=split_idx)
x,y = pets.subset(1)[0]

x.shape,y
x2,y2 = pets.valid[0]

test_eq(x.shape,x2.shape)

test_eq(y,y2)
xy = pets.valid.decode((x,y))

xy[1]
xy2 = decode_at(pets.valid, 0)

test_eq(type(xy2[1]), Category)

test_eq(xy2, xy)
pets.show((x,y));
ds_img_tfms = [ImageResizer(128), ToTensor()]

dl_tfms = [Cuda(), ByteToFloatTensor()]



trn_dl = TfmdDL(pets.train, bs=9, after_item=ds_img_tfms, after_batch=dl_tfms)

b = trn_dl.one_batch()



test_eq(len(b[0]), 9)

test_eq(b[0][0].shape, (3,128,128))

test_eq(b[0].type(), 'torch.cuda.FloatTensor' if default_device().type=='cuda' else 'torch.FloatTensor')
bd = trn_dl.decode_batch(b)



test_eq(len(bd), 9)

test_eq(bd[0][0].shape, (3,128,128))
_,axs = plt.subplots(3,3, figsize=(9,9))

trn_dl.show_batch(ctxs=axs.flatten())
dbch = pets.databunch(bs=9, after_item=ds_img_tfms, after_batch=dl_tfms)

dbch.train_dl.show_batch()
cv_source = Path('../input/camvid/camvid/CamVid')

cv_items = get_image_files(cv_source/'train')

cv_splitter = RandomSplitter(seed=42)

cv_split = cv_splitter(cv_items)

cv_label = lambda o: cv_source/'train_labels'/f'{o.stem}_L{o.suffix}'
tfms = [[PILImage.create], [cv_label, PILMask.create]]

camvid = DataSource(cv_items, tfms, splits=cv_split)

trn_dl = TfmdDL(camvid.train,  bs=4, after_item=ds_img_tfms, after_batch=dl_tfms)
_,axs = plt.subplots(2,2, figsize=(6,6))

trn_dl.show_batch(ctxs=axs.flatten())