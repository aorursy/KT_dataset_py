!pip install -q fastai2

!pip install -q jovian
from fastai2.vision.all import *

path = untar_data(URLs.PETS)
path.ls()
#hide

Path.BASE_PATH = path
path.ls()
(path/"images").ls()
fname = (path/"images").ls()[5]

fname
fname.name
re.findall(r'(.+)_\d+.jpg$', fname.name)
pets = DataBlock(blocks = (ImageBlock, CategoryBlock), # this is a tuple where we specify what type we want for independent and dependant variables.(independemt-->ImageBlock(images), dependent-->CategoryBlock(here strings))

                 get_items=get_image_files,# specify what func to use to get data(since its image classification here, we are using get_image_file which returns list of images in the path)

                 splitter=RandomSplitter(seed=42), # splits b/w train and validation set (by default 20%)

                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),# specify how to get labels a.k.a category

                 item_tfms=Resize(460),# applies for all images, before batching them. At each epoch

                 batch_tfms=aug_transforms(size=224, min_scale=0.75)) #applies for batches in run time (a.k.a while training)

dls = pets.dataloaders(path/"images") # this is path taken in DataBlock, dataloader is responsible to provide batches, appply trasnforms, have info about train and vaild and provides them to learner when required 
dls.vocab
dblock1 = DataBlock(blocks=(ImageBlock(), CategoryBlock()),

                   get_y=parent_label,

                   item_tfms=Resize(460))

dls1 = dblock1.dataloaders([(path/'images'/'american_pit_bull_terrier_79.jpg')]*100, bs=8)

dls1.train.get_idxs = lambda: Inf.ones

x,y = dls1.valid.one_batch()

_,axs = subplots(1, 2)



x1 = TensorImage(x.clone())

x1 = x1.affine_coord(sz=224)

x1 = x1.rotate(draw=30, p=1.)

x1 = x1.zoom(draw=1.2, p=1.)

x1 = x1.warp(draw_x=-0.2, draw_y=0.2, p=1.)



tfms = setup_aug_tfms([Rotate(draw=30, p=1, size=224), Zoom(draw=1.2, p=1., size=224),

                       Warp(draw_x=-0.2, draw_y=0.2, p=1., size=224)])

x = Pipeline(tfms)(x)

#x.affine_coord(coord_tfm=coord_tfm, sz=size, mode=mode, pad_mode=pad_mode)

TensorImage(x[0]).show(ctx=axs[0])

TensorImage(x1[0]).show(ctx=axs[1]);
dls.show_batch(nrows=1, ncols=3)
pets1 = DataBlock(blocks = (ImageBlock, CategoryBlock),

                 get_items=get_image_files, 

                 splitter=RandomSplitter(seed=42),

                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'))

pets1.summary(path/"images")
learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(2)
learn.loss_func
x,y = dls.one_batch()
y
dls.bs
dls.vocab[20]
preds,_ = learn.get_preds(dl=[(x,y)])

preds[0]
len(preds[0]),preds[0].sum()
preds[0].argmax()
preds[0].max()
show_image(x[0])
def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):

    x = torch.linspace(min,max)

    fig,ax = plt.subplots(figsize=figsize)

    ax.plot(x,f(x))

    if tx is not None: ax.set_xlabel(tx)

    if ty is not None: ax.set_ylabel(ty)

    if title is not None: ax.set_title(title)

plot_function(torch.sigmoid, min=-4,max=4)
#hide

torch.random.manual_seed(42);
acts = torch.randn((6,2))*2

acts
acts.sigmoid()
(acts[:,0]-acts[:,1]).sigmoid()
sm_acts = torch.softmax(acts, dim=1)

sm_acts
targ = tensor([0,1,0,1,1,0])
sm_acts
idx = range(6)

sm_acts[idx, targ]
from IPython.display import HTML

df = pd.DataFrame(sm_acts, columns=["3","7"])

df['targ'] = targ

df['idx'] = idx

df['loss'] = sm_acts[range(6), targ]

t = df.style.hide_index()

#To have html code compatible with our script

html = t._repr_html_().split('</style>')[1]

html = re.sub(r'<table id="([^"]+)"\s*>', r'<table >', html)

display(HTML(html))
-sm_acts[idx, targ]
F.nll_loss(sm_acts, targ, reduction='none')
plot_function(torch.log, min=0,max=4)
torch.log(tensor(0.)), torch.log(tensor(.5)), torch.log(tensor(1.)) 
loss_func = nn.CrossEntropyLoss()
loss_func(acts, targ)
F.cross_entropy(acts, targ)
nn.CrossEntropyLoss(reduction='none')(acts, targ)
import jovian
jovian.commit()