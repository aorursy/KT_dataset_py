%reload_ext autoreload 

%autoreload 2 

%matplotlib inline
from fastai import *

from fastai.vision import *
bs = 64
path = untar_data(URLs.PETS)

path_anno = path/'annotations'

path_img = path/'images'

fnames = get_image_files(path_img)

np.random.seed(2)

pat = re.compile(r'/([^/]+)_\d+.jpg$')
bs=4
# tfms=ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0

#                               )

tfms = None
np.random.seed(42)

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=tfms, size=224,  bs=bs, num_workers=0

                                  ).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet18, metrics=error_rate)

learn.fit_one_cycle(15)
np.random.seed(42)

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224,  bs=bs, num_workers=0

                                  ).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet18, metrics=error_rate)

learn.fit_one_cycle(15)
tfms = get_transforms()

type(tfms)
tfms = get_transforms()

len(tfms)
tfms
#Helper functions from fastai docs

def get_ex(): return open_image(path/'images/beagle_192.jpg')



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]
plots_f(2, 4, 12, 6, size=224)
tfms = get_transforms(max_rotate=180)

plots_f(2, 4, 12, 6, size=224)
def get_ex(): return open_image("../input/satt.jpg")



def plots_f_sate(rows, cols, width, height, **kwargs):

    [get_ex.apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]

    

tfms = get_transforms(max_rotate=180)

plots_f(2, 4, 12, 6, size=224)
def get_ex(): return open_image(path/'images/beagle_192.jpg')



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]
fig, axs = plt.subplots(1,5,figsize=(12,4))

for change, ax in zip(np.linspace(0.1,0.9,5), axs):

    brightness(get_ex(), change).show(ax=ax, title=f'change={change:.1f}')
def get_ex(): return open_image("../input/contrast_example.jpg")



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]

    

fig, axs = plt.subplots(1,5,figsize=(48,24))

for change, ax in zip(np.linspace(0.1,0.9,5), axs):

    brightness(get_ex(), change).show(ax=ax, title=f'change={change:.1f}')

def get_ex(): return open_image(path/'images/beagle_192.jpg')



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]



fig, axs = plt.subplots(1,5,figsize=(12,4))

for scale, ax in zip(np.exp(np.linspace(log(0.5),log(2),5)), axs):

    contrast(get_ex(), scale).show(ax=ax, title=f'scale={scale:.2f}')

def get_ex(): return open_image("../input/contrast_example.jpg")



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]

    

fig, axs = plt.subplots(1,5,figsize=(48,4))

for scale, ax in zip(np.exp(np.linspace(log(0.5),log(2),5)), axs):

    contrast(get_ex(), scale).show(ax=ax, title=f'scale={scale:.2f}')
fig, axs = plt.subplots(1,5,figsize=(12,4))

for center, ax in zip([[0.,0.], [0.,1.],[0.5,0.5],[1.,0.], [1.,1.]], axs):

    crop(get_ex(), 300, *center).show(ax=ax, title=f'center=({center[0]}, {center[1]})')
fig, axs = plt.subplots(1,5,figsize=(12,4))

for size, ax in zip(np.linspace(200,600,5), axs):

    crop_pad(get_ex(), int(size), 'zeros', 0.,0.).show(ax=ax, title=f'size = {int(size)}')
fig, axs = plt.subplots(1,5,figsize=(12,4))

for size, ax in zip(np.linspace(200,600,5), axs):

    crop_pad(get_ex(), int(size), 'reflection', 0.,0.).show(ax=ax, title=f'size = {int(size)}')
def get_ex(): return open_image(path/'images/beagle_192.jpg')



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]



    

fig, axs = plt.subplots(2,4,figsize=(12,8))

for k, ax in enumerate(axs.flatten()):

    dihedral(get_ex(), k).show(ax=ax, title=f'k={k}')

plt.tight_layout()
def get_ex(): return open_image("../input/satt.jpg")



def plots_f_sate(rows, cols, width, height, **kwargs):

    [get_ex.apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]

    

tfms = get_transforms(max_rotate=180)

plots_f(2, 4, 12, 6, size=224)
def get_ex(): return open_image(path/'images/beagle_192.jpg')



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]



fig, axs = plt.subplots(1,5,figsize=(12,4))

for magnitude, ax in zip(np.linspace(-0.05,0.05,5), axs):

    tfm = jitter(magnitude=magnitude)

    get_ex().jitter(magnitude).show(ax=ax, title=f'magnitude={magnitude:.2f}')
def get_ex(): return open_image(path/'images/beagle_192.jpg')



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]



fig, axs = plt.subplots(2,4,figsize=(12,8))

for i, ax in enumerate(axs.flatten()):

    magnitudes = torch.tensor(np.zeros(8))

    magnitudes[i] = 0.5

    perspective_warp(get_ex(), magnitudes).show(ax=ax, title=f'coord {i}')
def get_ex(): return open_image("../input/contrast_example.jpg")



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]



    

fig, axs = plt.subplots(2,4,figsize=(12,8))

for i, ax in enumerate(axs.flatten()):

    magnitudes = torch.tensor(np.zeros(8))

    magnitudes[i] = 0.5

    perspective_warp(get_ex(), magnitudes).show(ax=ax, title=f'coord {i}')
def get_ex(): return open_image("../input/cereal_ex.jpg")



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]

    

tfm = symmetric_warp(magnitude=(-0.2,0.2))

_, axs = plt.subplots(2,4,figsize=(12,6))

for ax in axs.flatten():

    img = get_ex().apply_tfms(tfm, padding_mode='zeros')

    img.show(ax=ax)
def get_ex(): return open_image("../input/cereal_ex.jpg")



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]

    

fig, axs = plt.subplots(2,4,figsize=(12,8))

for i in range(4):

    get_ex().tilt(i, 0.4).show(ax=axs[0,i], title=f'direction={i}, fwd')

    get_ex().tilt(i, -0.4).show(ax=axs[1,i], title=f'direction={i}, bwd')