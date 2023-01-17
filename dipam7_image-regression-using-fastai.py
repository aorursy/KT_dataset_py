%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
path = untar_data(URLs.BIWI_HEAD_POSE)
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6); cal
fname = '09/frame_00667_rgb.jpg'
def img2txt_name(f): return path/f'{str(f)[:-7]}pose.txt'
img = open_image(path/fname)

img.show()
ctr = np.genfromtxt(img2txt_name(fname), skip_header=3); ctr
def convert_biwi(coords):

    c1 = coords[0] * cal[0][0]/coords[2] + cal[0][2]

    c2 = coords[1] * cal[1][1]/coords[2] + cal[1][2]

    return tensor([c2,c1])



def get_ctr(f):

    ctr = np.genfromtxt(img2txt_name(f), skip_header=3)

    return convert_biwi(ctr)



def get_ip(img,pts): return ImagePoints(FlowField(img.size, pts), scale=True)
get_ctr(fname)
ctr = get_ctr(fname)

img.show(y=get_ip(img, ctr), figsize=(6, 6))
data = (PointsItemList.from_folder(path)

        .split_by_valid_func(lambda o: o.parent.name=='13')

        .label_from_func(get_ctr)

        .transform(get_transforms(), tfm_y=True, size=(120,160))

        .databunch().normalize(imagenet_stats)

       )
data.show_batch(3, figsize=(9,6))
learn = cnn_learner(data, models.resnet34)
learn.lr_find()

learn.recorder.plot()
lr = 2e-2
learn.fit_one_cycle(3, slice(lr))
learn.save('/kaggle/working/stage-1')
learn.show_results()
tfms = get_transforms(max_rotate=20, max_zoom=1.5, max_lighting=0.5, max_warp=0.4, p_affine=1., p_lighting=1.)



data = (PointsItemList.from_folder(path)

        .split_by_valid_func(lambda o: o.parent.name=='13')

        .label_from_func(get_ctr)

        .transform(tfms, tfm_y=True, size=(120,160))

        .databunch().normalize(imagenet_stats)

       )
def _plot(i,j,ax):

    x,y = data.train_ds[0]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,6))