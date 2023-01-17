%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.vision.gan import *
path = Path('../input/pokemon-ab-merged/pokemon-merged')

# path1 = Path("../input/kaggle-one-shot-pokemon/kaggle-one-shot-pokemon/pokemon-a")

# path2 = Path("../input/kaggle-one-shot-pokemon/kaggle-one-shot-pokemon/pokemon-b")
# tfms = get_transforms(flip_vert = False)
def get_data(bs, size):

    return (GANItemList.from_folder(path, noise_sz=100)

               .split_none()

               .label_from_func(noop)

#                .transform(tfms=None, size=size, tfm_y=True)

               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)

               .databunch(bs=bs)

               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))
data = get_data(128, 64)
data.show_batch(rows=5)
generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=1)

critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=1)
learn = GANLearner.wgan(data, generator, critic, switch_eval=False,

                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0., model_dir='/tmp/models')
# Note: After epoch 1000 started training with LR of 2e-2



!wget https://www.dropbox.com/s/xcdzb7e41620qu1/64pokemon_1200epoch.pth

learn.load('/kaggle/working/64pokemon_1200epoch')
# learn.lr_find(stop_div=False, num_it=200)

# learn.recorder.plot(suggestion=True)



# LR finder produced a result of 

# Min numerical gradient: 9.12E-04

# Min loss divided by 10: 4.37E-04



# https://i.imgur.com/rBRCqjW.png

# https://i.imgur.com/8rs6xem.png
# learn.fit(100,9.12E-04)

learn.fit(100,2e-03)
learn.gan_trainer.switch(gen_mode=True)

learn.show_results(ds_type=DatasetType.Train, rows=2, columns=8, figsize=(20,8))
learn.save('/kaggle/working/64pokemon_1300epoch')