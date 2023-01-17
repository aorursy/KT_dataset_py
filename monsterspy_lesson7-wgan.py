%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.vision.gan import *
path = Path('../input/lsun_bedroom/sample/data0/lsun/bedroom')
def get_data(bs, size):

    return (GANItemList.from_folder(path, noise_sz=100)

               .split_none()

               .label_from_func(noop)

               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)

               .databunch(bs=bs)

               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))
data = get_data(128, 64)
data.show_batch(rows=5)
generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=1)

critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=1)
learn = GANLearner.wgan(data, generator, critic, switch_eval=False,

                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)
learn.fit(3, 2e-4)
learn.gan_trainer.switch(gen_mode=True)

learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(8,8))