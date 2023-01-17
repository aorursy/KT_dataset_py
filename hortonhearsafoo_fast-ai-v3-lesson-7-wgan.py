%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision import *
from fastai.vision.gan import *
path = Path('/kaggle/input/sample/data0/lsun/bedroom')
def get_data(bs, size):
    return (GANItemList.from_folder(path, noise_sz=100)
               .no_split()
               .label_from_func(noop)
               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)
               .databunch(bs=bs, num_workers=0)
               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))
data = get_data(128, 64)
data.show_batch(rows=5)
generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=1)
critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=1)
learn = GANLearner.wgan(data, generator, critic, switch_eval=False,
                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0., model_dir='/tmp/models')
learn.fit(5,2e-4)
# had to restrict to 5 epochs because of Kaggle Kernel limits, uncomment below for full training
# learn.fit(30,2e-4)
learn.gan_trainer.switch(gen_mode=True)
learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(8,8))
