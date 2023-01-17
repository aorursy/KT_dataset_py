import sys
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn
prefix = '98'
path = Path('.')
path_hr = path/'hr' # low res images
path_lr = path/'lr' # high res images


path_test = Path('./test') # images for prediction
path_tmp = Path('./tmp')
export_path = Path('/export')
img_f = get_image_files(path_lr);
for i in range(20):
  img = open_image(img_f[i])
  print(img.shape)
img
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0)
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(tfms,size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
bs,size=32,128
arch = models.resnet34
data = get_data(bs,size)
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9,9))
t = data.valid_ds[0][1].data
t = torch.stack([t,t])
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)
gram_matrix(t)
base_loss = F.l1_loss
vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();
learn.lr_find()
learn.recorder.plot()
lr = 1e-3
def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    save_name = f'{prefix}_{save_name}'
    learn.fit_one_cycle(1, lrs, pct_start=pct_start)
    learn.save(export_path/save_name)
    learn.show_results(rows=4)
do_fit('1a', slice(lr*10))
learn.show_results(rows=5, imgsize=5)
learn.unfreeze()
do_fit('1b', slice(1e-5,lr))
data = get_data(12,size*2)
learn.data = data
learn.freeze()
gc.collect()
learn.show_results()
learn.load(export_path/f'{prefix}_1b');
do_fit('2a')
learn.unfreeze()
do_fit('2b', slice(1e-6,1e-4), pct_start=0.3)
learn.show_results(rows=15)
learn = None
gc.collect();
size = 500
arch = models.resnet34
#Good transfroms for orto
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0)
test_data = (ImageImageList.from_folder(path_test).split_none().label_from_func(lambda x: x)
      .transform(tfms, size=size, tfm_y=True)
      .databunch(bs=1, no_check=True).normalize(imagenet_stats, do_y=True))
learn = unet_learner(test_data, arch, loss_func=F.l1_loss, blur=True, norm_type=NormType.Weight)
learn = learn.load(export_path/f'{prefix}_2b');
img_path = test_data.train_ds.x.items[0]
img = open_image(img_path); img.shape
show_image(img, figsize=(9,9)); img.shape
preds = learn.predict(img)
pred_img = preds[0]
pred_img; pred_img.shape
def export_images(fnames, size):
  import shutil
  import zipfile
  from datetime import datetime
  now = datetime.now()
  timestamp = now.strftime('%Y%m%d_%H%M%z')

  with zipfile.ZipFile(export_path/f'predicted_{timestamp}.zip', 'w') as f:
    for img_f in fnames:
      img = open_image(img_f)
      pred = learn.predict(img)
      export_img = pred[0].resize(size)
      filename = f'{img_f.stem}.png'
      print(f'exporting {filename}...')
      export_img.save(path_tmp/filename)
      shutil.copy(path_test/f'{img_f.stem}.wld', path_tmp / f'{img_f.stem}.wld')
      f.write(path_tmp/filename)
      f.write(path_tmp/f'{img_f.stem}.wld')
test_img_f = get_image_files(path_test)
export_images(test_img_f, 500)