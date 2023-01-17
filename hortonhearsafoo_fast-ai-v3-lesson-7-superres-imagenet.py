import fastai
from fastai.vision import *
from fastai.callbacks import *
from torchvision.models import vgg16_bn
path = Path('data/imagenet')
path_hr = path/'images/train'
path_lr = path/'small-64/train'
path_mr = path/'small-256/train'
il = ImageItemList.from_folder(path_hr)
bs,size=16,256
arch = models.resnet34
# sample = 0.1
sample = False

tfms = get_transforms()
src = ImageImageList.from_folder(path_lr)
if sample: src = src.filter_by_rand(sample, seed=42)
src = src.random_split_by_pct(0.1, seed=42)
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.relative_to(path_lr))
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
data = get_data(bs,size)
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)
vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
base_loss = F.l1_loss

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
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics, blur=True, norm_type=NormType.Weight)
gc.collect();
learn.unfreeze()
learn.load(Path('data/oxford-iiit-pet/small-96/models/2b').absolute());
learn.fit_one_cycle(1, slice(1e-6,1e-4))
learn.save('imagenet')
learn.show_results(rows=3, imgsize=5)
learn.recorder.plot_losses()
data_mr = (ImageImageList.from_folder(path_mr).random_split_by_pct(0.1, seed=42)
          .label_from_func(lambda x: path_hr/x.relative_to(path_lr))
          .transform(get_transforms(), size=(819,1024), tfm_y=True)
          .databunch(bs=2).normalize(imagenet_stats, do_y=True))
learn.data = data_mr
fn = '/data0/datasets/part1v3/oxford-iiit-pet/other/dropout.jpg'
img = open_image(fn); img.shape
_,img_hr,b = learn.predict(img)
show_image(img, figsize=(18,15), interpolation='nearest');
Image(img_hr).show(figsize=(18,15))
