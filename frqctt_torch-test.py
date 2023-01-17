import fastai

from fastai.vision import *

from fastai.callbacks import *

from fastai.vision.gan import *



from torchvision.models import vgg16_bn
path_img = '../input/mise-16/16/load'

path_lbl = '../input/mise-16/16/mise_pic/'



src = ImageImageList.from_folder(path_img).split_by_rand_pct(0.1, seed=42)



bs,size=512, (64,8)

def get_data(bs,size):

    data = (src.label_from_func(lambda x: path_lbl + f'{x.stem}{x.suffix}')

           .transform(get_transforms(max_zoom=1), size=size, tfm_y=True)

           .databunch(bs=bs))



    data.c = 3

    return data



data_gen = get_data(bs,size)

data_gen.show_batch(3)

def gram_matrix(x):

    n,c,h,w = x.size()

    x = x.view(n, c, -1)

    return (x @ x.transpose(1,2))/(c*h*w)



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

        self.feat_losses = [base_loss(input,target) * 100]

        self.feat_losses += [base_loss(f_in, f_out)*w

                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3

                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))

        return sum(self.feat_losses)

    

    def __del__(self): self.hooks.remove()

        



feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
os.mkdir('/kaggle/output')
wd = 1e-3

arch = models.resnet34



learn = unet_learner(data_gen, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,

                     blur=True, norm_type=NormType.Weight, model_dir="/kaggle/output")
lr = 1e-3

learn.fit_one_cycle(50, lr)
learn.fit_one_cycle(50, lr)
learn.recorder.plot_losses()
learn.save("/kaggle/working/kaggle_1")
os.listdir('/kaggle/working')