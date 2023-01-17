%reload_ext autoreload

%autoreload 2

%matplotlib inline
import pathlib

import fastai

from fastai import *

from fastai.vision import *

from fastai.callbacks import *

from fastai.utils.mem import *



from torchvision.models import vgg16_bn

from subprocess import check_output
input_path = Path('/kaggle/input/denoising-dirty-documents')

items = list(input_path.glob("*.zip"))

print([x for x in items])
import zipfile



for item in items:

    print(item)

    with zipfile.ZipFile(str(item), "r") as z:

        z.extractall(".")
bs, size = 4, 128

arch = models.resnet34

path_train = Path("train")

path_train_cleaned = Path("train_cleaned")

path_test = Path("test")

path_submission = Path("submission")
src = ImageImageList.from_folder(path_train).split_by_rand_pct(0.2, seed=42)
def get_data(src, bs, size):

    data = (

        src.label_from_func(lambda x: path_train_cleaned / x.name)

           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)

           .databunch(bs=bs)       

           .normalize(imagenet_stats, do_y=True)

    )

    data.c = 3

    return data
data = get_data(src, bs, size)
# Show some validation examples

data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(5, 5), title="Some image")
t = data.valid_ds[0][1].data

t = torch.stack([t,t])
def gram_matrix(x):

    n,c,h,w = x.size()

    x = x.view(n, c, -1)

    return (x @ x.transpose(1,2))/(c*h*w)
base_loss = F.l1_loss
vgg_m = vgg16_bn(True).features.cuda().eval()

requires_grad(vgg_m, False)
# Show the layers before all pooling layers, which turn out to be ReLU activations.

# This is right before the grid size changes in the VGG model, which we are using

# for feature generation.

blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]

blocks, [vgg_m[i] for i in blocks]
class FeatureLoss(nn.Module):

    def __init__(self, m_feat, layer_ids, layer_wgts):

        """ m_feat is the pretrained model """

        super().__init__()

        self.m_feat = m_feat

        self.loss_features = [self.m_feat[i] for i in layer_ids]

        # hooking grabs intermediate layers

        self.hooks = hook_outputs(self.loss_features, detach=False)

        self.wgts = layer_wgts

        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))

              ] + [f'gram_{i}' for i in range(len(layer_ids))]



    def make_features(self, x, clone=False):

        self.m_feat(x)

        return [(o.clone() if clone else o) for o in self.hooks.stored]

    

    def forward(self, input, target):

        # get features for target

        out_feat = self.make_features(target, clone=True)

        # features for input

        in_feat = self.make_features(input)

        # calc l1 pixel loss

        self.feat_losses = [base_loss(input,target)]

        # get l1 loss from all the block activations

        self.feat_losses += [base_loss(f_in, f_out)*w

                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3

                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        # so we can show all the layer loss amounts

        self.metrics = dict(zip(self.metric_names, self.feat_losses))

        return sum(self.feat_losses)

    

    def __del__(self): self.hooks.remove()
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
wd = 1e-3

learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,

                     blur=True, norm_type=NormType.Weight)

gc.collect();
learn.model_dir = Path('models').absolute()
learn.lr_find()

learn.recorder.plot()
print(f"Validation set size: {len(data.valid_ds.items)}")
lr = 1e-3
def do_fit(save_name, lrs=slice(lr), pct_start=0.9):

    learn.fit_one_cycle(10, lrs, pct_start=pct_start)

    learn.save(save_name)

    learn.show_results(rows=1, imgsize=5)
do_fit('1a', slice(lr*10))
learn.unfreeze()
do_fit('1b', slice(1e-5, lr))
# Increase resolution of the images.

data = get_data(src, 12, size*2)
learn.data = data

learn.freeze()

gc.collect()
learn.load('1b');
do_fit('2a')
learn.unfreeze()
do_fit('2b', slice(1e-6,1e-4), pct_start=0.3)



# save entire configuration

#learn.export(file = model_path)
fn = data.valid_ds.x.items[10]; fn
img = open_image(fn); img.shape
p,img_pred,b = learn.predict(img)

show_image(img, figsize=(8,5), interpolation='nearest');
Image(img_pred).show(figsize=(8,5))
# Turn off resizing transformations for inference time.

# https://forums.fast.ai/t/segmentation-mask-prediction-on-different-input-image-sizes/44389

learn.data.single_ds.tfmargs['size'] = None
test_images = ImageImageList.from_folder(path_test)

print(test_images)
img = test_images[0]

img.show()

img.shape
p, img_pred, b = learn.predict(img)
def rgb2gray(_img):

    """ Convert from 3 channels to 1 channel """

    from skimage.color import rgb2gray as _rgb2gray



    # Rotate channels dimension to the end, per skimage's expectations

    _img_pred_np = _img.permute(1, 2, 0).numpy()

    _img_pred_2d = Tensor(_rgb2gray(_img_pred_np))

    # Add the channel dimension back

    _img_pred = _img_pred_2d.unsqueeze(0)

    return _img_pred

  

Image(rgb2gray(img_pred)).show(figsize=(8,5))
def write_image(fname, _img_tensor):

    _img_tensor = (_img_tensor * 255).to(dtype=torch.uint8)

    imwrite(path_submission/fname, _img_tensor.squeeze().numpy())
import csv

from imageio import imread, imwrite



path_submission.mkdir(exist_ok=True)



with Path('submission.csv').open('w', encoding='utf-8', newline='') as outf:

    writer = csv.writer(outf)

    writer.writerow(('id', 'value'))

    for i, fname in enumerate(path_test.glob("*.png")):

        img = open_image(fname)

        img_id = int(fname.name[:-4])

        print('Processing: {} '.format(img_id))

        # Predictions

        p, img_pred, b = learn.predict(img)

        # Convert to grayscale and clip out of range values.

        img_2d = rgb2gray(img_pred).clamp(0, 1)

        # Write an image file for examination.

        write_image(fname.name, img_2d)

        # Write to the submission file, in a very inefficient way.

        for r in range(img_2d.shape[1]):

            for c in range(img_2d.shape[2]):

                id = str(img_id)+'_'+str(r + 1)+'_'+str(c + 1)

                val = img_2d[0, r, c].item()

                writer.writerow((id, val))