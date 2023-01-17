import fastai.vision as fv
path = fv.Path('/kaggle/input/images/images'); path.ls()[:5]
src = fv.ImageList.from_folder(path).split_by_rand_pct(0.2, seed=2)

tfms = fv.get_transforms()
def get_data(size, bs, padding_mode='reflection'): # 提供图片尺寸，批量和 padding模式

    return (src.label_from_re(r'([^/]+)_\d+.jpg$') # 从图片名称中提取label标注

           .transform(tfms, size=size, padding_mode=padding_mode) # 对图片做变形

           .databunch(bs=8).normalize(fv.imagenet_stats))
data = get_data(224, 8, 'zeros') # 图片统一成224的尺寸
learn = fv.cnn_learner(data, 

                    fv.models.resnet34, 

                    metrics=fv.error_rate, 

                    bn_final=True, # bn_final=True， 最后一层加入BatchNorm

                    model_dir='/kaggle/working') # 确保模型可被写入，且方便下载
res34 = fv.models.resnet34()
res34.layer1
res34.layer2
res34.layer3
res34.layer4
learn.layer_groups