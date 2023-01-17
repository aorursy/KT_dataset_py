# 三行魔法代码

%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
bs = 8 # 64 # 批量设置, 过大会超出Kaggle GPU Disk 容量 
# path = untar_data(URLs.PETS)/'images' # 从云端下载数据集，图片全部在一个文件夹中

path = Path('/kaggle/input/'); path.ls()

path_data = path/'the-oxfordiiit-pet-dataset'/'images'/'images'; path_data.ls()[:5]

path_model12 = path/'v3lesson6models'; path_model12.ls()

path_model3 = path/'v3lesson6modelsmore'; path_model3.ls()

path_img = path/'catdogtogether'; path_img.ls()
# 图片变形设计

tfms = get_transforms(max_rotate=20, # 以后逐一尝试

                      max_zoom=1.3, 

                      max_lighting=0.4, 

                      max_warp=0.4,

                      p_affine=1., 

                      p_lighting=1.)
# 将图片夹转化成ImageList

src = ImageList.from_folder(path_data).split_by_rand_pct(0.2, seed=2) # 无需单独做np.random.seed(2)

src

src.train[0:2] # 查看训练集中图片

src.valid[0] # 直接看图

src.train.__class__ # fastai.vision.data.ImageList

src.__class__ # fastai.data_block.ItemLists
# 快捷生成DataBunch

def get_data(size, bs, padding_mode='reflection'): # 提供图片尺寸，批量和 padding模式

    return (src.label_from_re(r'([^/]+)_\d+.jpg$') # 从图片名称中提取label标注

           .transform(tfms, size=size, padding_mode=padding_mode) # 对图片做变形

           .databunch(bs=bs).normalize(imagenet_stats))
data = get_data(224, bs, 'zeros') # 图片统一成224的尺寸

data.train_ds.__class__ # fastai.data_block.LabelList 所以可以像list一样提取数据

data.train_ds[0]

data.train_ds[0][0] # 提取图片，且已经变形，Image class

data.train_ds[0][1] # 提取label， Category class

data.train_ds[0][1].__class__

data.train_ds[0][0].__class__
def _plot(i,j,ax):

    x,y = data.train_ds[3]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,8))
data = get_data(224,bs) # padding mode = reflection 效果更加，无边框黑区
plot_multi(_plot, 3, 3, figsize=(8,8))
gc.collect() # 释放GPU内存，但是数据无从查看？？？

learn = cnn_learner(data, 

                    models.resnet34, 

                    metrics=error_rate, 

                    bn_final=True, # bn_final=True什么意思，出处在哪里？看下面的结果对比

                    model_dir='/kaggle/working') # 确保模型可被写入，且方便下载
learn.summary()
learn.load(path_model12/'3_1e-2_0.8')
learn.load(path_model12/'2_1e-6_1e-3_0.8')
data = get_data(352,bs) # 放大图片尺寸

learn.data = data
data = get_data(352,16)
learn = cnn_learner(data, 

                    models.resnet34, 

                    metrics=error_rate, 

                    bn_final=True,

                    model_dir='/kaggle/working/').load(path_model3/'2_1e-6_1e-4')
idx=150

x,y = data.valid_ds[idx] # 验证集图片保持不变（不论运行多少次）

y

y.data

data.valid_ds.y[idx] # 打印label

data.classes[25] # 说明25是leonberger的序号

x.show()
# 创造一个3x3的matrix作为kernel

k = tensor([

    [0.  ,-5/3,1],

    [-5/3,-5/3,1],

    [1.  ,1   ,1],

]).expand(1,3,3,3)/6 # 然后在转化为一个4D，rank4 tensor，在缩小6倍
k
k.shape # 查看尺寸
t = data.valid_ds[idx][0].data # 从图片中提取数据tensor

t.shape # 展示tensor尺寸
t[None].shape # 将图片tensor转化为一个rank 4 tensor
# F.conv2d??

# 对图片tensor做filter处理

edge = F.conv2d(t[None], k)
show_image(edge[0], figsize=(5,5)) # 展示被kernel处理过的图片的样子
data.c # 可以理解成类别数量
learn.model # 查看模型结构
print(learn.summary()) # 查看layer tensor尺寸和训练参数数量
# learn.model.eval?

m = learn.model.eval(); # 进入 evaluation 模式
xb,_ = data.one_item(x); xb.shape; # 获取一个图片tensor, 应该是变形过后的，

xb # xb tensor长什么样子

# Image(xb) # 是rank 4 tensor, dim 过多，无法作图

# data.denorm? 

data.denorm(xb) # 给予一个新的mean, std转化xb，展示新tensor

data.denorm(xb)[0].shape # 4D 转化为 3D





xb_im = Image(data.denorm(xb)[0]); xb_im # denorm之后就能作图了

xb = xb.cuda(); xb # tensor 后面带上了cuda
from fastai.callbacks.hooks import * # import hooks functions
def hooked_backward(cat=y): # y = leonberger label

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(xb) # xb  = leonberger tensor

            print(preds.shape)

            print(int(cat))

            print(preds[0, int(cat)])

            print(preds)

            preds[0,int(cat)].backward() # 返回 leonberger对应的grad给到hook_g

    return hook_a,hook_g
y

int(y) # 获取类别对应的序号

hook_a,hook_g = hooked_backward()
# hook_a -> <fastai.callbacks.hooks.Hook at 0x7f8b78205278>

# hook_g -> <fastai.callbacks.hooks.Hook at 0x7f8b78205208>

# hook_a.stored.shape # 4D tensor, torch.Size([1, 512, 11, 11])

# hook_a.stored[0].shape # from 4D to 3D 

acts  = hook_a.stored[0].cpu() # 从gpu模式到cpu模式

acts.shape
avg_acts = acts.mean(0) # 压缩512个值，来获取他们的均值

avg_acts.shape
def show_heatmap(hm): # 用kernel来做热力图

    _,ax = plt.subplots(1,3)

    xb_im.show(ax[0]) # 画出原图

    ax[1].imshow(hm, alpha=0.6, extent=(0,352,352,0),

              interpolation='bilinear', cmap='magma');

    xb_im.show(ax[2]) # 两图合并

    ax[2].imshow(hm, alpha=0.6, extent=(0,352,352,0),

              interpolation='bilinear', cmap='magma');
show_heatmap(avg_acts)
# hook_g.stored.__class__ # is a list

# len(hook_g.stored) # just 1

# hook_g.stored[0].__class__ # is a tensor

# hook_g.stored[0].shape # 4D tensor

# hook_g.stored[0][0].shape # 3D tensor

grad = hook_g.stored[0][0].cpu()

# grad.mean(1).shape # 对中间的11取均值

# grad.mean(1).mean(1).shape # 对中间的两个11取均值

grad_chan = grad.mean(1).mean(1)

grad.shape,grad_chan.shape
# grad_chan[...,None,None].shape # 将压缩后的grad从1D变3D

mult = (acts*grad_chan[...,None,None]).mean(0) # activation 与 grad 的相乘，再取一个维度的均值，变成一个kernel

# 最后一层的activation * 最后一层压缩的grad 再求和，并压缩512层取均值

mult.shape
show_heatmap(mult)
# fn = get_image_files(path_img); fn

path_img/'catdogTogether.png'
# x = open_image(fn[0]); x

x = open_image(path_img/'catdogTogether.png'); x
# data.one_item?? # 将一张图作为一整个batch



xb,_ = data.one_item(x)

xb_im = Image(data.denorm(xb)[0]) # 生成图片

xb = xb.cuda()

xb_im
hook_a,hook_g = hooked_backward() # y依旧是序号为25的leonberger
acts = hook_a.stored[0].cpu() # 本图片 最后一层activation 

grad = hook_g.stored[0][0].cpu() # 本图片 最后一层 grad, 并且是基于leonberger类别去提取的grad！！！！！！！！



grad_chan = grad.mean(1).mean(1) # 对 11x11 取均值， 512 长的vector

mult = (acts*grad_chan[...,None,None]).mean(0); mult.shape # 生成11x11 tensor
show_heatmap(mult)
data.classes[0]
hook_a,hook_g = hooked_backward(0)
acts = hook_a.stored[0].cpu()

grad = hook_g.stored[0][0].cpu()



grad_chan = grad.mean(1).mean(1)

mult = (acts*grad_chan[...,None,None]).mean(0)
show_heatmap(mult)