%matplotlib inline

%reload_ext autoreload

%autoreload 2

# 多行输出

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all" 
# !pip install pretrainedmodels
from fastai.vision import *

# import pretrainedmodels
torch.cuda.is_available()
# 设置数据路径

path = Path('../input/')
d_path = path/'iwildcam-2019-fgvc6'

m_path = path/'pytorch-model-zoo'
train_df = pd.read_csv(d_path/'train.csv')

train_df = pd.concat([train_df['id'],train_df['category_id']],axis=1,keys=['id','category_id'])

train_df.head()
test_df = pd.read_csv(d_path/'test.csv')

test_df = pd.DataFrame(test_df['id'])

test_df['predicted'] = 0

test_df.head()
# 图片变换

tfms = get_transforms(do_flip=True, max_rotate=20, max_zoom=1.3, max_lighting=0.4,

                      max_warp=0.4, p_affine=1., p_lighting=1.)
test_set = ImageList.from_df(test_df, path=d_path, cols='id', folder='test_images', suffix='.jpg')
# 构建数据集

np.random.seed(42)

# 使用 ImageList 是因为图像是多标签的

src = (ImageList.from_df(train_df, path=d_path, folder='train_images', cols='id', suffix='.jpg')

       .split_by_rand_pct(0.1)

       .label_from_df(cols='category_id')

       .add_test(test_set)

      )
# img_size=128

img_size=224

bs=32
data = (src.transform(tfms, size=img_size)

        .databunch(path='.', bs=bs, device= torch.device('cuda:0')).normalize(imagenet_stats))
# 查看部分数据

data.show_batch(rows=3, figsize=(12,9))
f1 = partial(fbeta, beta=1)
from collections import OrderedDict

import math



import torch.nn as nn

from torch.utils import model_zoo



pretrained_settings = {

    'senet154': {

        'url': m_path/'senet154-c7b49a05.pth'

    },

    'se_resnet152': {

        'url': m_path/'se_resnet152-d17c99b7.pth'

    },

    'se_resnext101_32x4d': {

        'url': m_path/'se_resnext101_32x4d-3b2fe3d8.pth'

    }

}





class SEModule(nn.Module):



    def __init__(self, channels, reduction):

        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,

                             padding=0)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,

                             padding=0)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        module_input = x

        x = self.avg_pool(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return module_input * x





class Bottleneck(nn.Module):

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)

        if self.downsample is not None:

            residual = self.downsample(x)



        out = self.se_module(out) + residual

        out = self.relu(out)

        return out





class SEBottleneck(Bottleneck):

    expansion = 4



    def __init__(self, inplanes, planes, groups, reduction, stride=1,

                 downsample=None):

        super(SEBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,

                               stride=stride, padding=1, groups=groups,

                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes * 4)

        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,

                               bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.se_module = SEModule(planes * 4, reduction=reduction)

        self.downsample = downsample

        self.stride = stride





class SEResNetBottleneck(Bottleneck):

    expansion = 4



    def __init__(self, inplanes, planes, groups, reduction, stride=1,

                 downsample=None):

        super(SEResNetBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,

                               stride=stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,

                               groups=groups, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.se_module = SEModule(planes * 4, reduction=reduction)

        self.downsample = downsample

        self.stride = stride





class SEResNeXtBottleneck(Bottleneck):

    expansion = 4



    def __init__(self, inplanes, planes, groups, reduction, stride=1,

                 downsample=None, base_width=4):

        super(SEResNeXtBottleneck, self).__init__()

        width = math.floor(planes * (base_width / 64)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,

                               stride=1)

        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,

                               padding=1, groups=groups, bias=False)

        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.se_module = SEModule(planes * 4, reduction=reduction)

        self.downsample = downsample

        self.stride = stride





class SENet(nn.Module):



    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,

                 inplanes=128, input_3x3=True, downsample_kernel_size=3,

                 downsample_padding=1, num_classes=1000):

        super(SENet, self).__init__()

        self.inplanes = inplanes

        if input_3x3:

            layer0_modules = [

                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,

                                    bias=False)),

                ('bn1', nn.BatchNorm2d(64)),

                ('relu1', nn.ReLU(inplace=True)),

                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,

                                    bias=False)),

                ('bn2', nn.BatchNorm2d(64)),

                ('relu2', nn.ReLU(inplace=True)),

                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,

                                    bias=False)),

                ('bn3', nn.BatchNorm2d(inplanes)),

                ('relu3', nn.ReLU(inplace=True)),

            ]

        else:

            layer0_modules = [

                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,

                                    padding=3, bias=False)),

                ('bn1', nn.BatchNorm2d(inplanes)),

                ('relu1', nn.ReLU(inplace=True)),

            ]

        # To preserve compatibility with Caffe weights `ceil_mode=True`

        # is used instead of `padding=1`.

        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,

                                                    ceil_mode=True)))

        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.layer1 = self._make_layer(

            block,

            planes=64,

            blocks=layers[0],

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=1,

            downsample_padding=0

        )

        self.layer2 = self._make_layer(

            block,

            planes=128,

            blocks=layers[1],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.layer3 = self._make_layer(

            block,

            planes=256,

            blocks=layers[2],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.layer4 = self._make_layer(

            block,

            planes=512,

            blocks=layers[3],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.avg_pool = nn.AvgPool2d(7, stride=1)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

        self.last_linear = nn.Linear(512 * block.expansion, num_classes)



    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,

                    downsample_kernel_size=1, downsample_padding=0):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, planes * block.expansion,

                          kernel_size=downsample_kernel_size, stride=stride,

                          padding=downsample_padding, bias=False),

                nn.BatchNorm2d(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, groups, reduction, stride,

                            downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes, groups, reduction))



        return nn.Sequential(*layers)



    def features(self, x):

        x = self.layer0(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x



    def logits(self, x):

        x = self.avg_pool(x)

        if self.dropout is not None:

            x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.last_linear(x)

        return x



    def forward(self, x):

        x = self.features(x)

        x = self.logits(x)

        return x



def senet154(pretrained=False):

    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,

                  dropout_p=0.2, num_classes=1000)

    if pretrained:

        model.load_state_dict(torch.load(pretrained_settings['senet154']['url']))

    return model





def se_resnet152(pretrained=False):

    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,

                  dropout_p=None, inplanes=64, input_3x3=False,

                  downsample_kernel_size=1, downsample_padding=0,

                  num_classes=1000)

    if pretrained:

        model.load_state_dict(torch.load(pretrained_settings['se_resnet152']['url']))

    return model





def se_resnext101_32x4d(pretrained=False):

    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,

                  dropout_p=None, inplanes=64, input_3x3=False,

                  downsample_kernel_size=1, downsample_padding=0,

                  num_classes=1000)

    if pretrained:

        model.load_state_dict(torch.load(pretrained_settings['se_resnext101_32x4d']['url']))

    return model
learn = cnn_learner(data, base_arch=models.densenet169, metrics=[f1, accuracy])

# learn = cnn_learner(data, se_resnet152, metrics=[f1, accuracy])
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = 0.01
learn.fit_one_cycle(30, slice(lr))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
lr1 = 1e-4

learn.fit_one_cycle(15, slice(lr1/2.6**3, lr1))
learn.save('stage-2')
# learn=None

# gc.collect()
# learn = cnn_learner(data, base_arch=se_resnet152, metrics=[f1, accuracy]).load('stage-2');
# bs=32

# img_sz=224
# data = (src.transform(tfms, size=img_sz)

#         .databunch(path='.', bs=bs, device= torch.device('cuda:0')).normalize(imagenet_stats))



# learn.data = data
# data.show_batch(rows=3, figsize=(12,9))
# learn.freeze()
# learn.lr_find()

# learn.recorder.plot()
# lr2=1e-4
# learn.fit_one_cycle(3, slice(lr2))
# learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(1, slice(lr2/2.6**3, lr2/5))
df = pd.read_csv(d_path/'sample_submission.csv')

df.head()
test_preds = learn.get_preds(DatasetType.Test)

df['Predicted'] = test_preds[0].argmax(dim=1)

df.to_csv('submission.csv', index=False)
# !kaggle competitions submit -c iwildcam-2019-fgvc6 -f submission.csv -m "submit"