#Run 2 too check consistency in accuracy. last results : 0.98560



#from fastai import *

#from fastai.vision import *

#DATAPATH = Path('/kaggle/input/Kannada-MNIST/')



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))
%reload_ext autoreload

%autoreload 2

%matplotlib inline





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.vision import *

from fastai.metrics import *



import os

path = '/kaggle/input/Kannada-MNIST/'

print(os.listdir(path))



def random_seed(seed_value, use_cuda):

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False

        

random_seed(13,True)
class CustomImageList(ImageList):

    def open(self, fn):

        if(fn.size == 785):

            fn = fn[1:]

        img = fn.reshape(28,28)

        img = np.stack((img,)*3, axis=-1)

        return Image(pil2tensor(img, dtype=np.float32))

    

    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res

    

    @classmethod

    def from_csv_custom_test(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        print(res)

        return res

    

    

    

    @classmethod

    def from_df_custom(cls, path:PathOrStr, df:DataFrame, imgIdx:int=1, header:str='', **kwargs)->'ItemList': 

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res
test = CustomImageList.from_csv_custom_test(path=path, csv_name='test.csv', imgIdx=0)
#additional_aug=[[jitter(magnitude=-0.01)]]







data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)

                .split_by_rand_pct(.02)

                .label_from_df(cols='label') #cols='label'

                .add_test(test, label=0)

                .transform(get_transforms(do_flip=False,max_rotate=15,max_warp=0.4))

                .databunch(bs=128, num_workers=0)

                .normalize(imagenet_stats))



#,xtra_tfms=[[jitter(magnitude=-0.01)]]
data.show_batch(rows=3, figsize=(5,5))
def conv2(ni,nf,stride=2,ks=5): return conv_layer(ni,nf,stride=stride,ks=ks)



# Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
def mish(input):

  return input * torch.tanh(F.softplus(input))
class BasicBlock(nn.Module):

  expansion = 1



  def __init__(self, in_planes, planes, stride=1):

    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(

        in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = nn.Conv2d(

        planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

    self.bn2 = nn.BatchNorm2d(planes)



    self.shortcut = nn.Sequential()

    if stride != 1 or in_planes != self.expansion * planes:

      self.shortcut = nn.Sequential(

          nn.Conv2d(

              in_planes,

              self.expansion * planes,

              kernel_size=1,

              stride=stride,

              bias=False), nn.BatchNorm2d(self.expansion * planes))



  def forward(self, x):

    out = self.bn1(self.conv1(x))

    out = mish(out)

    out = self.bn2(self.conv2(out))

    out = mish(out)

    #out += nn.GroupNorm(Mish(self.shortcut(x)))

    shrt = self.shortcut(x)

    out += mish(shrt)

    return out





class ResNet(nn.Module):



  def __init__(self, block, num_blocks, num_classes=10):

    super(ResNet, self).__init__()

    self.in_planes = 64



    self.conv1 = nn.Conv2d(

        3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    self.bn1 = nn.BatchNorm2d(64)

    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    self.linear = nn.Linear(512 * block.expansion, 200)

    self.linear2 = nn.Linear(200, num_classes)

    self.bn_1d_1 = nn.BatchNorm1d(num_features=200)

    self.dropout = nn.Dropout(p=0.2)



  def _make_layer(self, block, planes, num_blocks, stride):

    strides = [stride] + [1] * (num_blocks - 1)

    layers = []

    for stride in strides:

      layers.append(block(self.in_planes, planes, stride))

      self.in_planes = planes * block.expansion

    return nn.Sequential(*layers)



  def forward(self, x):

    out = self.bn1(self.conv1(x))

    out = mish(out)

    out = self.layer1(out)

    out = self.layer2(out)

    out = self.layer3(out)

    out = self.layer4(out)

    out = F.avg_pool2d(out, 4)

    out = self.dropout(torch.flatten(out, 1))

    out = self.linear(out)

    out = self.dropout(mish(self.bn_1d_1(out)))

    out = self.linear2(out)

    return F.log_softmax(out, dim=1)





def ResNetCustom():

  return ResNet(BasicBlock, [4, 4, 4, 4])
model = ResNetCustom()
model


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy])
learn.fit_one_cycle(25)
learn.fit(1, 1e-5)  #Finetune loop

learn.fit(1, 3e-6)  #Finetune loop

learn.fit(1, 1e-6)

#learn.fit(1, 3e-7)  #Finetune loop

#learn.fit(1, 1e-7)  #Finetune loop

#learn.fit(1, 3e-8)

#learn.fit(1, 1e-8)  



#learn.fit(3, 3e-9)#fine tune with very low LR
# get the predictions

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'id': list(range(0,len(labels))), 'label': labels})

submission_df.to_csv(f'submission.csv', index=False)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9)