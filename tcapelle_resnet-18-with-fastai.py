import fastai
from fastai.vision import *
from pathlib import Path
import cv2
torch.backends.cudnn.benchmark = False
fastai.__version__, torch.__version__
MASKS = 'train.csv'

PATH = Path('/kaggle/input')
TRAIN = Path('train')
TEST = Path('test')
TMP = Path('/kaggle/working')

SAMPLE = Path('sample_submission.csv')

seg = pd.read_csv(PATH/MASKS)
sample_sub = pd.read_csv(PATH/SAMPLE)
train_names = list(seg.Id.values)
test_names = list(sample_sub.Id.values)

classes = [str(l) for l in range(28)]
df = pd.read_csv(PATH/MASKS); len(df)
arch = models.resnet18;
stats = ([0.08069, 0.05258, 0.05487], [0.13704,0.10145, 0.15313])
tfms = get_transforms(do_flip=True, flip_vert=True, 
                      max_lighting=0.1, max_warp=0.4)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()
def open_image4d(path:PathOrStr)->Image:
    '''open RGBA image from 4 different 1-channel files.
    return: numpy array [4, sz, sz]'''
    path=str(path)
    flags = cv2.IMREAD_GRAYSCALE
    red = cv2.imread(path+ '_red.png', flags)
    blue = cv2.imread(path+ '_blue.png', flags)
    green = cv2.imread(path+ '_green.png', flags)
#     yellow = cv2.imread(path+ '_yellow.png', flags)
    im = np.stack(([red, green, blue]))

    return Image(Tensor(im/255).float())
class MyImageItemList(ImageItemList):
    def open(self, fn:PathOrStr)->Image:
        return open_image4d(fn)
def get_data(sz=64, bs=64, pct=0.2, sample=5000):
#     sz, pct, bs = 64, 0.2, 64
    src = (MyImageItemList.from_df(df=df, path=PATH, folder=TRAIN)
           .random_split_by_pct(pct)
           .label_from_df(sep=' ', classes=classes)
           .add_test([PATH/TEST/f for f in test_names]))
    data = (src.transform(tfms, size=sz)
            .databunch(bs=bs, num_workers=0).normalize(stats)) #this really sucks!
    return data
data = get_data(sample=100)
# data.show_batch(rows=3, figsize=(12,9))
f1 = partial(fbeta, beta=1)

def get_learner(data, focal=False, fp16=False):
    learn = create_cnn(data, arch, metrics=[accuracy_thresh, f1], 
               callback_fns=[partial(GradientClipping, clip=0.1), ShowGraph], model_dir=TMP)
    if focal: learn.loss_func = FocalLoss()
    if fp16: learn.to_fp16();
    return learn.mixup(stack_y=False)
data = get_data(256, 128, 0.1)
learn = get_learner(data, focal=True, fp16=True)
learn.lr_find()
learn.recorder.plot()
lr = 1e-2
learn.fit_one_cycle(3,slice(lr))
learn.unfreeze()
learn.fit_one_cycle(4,slice(lr/10, lr/3))
learn.save('r18_256')
learn.data.test_dl.add_tfm(to_half)
p,t = learn.get_preds(ds_type=DatasetType.Test)
model_name = 'r18_256'
preds = to_np(p.sigmoid())  #Check if we are using focal loss or BCE.
np.save(model_name, preds)  #save for further model ensemble
threshold = 0.4 #ths
print(preds.shape)
classes = np.array(data.classes)
res = np.array([" ".join(classes[(np.where(pp>threshold))])for pp in preds])
frame = pd.DataFrame(np.array([test_names, res]).T, columns = ['Id','Predicted'])
frame.to_csv(f'{model_name}.csv', index=False)
frame.head()

