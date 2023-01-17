from fastai.vision import *

import torchvision.transforms as T



# from own.show_filts import show_filts
import matplotlib.pyplot as plt

import math



def show_filts(model, n_cols=32, figsize=(24,2), global_scale=True):

    filts = list(model.parameters())[0]

    filts = filts.detach().cpu().numpy().transpose(0,2,3,1)

    n_filts = filts.shape[0]

    

    n_rows = math.ceil(n_filts/n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    axs = axs.flatten()

    for i, ax in enumerate(axs):

        ax = axs[i]

        if i < n_filts:

            filt = filts[i]

            ax.axis('off')

            if global_scale:

                filt = (filt - filts.min())/max(filts.max() - filts.min(), 1e-6)

            else:

                filt = (filt - filt.min())/max(filt.max() - filt.min(), 1e-6)

            ax.imshow(filt)

        else:

            fig.delaxes(ax)
path = Path('/media/nofreewill/Datasets_nvme/Visual/Imagenet-sz/80')

path.ls()
class AugDS(Dataset):

    def __init__(self, folder, tfms, recurse=False):

        self.x = get_image_files(folder, recurse=recurse)

        self.tfms = tfms

    

    def __len__(self):

        return len(self.x)

    

    def __getitem__(self, idx):

        img = PIL.Image.open(self.x[idx]).convert('RGB')

        img1 = self.tfms(img)

        img2 = self.tfms(img)

        return torch.stack((img1, img2)), torch.tensor([0, 0])
size = 64

tfms = T.Compose([T.RandomResizedCrop(size, scale=(0.5,1.)),

                  T.RandomOrder([T.RandomGrayscale(0.1),

                                 T.ColorJitter(.5,.5,.5,.1),

                                 T.RandomHorizontalFlip(0.5),

                                 T.RandomRotation(10., resample=2),

                                ]),

                  T.ToTensor(),

                 ])
ds_train = AugDS(path/'train', recurse=True, tfms=tfms)
ds_valid = AugDS(path/'val', recurse=True, tfms=tfms)
def collate_fn(batch):

    inp = [x[0] for x in batch]

    targ = [x[1] for x in batch]

    inp = torch.cat(inp)

    targ = torch.cat(targ)

    return inp, targ
data = ImageDataBunch.create(ds_train, ds_valid, bs=512, collate_fn=collate_fn)
x, y = data.train_ds[0]

x.shape
T.ToPILImage()(x[0])
T.ToPILImage()(x[1])
model = models.resnet18()

model.conv1, model.fc
h, s = 512, 256

model.fc = nn.Sequential(nn.Linear(model.fc.in_features, h),

                        nn.ReLU(inplace=True),

                        nn.Linear(h, s),

                        )

model.fc
device = data.device
def loss_fn(inp, targ):

    thau = 1.

    # Similarities

    inp_norm = inp / inp.norm(dim=1)[:,None]

    simils = torch.mm(inp_norm, inp_norm.transpose(0,1))

    # Good and Bad

    N = len(inp_norm)

    eye = (2*torch.eye(N)-1).to(device)

    eye[torch.arange(1,N, step=2), torch.arange(0,N, step=2)] = 1

    eye[torch.arange(0,N, step=2), torch.arange(1,N, step=2)] = 1

    #

    exps = torch.exp(simils)

    num = ((exps/thau)*(eye+1)/2).sum(dim=1)

    den = ((exps/thau)*(-eye+1)/2).sum(dim=1)

    loss = (-torch.log(num/den)).mean()

    return loss
learn = Learner(data, model, loss_func=loss_fn)
show_filts(learn.model)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 3e-3)
show_filts(learn.model)
show_filts(learn.model, global_scale=False)
learn.recorder.plot_losses()
path = untar_data(URLs.IMAGENETTE)

path.ls()
data = ImageDataBunch.from_folder(path, valid='val', size=224, ds_tfms=get_transforms())
model.fc
model.fc = nn.Linear(model.fc[0].in_features, data.c)

model.fc
learn = Learner(data, learn.model, metrics=[accuracy])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 1e-3)
show_filts(learn.model)
model = models.resnet18()

model.fc = nn.Linear(model.fc.in_features, data.c)
show_filts(model)
learn = Learner(data, model, metrics=[accuracy])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 1e-3)
show_filts(learn.model)