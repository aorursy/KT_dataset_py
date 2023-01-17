import torch

import torchvision

from PIL import Image

import numpy as np

import pandas as pd

import pydicom

from fastai import vision

from sklearn.model_selection import train_test_split

from torchvision import transforms
train_path = vision.Path("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images")

test_path = vision.Path("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images")

all_classes = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
dataframe = pd.read_csv("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")

dataframe["Classes"] = dataframe.ID.str.rsplit(pat="_", expand=True)[2]

dataframe["ID"] = dataframe["ID"].apply(lambda x: "_".join(x.split('_')[:-1]))
dataframe = pd.pivot_table(dataframe, index="ID", columns="Classes", values="Label")

dataframe = pd.DataFrame(dataframe.to_records())
for curr_class in all_classes:

    dataframe[curr_class] = dataframe[curr_class].apply(

        lambda x: curr_class if x == 1 else "")

dataframe["Class"] = dataframe[all_classes].apply(

    lambda x: ' '.join(list(filter(None, x))), axis=1)

dataframe = dataframe.drop(columns=all_classes)

dataframe["ID"] = dataframe["ID"].apply(lambda x: x + ".dcm")
broken_samples = ["ID_6431af929.dcm"]

id_list = dataframe["ID"].tolist()

broken_indexes = [id_list.index(sample) for sample in broken_samples]

dataframe = dataframe.drop(broken_indexes).reset_index(drop=True)
train_df, val_df = train_test_split(dataframe, test_size=0.2)

train_df = train_df.reset_index(drop=True)

val_df = val_df.reset_index(drop=True)
class Data(vision.Dataset):

    """This class loads one sample data using a DataFrame.

    

    Attributes:

        df: Pandas DataFrame. DataFrame Structure:

            ----------------------------------------

            |      ID     |         Class          |

            ----------------------------------------

            | "name1.dcm" |          ""            | 

            ----------------------------------------

            | "name2.dcm" | "class1 class2 class3" |

            ----------------------------------------

            |     ...     |          ...           |

            ----------------------------------------

        classes: A list containing all image classes

        folder: Object of "vision.Path" class

        transfrom: Transforms from PyTorch (more about this:

            https://pytorch.org/docs/stable/torchvision/transforms.html)

    """

    def __init__(self, df, classes, folder, transform):

        self.df = df

        self.df_cols_name = self.df.columns

        self.classes = classes

        self.folder = folder

        self.transform = transform

        self.c = 6



    def __getitem__(self, idx):

        path_to_img = self.folder/self.df[self.df_cols_name[0]][idx]

        

        # Image processing

        img = pydicom.dcmread(str(path_to_img)).pixel_array

        img = np.maximum(img,0) / img.max() * 255

        img = img.astype("uint8")

        img = np.expand_dims(img, -1)

        img = np.broadcast_to(img, [img.shape[0], img.shape[1], 3])

        img = Image.fromarray(img)

        img = self.transform(img)

        

        # Label processing

        label = vision.torch.zeros(len(self.classes), dtype=vision.torch.long)

        all_image_classes = self.df[self.df_cols_name[1]][idx].split(' ')

        all_image_classes = list(filter(None, all_image_classes))

        for curr_class in all_image_classes:

            label[self.classes.index(curr_class)] = 1

        return img, label

    

    def __len__(self):

        return self.df.shape[0]
transform = transforms.Compose([transforms.RandomHorizontalFlip(),

                                transforms.Resize((224, 224)),

                                transforms.ToTensor(),

                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

                                ])
train_dataset = Data(df=train_df, classes=all_classes, folder=train_path, transform=transform)

train_dataloader = vision.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = Data(df=val_df, classes=all_classes, folder=train_path, transform=transform)

val_dataloader = vision.DataLoader(val_dataset, batch_size=16, shuffle=True)
databunch = vision.DataBunch(train_dataloader, val_dataloader)
import sys

sys.path.append("/kaggle/input/efficient-net-for-pytorch-or-fast-ai/")
from efficientnet_pytorch import model

model_eff = model.EfficientNet.from_pretrained('efficientnet-b7')
import math

import torch

from torch.optim.optimizer import Optimizer, required



class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):

        if not 0.0 <= lr:

            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:

            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:

            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:

            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        

        self.degenerated_to_sgd = degenerated_to_sgd

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):

            for param in params:

                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):

                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = group['buffer'][int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    elif self.degenerated_to_sgd:

                        step_size = 1.0 / (1 - beta1 ** state['step'])

                    else:

                        step_size = -1

                    buffered[2] = step_size



                # more conservative since it's an approximated value

                if N_sma >= 5:

                    if group['weight_decay'] != 0:

                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

                    p.data.copy_(p_data_fp32)

                elif step_size > 0:

                    if group['weight_decay'] != 0:

                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                    p.data.copy_(p_data_fp32)



        return loss
optar = vision.partial(RAdam)
len(dir(model_eff.parameters))
for i, elem in enumerate(model_eff.parameters()):

    if i == 26:

        del elem
model = vision.nn.Sequential(

                            model_eff,

                            vision.nn.Linear(1000, 6),

                            vision.nn.Sigmoid())
learn = vision.Learner(databunch, model,

                        metrics=vision.error_rate, loss_func=vision.nn.BCEWithLogitsLoss(),

                        opt_func=optar)
learn.fit_one_cycle(1, callbacks=[vision.callbacks.SaveModelCallback(learn, every='improvement', monitor='loss_func', mode='min', name='best')]).cuda()
test_df = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv")

all_test_images = test_df["ID"].apply(lambda x: "_".join(

    list(filter(None, x.split("_")[:-1])))).drop_duplicates().tolist()
test_transforms = [transforms.RandomVerticalFlip(p=1),

                   transforms.RandomHorizontalFlip(p=1)]
test_path = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"

for curr_img_path in all_test_images:

    img = pydicom.dcmread(str(f"{test_path}{curr_img_path}.dcm")).pixel_array

    img = np.maximum(img,0) / img.max() * 255

    img = img.astype("uint8")

    img = np.expand_dims(img, -1)

    img = np.broadcast_to(img, [img.shape[0], img.shape[1], 3])

    img = Image.fromarray(img)

    transformed_images = torch.zeros((0, 3, 224, 224))

    for test_transform in test_transforms:

        transformed_image = test_transform(img)

        transformed_image = transforms.functional.resize(transformed_image, (224, 224))

        transformed_image = np.array(transformed_image).reshape((3, 224, 224))

        transformed_tensor_image = torch.from_numpy(transformed_image).float()

        transformed_tensor_image /= 255.

        transformed_tensor_image = transforms.Normalize(

            [0.485, 0.456, 0.406],

            [0.229, 0.224, 0.225])(transformed_tensor_image)

        transformed_tensor_image = transformed_tensor_image.expand(1, -1, -1, -1)

        transformed_images = np.append(

            transformed_images, transformed_tensor_image, 0)

    transformed_tensor = torch.from_numpy(transformed_images)

    if torch.cuda.is_available():

        transformed_tensor = transformed_tensor.to("cuda:0")

    preds = model(transformed_tensor)

    preds = torch.mean(preds, 0).detach().cpu().numpy()

    preds = np.where(preds > 0.5, 1, 0)

    for i, elem in enumerate(preds):

        col_name = f"{curr_img_path}_{all_classes[i]}"

        img_idx = test_df.iloc[:, 0].tolist().index(col_name)

        test_df.iloc[img_idx, 1] = elem
test_df.to_csv("submission.csv", index=False)