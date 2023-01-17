UNLABELED_PATH = ["/kaggle/input/ecssd/ECSSD/Image", "/kaggle/input/ecssd/ECSSD/Mask"]

LABELED_PATH = ["/kaggle/input/pascal-s/Pascal-S/Image", "/kaggle/input/pascal-s/Pascal-S/Mask"]
!cd /kaggle/input/pascal-s/ && ls
import os



import torch.utils.data as data

from PIL import Image

from torchvision import transforms

from torch.utils.data import DataLoader

import math





class JointResize(object):

    def __init__(self, size):

        if isinstance(size, int):

            self.size = (size, size)

        elif isinstance(size, tuple):

            self.size = size

        else:

            raise RuntimeError("size参数请设置为int或者tuple")



    def __call__(self, img, mask):

        img = img.resize(self.size)

        mask = mask.resize(self.size)

        return img, mask



def make_dataset(root, prefix=('jpg', 'png')):

    img_path = root[0]

    gt_path = root[1]

    img_list = [os.path.splitext(f)[0] for f in os.listdir(img_path) if f.endswith(prefix[0])]

    return [(os.path.join(img_path, img_name + prefix[0]), os.path.join(gt_path, img_name + prefix[1])) for img_name in img_list]





# 仅针对训练集

class ImageFolder(data.Dataset):

    def __init__(self, root, mode, in_size, prefix, use_bigt=False, split_rate=(1, 3)):

        """split_rate = label:unlabel"""

        assert isinstance(mode, str), 'isTrain参数错误，应该为bool类型'

        self.root_labeled = root[0]

        self.mode = mode

        self.use_bigt = use_bigt

        

        self.imgs_labeled = make_dataset(self.root_labeled, prefix=prefix)

        self.split_rate = split_rate

        self.r_l_rate = split_rate[1] // split_rate[0]

        len_labeled = len(self.imgs_labeled)



        self.root_unlabeled = root[1]

        self.imgs_unlabeled = make_dataset(self.root_unlabeled, prefix=prefix)

        len_unlabeled = len(self.imgs_unlabeled)



        len_unlabeled = self.r_l_rate * len_labeled

        self.imgs_unlabeled = self.imgs_unlabeled * (self.r_l_rate + math.ceil(len_labeled / len_unlabeled))  # 扩展无标签的数据列表

        self.imgs_unlabeled = self.imgs_unlabeled[0:len_unlabeled]



        self.length = len_labeled + len_unlabeled

        print(f"使用扩充比例为：{len(self.imgs_labeled) / len(self.imgs_unlabeled)}")



        # 仅是为了简单而仅使用一种变换

        self.train_joint_transform = JointResize(in_size)

        self.train_img_transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 处理的是Tensor

        ])

        # ToTensor 操作会将 PIL.Image 或形状为 H×W×D，数值范围为 [0, 255] 的 np.ndarray 转换为形状为 D×H×W，

        # 数值范围为 [0.0, 1.0] 的 torch.Tensor。

        self.train_gt_transform = transforms.ToTensor()



    def __getitem__(self, index):

        if index % (self.r_l_rate + 1) == 0:

            labeled_index = index // (self.r_l_rate + 1)

            img_path, gt_path = self.imgs_labeled[labeled_index]  # 0, 1 => 10550

        else:

            unlabeled_index = index // (self.r_l_rate + 1) + index % (self.r_l_rate + 1)

            img_path, gt_path = self.imgs_unlabeled[unlabeled_index]  # 1, 2, 3



        img = Image.open(img_path).convert('RGB')

        img_name = (img_path.split(os.sep)[-1]).split('.')[0]



        gt = Image.open(gt_path).convert('L')

        img, gt = self.train_joint_transform(img, gt)

        img = self.train_img_transform(img)

        gt = self.train_gt_transform(gt)

        if self.use_bigt:

            gt = gt.ge(0.5).float()  # 二值化

        return img, gt, img_name  # 输出名字方便比较



    def __len__(self):

        return self.length

    

print(f" ==>> 使用的训练集 <<==\n -->> LABELED_PATH：{LABELED_PATH}\n -->> UNLABELED_PATH：{UNLABELED_PATH}")

train_set = ImageFolder((LABELED_PATH, UNLABELED_PATH), "train", 320, prefix=('.jpg', '.png'), use_bigt=True, split_rate=(12, 36))

# 由于train_set内部的比例顺序是固定的，所以为了保持比例关系，不能再使用`shuffle=True`

train_loader = DataLoader(train_set, batch_size=48, num_workers=8, shuffle=False, drop_last=True, pin_memory=True)  



for train_idx, train_data in enumerate(train_loader):

    train_inputs, train_gts, train_names = train_data

    print(train_names)

    

    # 正常训练中下面应该有，这里为了方便就关掉了

    # train_inputs = train_inputs.to(self.dev)

    # train_gts = train_gts.to(self.dev)

    train_labeled_inputs, train_unlabeled_inputs = train_inputs.split((12, 36), dim=0)

    train_labeled_gts, _ = train_gts.split((12, 36), dim=0)



    # otr_total = self.net(train_inputs)

    # labeled_otr, unlabeled_otr = otr_total.split((12, 36), dim=0)

    # with torch.no_grad():

    #     ema_unlabeled_otr = ema_model(train_unlabeled_inputs)

    print(" ==>> 一个Batch结束了 <<== ")

    if train_idx == 2:

        break

print(" ==>> 一个Epoch结束了 <<== ")
import os



import torch.utils.data as data

from PIL import Image

import torch

from torchvision import transforms

from torch.utils.data import DataLoader

import math





class JointResize(object):

    def __init__(self, size):

        if isinstance(size, int):

            self.size = (size, size)

        elif isinstance(size, tuple):

            self.size = size

        else:

            raise RuntimeError("size参数请设置为int或者tuple")



    def __call__(self, img, mask):

        img = img.resize(self.size)

        mask = mask.resize(self.size)

        return img, mask



def make_dataset(root, prefix=('jpg', 'png')):

    img_path = root[0]

    gt_path = root[1]

    img_list = [os.path.splitext(f)[0] for f in os.listdir(img_path) if f.endswith(prefix[0])]

    return [(os.path.join(img_path, img_name + prefix[0]), os.path.join(gt_path, img_name + prefix[1])) for img_name in img_list]





# 仅针对训练集

class ImageFolder(data.Dataset):

    def __init__(self, root, mode, in_size, prefix, use_bigt=False, split_rate=(1, 3)):

        """split_rate = label:unlabel"""

        assert isinstance(mode, str), 'isTrain参数错误，应该为bool类型'

        self.mode = mode

        self.use_bigt = use_bigt

        self.split_rate = split_rate

        self.r_l_rate = split_rate[1] // split_rate[0]



        self.root_labeled = root[0]

        self.imgs_labeled = make_dataset(self.root_labeled, prefix=prefix)



        len_labeled = len(self.imgs_labeled)

        self.length = len_labeled



        self.root_unlabeled = root[1]

        self.imgs_unlabeled = make_dataset(self.root_unlabeled, prefix=prefix)

        

        len_unlabeled = self.r_l_rate * len_labeled

        

        self.imgs_unlabeled = self.imgs_unlabeled * (self.r_l_rate + math.ceil(len_labeled / len_unlabeled))  # 扩展无标签的数据列表

        self.imgs_unlabeled = self.imgs_unlabeled[0:len_unlabeled]



        print(f"使用比例为：{len_labeled / len_unlabeled}")



        # 仅是为了简单而仅使用一种变换

        self.train_joint_transform = JointResize(in_size)

        self.train_img_transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 处理的是Tensor

        ])

        # ToTensor 操作会将 PIL.Image 或形状为 H×W×D，数值范围为 [0, 255] 的 np.ndarray 转换为形状为 D×H×W，

        # 数值范围为 [0.0, 1.0] 的 torch.Tensor。

        self.train_gt_transform = transforms.ToTensor()



    def __getitem__(self, index):

        # 这里一次性读取最简化比例数量的样本，所有的样本需要单独处理

        img_labeled_path, gt_labeled_path = self.imgs_labeled[index]  # 0, 1 => 850

        img_labeled = Image.open(img_labeled_path).convert('RGB')

        img_labeled_name = (img_labeled_path.split(os.sep)[-1]).split('.')[0]



        gt_labeled = Image.open(gt_labeled_path).convert('L')

        back_gt_labeled = gt_labeled  # 用于无标签数据使用联合调整函数的时候代替无标签数据真值进行占位

        img_labeled, gt_labeled = self.train_joint_transform(img_labeled, gt_labeled)

        img_labeled = self.train_img_transform(img_labeled)

        gt_labeled = self.train_gt_transform(gt_labeled)

        if self.use_bigt:

            gt_labeled = gt_labeled.ge(0.5).float()  # 二值化

        data_labeled = [img_labeled, gt_labeled, img_labeled_name]

        

        data_unlabeled = [[], []]

        for idx_periter in range(self.r_l_rate):

            # 这里不再使用真值，直接使用`_`接收

            img_unlabeled_path, _ = self.imgs_unlabeled[index//self.r_l_rate+idx_periter]  # 0, 1, 2, 3 => 3*850

            img_unlabeled = Image.open(img_unlabeled_path).convert('RGB')

            img_unlabeled_name = (img_unlabeled_path.split(os.sep)[-1]).split('.')[0]



            img_unlabeled, _ = self.train_joint_transform(img_unlabeled, back_gt_labeled)  # 这里为了使用那个联合调整的转换类，使用上面的target进行替代，但是要注意，不要再返回了

            img_unlabeled = self.train_img_transform(img_unlabeled)

                        

            data_unlabeled[0].append(img_unlabeled)

            data_unlabeled[1].append(img_unlabeled_name)



        return data_labeled, data_unlabeled  # 输出名字方便比较



    def __len__(self):

        return self.length

    

print(f" ==>> 使用的训练集 <<==\n -->> LABELED_PATH：{LABELED_PATH}\n -->> UNLABELED_PATH：{UNLABELED_PATH}")

train_set = ImageFolder((LABELED_PATH, UNLABELED_PATH), "train", 320, prefix=('.jpg', '.png'), use_bigt=True, split_rate=(12, 36))

# 由于train_set内部的比例顺序已经被固定到每一次iter中，所以可以使用`shuffle=True`

train_loader = DataLoader(train_set, batch_size=12, num_workers=8, shuffle=True, drop_last=False, pin_memory=True)  



for train_idx, train_data in enumerate(train_loader):

    data_labeled, data_unlabeled = train_data

    

    train_labeled_inputs, train_labeled_gts, train_labeled_names = data_labeled

    print(train_labeled_inputs.size(), train_labeled_gts.size(), train_labeled_names)

    

    train_unlabeled_inputs_list, train_unlabeled_names = data_unlabeled

    train_unlabeled_inputs = torch.cat(train_unlabeled_inputs_list, dim=0)

    print(train_unlabeled_inputs.size(), train_unlabeled_names)

    

    train_labeled_inputs_batchsize = train_labeled_inputs.size(0)

    train_unlabeled_inputs_batchsize = train_unlabeled_inputs.size(0)

    

    # 正常训练中下面应该有，这里为了方便就关掉了，这里之所以不先进行cat再进行to(dev)，是为了便于后面ema_model输入的时候使用一个已经在gpu上的张量，免去了再次搬运的麻烦

    # train_labeled_inputs = train_labeled_inputs.to(dev)

    # train_unlabeled_inputs = train_unlabeled_inputs.to(dev)

    # train_gts = train_labeled_gts.to(self.dev)

    train_inputs = torch.cat([train_labeled_inputs, train_unlabeled_inputs], dim=0)



    # otr_total = net(train_inputs)

    # labeled_otr, unlabeled_otr = otr_total.split((train_labeled_inputs_batchsize, train_unlabeled_inputs_batchsize), dim=0)

    # with torch.no_grad():

    #     ema_unlabeled_otr = ema_model(train_unlabeled_inputs)

    print(" ==>> 一个Batch结束了 <<== ")

    if train_idx == 2:

        break

print(" ==>> 一个Epoch结束了 <<== ")
sampler = [x for x in range(10)]

print(f"原始Sampler：{sampler}")



from torch.utils.data.sampler import SequentialSampler

print(f"顺序采样：{[x for x in SequentialSampler(sampler)]}")



from torch.utils.data.sampler import RandomSampler

print(f"随机置乱：{[x for x in RandomSampler(data_source=sampler, replacement=True, num_samples=5)]}")
import os



import torch.utils.data as data

from PIL import Image

import torch

from torchvision import transforms

from torch.utils.data import DataLoader

import math





class JointResize(object):

    def __init__(self, size):

        if isinstance(size, int):

            self.size = (size, size)

        elif isinstance(size, tuple):

            self.size = size

        else:

            raise RuntimeError("size参数请设置为int或者tuple")



    def __call__(self, img, mask):

        img = img.resize(self.size)

        mask = mask.resize(self.size)

        return img, mask



def make_dataset(root, prefix=('jpg', 'png')):

    img_path = root[0]

    gt_path = root[1]

    img_list = [os.path.splitext(f)[0] for f in os.listdir(img_path) if f.endswith(prefix[0])]

    return [(os.path.join(img_path, img_name + prefix[0]), os.path.join(gt_path, img_name + prefix[1])) for img_name in img_list]





# 仅针对训练集

class ImageFolder(data.Dataset):

    def __init__(self, root, mode, in_size, prefix, use_bigt=False, split_rate=(1, 3)):

        """split_rate = label:unlabel"""

        assert isinstance(mode, str), 'isTrain参数错误，应该为bool类型'

        self.mode = mode

        self.use_bigt = use_bigt

        self.split_rate = split_rate

        self.r_l_rate = split_rate[1] // split_rate[0]



        self.root_labeled = root[0]

        self.imgs_labeled = make_dataset(self.root_labeled, prefix=prefix)



        len_labeled = len(self.imgs_labeled)

        self.length = len_labeled



        self.root_unlabeled = root[1]

        self.imgs_unlabeled = make_dataset(self.root_unlabeled, prefix=prefix)

        

        len_unlabeled = self.r_l_rate * len_labeled

        

        self.imgs_unlabeled = self.imgs_unlabeled * (self.r_l_rate + math.ceil(len_labeled / len_unlabeled))  # 扩展无标签的数据列表

        self.imgs_unlabeled = self.imgs_unlabeled[0:len_unlabeled]



        print(f"使用比例为：{len_labeled / len_unlabeled}")



        # 仅是为了简单而仅使用一种变换

        self.train_joint_transform = JointResize(in_size)

        self.train_img_transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 处理的是Tensor

        ])

        # ToTensor 操作会将 PIL.Image 或形状为 H×W×D，数值范围为 [0, 255] 的 np.ndarray 转换为形状为 D×H×W，

        # 数值范围为 [0.0, 1.0] 的 torch.Tensor。

        self.train_gt_transform = transforms.ToTensor()



    def __getitem__(self, index):

        # 这里一次性读取最简化比例数量的样本，所有的样本需要单独处理

        img_labeled_path, gt_labeled_path = self.imgs_labeled[index]  # 0, 1 => 850

        img_labeled = Image.open(img_labeled_path).convert('RGB')

        img_labeled_name = (img_labeled_path.split(os.sep)[-1]).split('.')[0]



        gt_labeled = Image.open(gt_labeled_path).convert('L')

        back_gt_labeled = gt_labeled  # 用于无标签数据使用联合调整函数的时候代替无标签数据真值进行占位

        img_labeled, gt_labeled = self.train_joint_transform(img_labeled, gt_labeled)

        img_labeled = self.train_img_transform(img_labeled)

        gt_labeled = self.train_gt_transform(gt_labeled)

        if self.use_bigt:

            gt_labeled = gt_labeled.ge(0.5).float()  # 二值化

        data_labeled = [img_labeled, gt_labeled, img_labeled_name]

        

        data_unlabeled = [[], []]

        for idx_periter in range(self.r_l_rate):

            # 这里不再使用真值，直接使用`_`接收

            img_unlabeled_path, _ = self.imgs_unlabeled[index//self.r_l_rate+idx_periter]  # 0, 1, 2, 3 => 3*850

            img_unlabeled = Image.open(img_unlabeled_path).convert('RGB')

            img_unlabeled_name = (img_unlabeled_path.split(os.sep)[-1]).split('.')[0]



            img_unlabeled, _ = self.train_joint_transform(img_unlabeled, back_gt_labeled)  # 这里为了使用那个联合调整的转换类，使用上面的target进行替代，但是要注意，不要再返回了

            img_unlabeled = self.train_img_transform(img_unlabeled)

                        

            data_unlabeled[0].append(img_unlabeled)

            data_unlabeled[1].append(img_unlabeled_name)



        return data_labeled, data_unlabeled  # 输出名字方便比较



    def __len__(self):

        return self.length

    

    

def my_collate(batch):

    # 针对送进来的一个batch的数据进行整合，batch的各项表示各个样本

    # batch 仅有一项 batch[0] 对应于下面的 train_data

    # batch[0][0], batch[0][1] <==> data_labeled, data_unlabeled = train_data

    # batch[0][0][0], batch[0][0][1], batch[0][0][2] <==> train_labeled_inputs, train_labeled_gts, train_labeled_names = data_labeled

    # batch[0][1][0], batch[0][2][1] <==> train_unlabeled_inputs_list, train_unlabeled_names = data_unlabeled

    

    # 最直接的方法：

    train_labeled_inputs, train_labeled_gts, train_labeled_names = [], [], []

    train_unlabeled_inputs_list, train_unlabeled_names = [], []

    for batch_iter in batch:

        x, y = batch_iter

        train_labeled_inputs.append(x[0])

        train_labeled_gts.append(x[1])

        train_labeled_names.append(x[2])

        

        train_unlabeled_inputs_list += y[0]

        train_unlabeled_names += y[1]



    train_labeled_inputs = torch.stack(train_labeled_inputs, 0)

    train_unlabeled_inputs_list = torch.stack(train_unlabeled_inputs_list, 0)

    train_labeled_gts = torch.stack(train_labeled_gts, 0)

    print(train_unlabeled_inputs_list.size())

    return ([train_labeled_inputs, train_unlabeled_inputs_list], [train_labeled_gts],

            [train_labeled_names, train_unlabeled_names])



print(f" ==>> 使用的训练集 <<==\n -->> LABELED_PATH：{LABELED_PATH}\n -->> UNLABELED_PATH：{UNLABELED_PATH}")

train_set = ImageFolder((LABELED_PATH, UNLABELED_PATH), "train", 320, prefix=('.jpg', '.png'), use_bigt=True, split_rate=(3, 9))

# a simple custom collate function, just to show the idea

train_loader = DataLoader(train_set, batch_size=3, num_workers=4, collate_fn=my_collate, shuffle=True, drop_last=False, pin_memory=True)

print(" ==>> data_loader构建完毕 <<==")



for train_idx, train_data in enumerate(train_loader):



    train_inputs, train_gts, train_names = train_data

    

    train_labeled_inputs, train_unlabeled_inputs = train_inputs

    train_labeled_gts = train_gts[0]

    train_labeled_names, train_unlabeled_names = train_names

    print("-->>", train_labeled_inputs.size(), train_labeled_gts.size(), train_labeled_names)

    print("-->>", train_unlabeled_inputs.size(), train_unlabeled_names)

    

    train_labeled_inputs_batchsize = train_labeled_inputs.size(0)

    train_unlabeled_inputs_batchsize = train_unlabeled_inputs.size(0)

    

    # 正常训练中下面应该有，这里为了方便就关掉了，这里之所以不先进行cat再进行to(dev)，是为了便于后面ema_model输入的时候使用一个已经在gpu上的张量，免去了再次搬运的麻烦

    # train_labeled_inputs = train_labeled_inputs.to(dev)

    # train_unlabeled_inputs = train_unlabeled_inputs.to(dev)

    # train_gts = train_labeled_gts.to(self.dev)

    train_inputs = torch.cat([train_labeled_inputs, train_unlabeled_inputs], dim=0)



    # otr_total = net(train_inputs)

    # labeled_otr, unlabeled_otr = otr_total.split((train_labeled_inputs_batchsize, train_unlabeled_inputs_batchsize), dim=0)

    # with torch.no_grad():

    #     ema_unlabeled_otr = ema_model(train_unlabeled_inputs)

    print(" ==>> 一个Batch结束了 <<== ")

    if train_idx == 0:

        break

print(" ==>> 一个Epoch结束了 <<== ")