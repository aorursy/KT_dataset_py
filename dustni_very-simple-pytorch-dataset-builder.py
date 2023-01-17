import os
from PIL import Image
from torch.utils import data
class TGSDataset(data.Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.all_images = os.listdir(img_path)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        file_name = self.all_images[index]
        input_img = Image.open(os.path.join(self.img_path, file_name)).convert('L')
        mask_img = Image.open(os.path.join(self.mask_path, file_name)).convert('L')
        if self.transform is not None:
            input_img = self.transform(input_img)
            mask_img = self.transform(mask_img)
        return input_img, mask_img
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
def show_sample(tensor):
    '''
    :param data: [c,w,h]
    :return:
    '''
    img = tensor.data.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()
img_path = '../input/train/images'
mask_path = '../input/train/masks'
transform = transforms.Compose([transforms.ToTensor()])
tgs_dataset = TGSDataset(img_path, mask_path, transform)
tgs_dataloader = DataLoader(tgs_dataset, batch_size=8, shuffle=True, num_workers=0)
_iter = iter(tgs_dataloader)

example_imgs, example_masks = next(_iter)
example_imgs = torchvision.utils.make_grid(example_imgs, padding=0)
example_masks = torchvision.utils.make_grid(example_masks, padding=0)
example = torch.cat((example_imgs, example_masks), 1)

show_sample(example)
