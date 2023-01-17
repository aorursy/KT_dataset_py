!pip install orjson efficientnet_pytorch
import os



import numpy as np

import PIL.Image as Img

import torch

import orjson as json



from torchvision import transforms

from efficientnet_pytorch import EfficientNet

from tqdm.auto import tqdm
class VaccineImgDataset:

    def __init__(self, img_dir: str, model_name: str):

        self.dir = img_dir

        self.files = list(filter(lambda x: ".npy" in x, os.listdir(self.dir)))

        self.model = EfficientNet.from_pretrained(model_name).to(torch.device('cuda:0'))

        self.pipeline = transforms.Compose([

            transforms.Resize(224),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ])

        

    @staticmethod

    def load_image(path: str) -> np.ndarray:

        image = np.load(path)

        image = np.stack((image,)*3, axis=-1)

        return Img.fromarray(image.astype(np.uint8))

        

    def __len__(self):

        return len(self.files)

    

    def ids(self):

        return list(map(lambda x: x.replace('.npy', ''), self.files))

    

    def __getitem__(self, idx):

        image = VaccineImgDataset.load_image(os.path.join(self.dir, self.files[idx]))

        image = self.pipeline(image).unsqueeze(0).to(torch.device("cuda:0"))

        with torch.no_grad():

            features = self.model(image).squeeze().cpu().numpy()

        return features
ds = VaccineImgDataset("/kaggle/input/stanford-covid-vaccine/bpps/", "efficientnet-b7")
features = [x for x in tqdm(ds)]

mapping = {k:v for k,v in zip(ds.ids(), features)}
with open("efficientnet-features.json", 'wb') as f:

    f.write(json.dumps(mapping, option=json.OPT_SERIALIZE_NUMPY))