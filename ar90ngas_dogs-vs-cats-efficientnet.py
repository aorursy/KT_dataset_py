def __bootstrap__():

    import sys

    import base64

    import gzip

    import tarfile

    import os

    import io

    from pathlib import Path

    from tempfile import TemporaryDirectory



    # install required packages

    pkg_dataset_path = Path.cwd().parent / "input" / "dogs-vs-cats-efficientnet-requirements"

    pkg_path_list = []

    for p in pkg_dataset_path.glob("*"):

        if p.is_dir():

            pkg_config_files = [str(p.parent) for p in p.glob("**/*") if p.name in ["pyproject.toml", "setup.py"]]

            pkg_root_dir = min(pkg_config_files, key=len)

            pkg_path_list.append(pkg_root_dir)

        else:

            pkg_path_list.append(str(p))

    if 0 < len(pkg_path_list):

        pkg_paths = " ".join(pkg_path_list)

        os.system(f"pip install --no-deps {pkg_paths}")



    # this is base64 encoded source code

    tar_io = io.BytesIO(gzip.decompress(base64.b64decode("H4sIAEZzDF8C/0vJTy+OLyuOT04sKY5PTUvLTM5MzSvJSy3RNdAz1DPQLag01s3Lz0vVTcyr1CvPyGEgAxgAgZmJCZgGAnTayNjMEMaGiBuam5saMigYMNABlBaXJBYpKFBsDrrnhggI8GZmEQHSHECsCBZhgsqoAnEKrtShHx+fmZdZEh+vV1DJjGqIR2fOyi8BQFYwEOvjMwSSxPRSMotLdDPz0vL1wz1cXX24z3v4nr3I662rde7M+c1BBleMHxT5n/XW9dI5qb8pSOOEv67mKpZOpqPNnz57l9h2lzJ7dQcVG01eI+LV5RW800nocuMrrd7Sz58+B5UKf9baieY4mV9L/gszAllAbxqR5Dhf1xBHF8cQx6kTD3oxGwq0vf8tYqllcjhv+67SzO5F7tqujv2pN+amsZt7+m6aum+vp7bWJW99xTM19+2ZNy9p0GWbsv6JTXRwt6HITeGsVa7ZU6db60vzxF1KO8w7mbPN6Rvz8wCHeXt5ohbfvq60W9opPVZ32ouNyelHo33+r/O6dKJW94Jm9oPz1auDmKVWb07/JXFj4vTVtSGPGo7uackymX1E87tt1XnpZTP2F4VZXpnVK9SfcaWhy0X2Nbvu31PFlSvOmS3b/9tuA//Gk7X59kePJsUdlPh0MYfL8sn0/xvLQ1YtL01Yez3a/aCidNbkac4zt8ioqW95axdxvWRT+lXWyWLWEsE7vI6d5VT2tXBindJ+56Om1tJ3FrFfOOKdJ97Q2hiPGsQfwo6+uAVkxQGD2YCkIA5ydfYPcpl69mBUE1Djg93nNylyeNuUKEhIuan1eOj9vsA785CC+ZGqX35y1j+Y/p/ftXhRuCv/0rAp4VMtOPS2PqnyWjnBKTjri73D1ac8trbWnxnyf0/eXeXp7XEye7azyM+aG5+e3Fn0fUf/p6TM6y8F0g7M2fA/I0NqWqKUjwZv65pdmfsO/Xg7b4m4rv20+SrPdse8dtPNM/ru052+Nf6Gs+gmjdtv157Yd4Bnxzdm755O8Yoju92v/ahtWpfhXllW9/GGyBML7kT5zQdiHz/Z9jJl87zVR8O3lTMUXWAObvrrK8Rg8JM5wJuRSYQZd3aDgSWNDMRmPjQjMTIf3EhGV9KzIrrh6JkHYfgjsrISuvnoKQfJ8Uykp6MAb1Y2kG4WIMwDmpjLzDAKRsEoGAWjYBSMglEwCkbBKBgFo2AUjIJRMApGwSgYBaNgFIyCUTAKRsEoGAWjYBSMglEw/AAAxjVP1AAoAAA=")))

    with TemporaryDirectory() as temp_dir:

        with tarfile.open(fileobj=tar_io) as tar:

            for member in tar.getmembers():

                pkg_path = Path(temp_dir) / f"{member.name}"

                content_bytes = tar.extractfile(member).read()

                pkg_path.write_bytes(content_bytes)

                os.system("pip install --no-deps {pkg_path}".format(pkg_path=pkg_path))



    sys.path.append("/kaggle/working")

    os.environ.update({})

__bootstrap__()
import os

import torch

import torch.nn.functional as F

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

import timm

from pathlib import Path

from torchvision import transforms

from PIL import Image

import zipfile
if "KAGGLE_CONTAINER_NAME" in os.environ:

    import kaggle_timm_pretrained

    kaggle_timm_pretrained.patch()
ROOT_DIR =  Path(os.environ.get("ROOT_DIR", "../input/dogs-vs-cats"))

with zipfile.ZipFile(str(ROOT_DIR / "train.zip"),"r") as z:

    z.extractall(".")



with zipfile.ZipFile(str(ROOT_DIR / "test1.zip"),"r") as z:

    z.extractall(".")

    

TRAIN_DATA_DIR = Path("/kaggle/working/train")

TEST_DATA_DIR = Path("/kaggle/working/test1")
class DogsVsCatsDataset(Dataset):

    def __init__(self,  root_dir, transform=None, train=True):

        self._transform = transform

        self._train = train

        self._img_paths =list(root_dir.glob("*.jpg"))

        if not self._train:

            self._img_paths = sorted(self._img_paths, key=lambda p:int(int(p.stem)))

        

    def __len__(self):

        return len(self._img_paths)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        img_path = self._img_paths[idx]

        img = Image.open(img_path)

        if self._transform:

            img = self._transform(img)

            

        if self._train:

            label = int(img_path.name.startswith("dog"))

            return img, label

        return img
class Network(pl.LightningModule):

    def __init__(self):

        super().__init__()

        self.net = timm.create_model("efficientnet_b2", pretrained=True, num_classes=2)

        

    def forward(self, x):

        return self.net(x)

    

    def training_step(self, batch, batch_idx):

        x, y = batch

        loss = F.cross_entropy(self(x), y)

        return {'loss': loss}

    

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=0.02)

    

    def setup(self, stage):

        train_dataset = DogsVsCatsDataset(

            TRAIN_DATA_DIR,

            transform=transforms.Compose([

                transforms.Resize((224, 224)), 

                transforms.ToTensor()

            ])

        )

        train_size = int(len(train_dataset) * 0.8)

        val_size = int(len(train_dataset) - train_size)

        self._train_dataset, self._val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        self._test_dataset =DogsVsCatsDataset(

            TEST_DATA_DIR,

            transform=transforms.Compose([

                transforms.Resize((224, 224)), 

                transforms.ToTensor()

            ]),

            train=False

        )     

    

    def train_dataloader(self):

        return DataLoader(self._train_dataset, batch_size=32, num_workers=4, shuffle=True)



    def validation_step(self, batch, batch_idx):

        x ,y = batch

        loss = F.cross_entropy(self(x), y)

        return {'val_loss': loss}

    

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        return {"val_loss": avg_loss}

    

    def val_dataloader(self):

        return DataLoader(self._val_dataset, batch_size=4, num_workers=4)

    

    def test_step(self, batch, batch_idx):

        x = batch

        label = torch.argmax(self(x), dim=1)

        return {"label": (batch_idx, label)}



    def test_epoch_end(self, outputs):

        return dict([x["label"] for x in outputs])

    

    def test_dataloader(self):

        return DataLoader(self._test_dataset, batch_size=1)
model = Network()

trainer = pl.Trainer(gpus=1, max_epochs=10)

trainer.fit(model)
result = trainer.test()
submission_csv = "\n".join([f"{id},{label}" for id, label in result.items()])

Path("./submission.csv").write_text(submission_csv)