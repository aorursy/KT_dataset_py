from fastai.vision import *

from fastai.basic_data import *
folder='BoJo'; file='../input/BoJo.csv'



path = Path('../pics')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)



download_images('../input/BoJo.csv', dest, max_pics=120, max_workers=0)

verify_images(path, delete=True, max_size=120)
folder='JeremyC'; file='../input/JeremyC.csv'



path = Path('../pics')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)



download_images('../input/JeremyC.csv', dest, max_pics=120, max_workers=0)



verify_images(path, delete=True, max_size=120)
folder='JoSwinson'; file='../input/JoSwinson.csv'



path = Path('../pics')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)



download_images('../input/JoSwinson.csv', dest, max_pics=120, max_workers=0)



verify_images(path, delete=True, max_size=120)
folder='NicolaSturgeon'; file='../input/NicolaSturgeon.csv'



path = Path('NicolaSturgeon_pics')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)



download_images('../input/NicolaSturgeon.csv', dest, max_pics=120, max_workers=0)



verify_images(path, delete=True, max_size=500)
classes = ['BoJo', 'JeremyC', 'JoSwinson', 'NicolaSturgeon']
cnt=0

dirname = '../pics'

for filename in dirname:

    try:

        img=Image.open(dirname+"/"+filename)

    except OSError:

        print("FILE: ", filename, "is corrupt!")

        cnt+=1

        print(dirname+"/"+filename)

        #os.remove(dirname+"/"+filename)

print("Successfully Completed Operation! Files Courrupted are ", cnt)
import os

for dirname, _, filenames in os.walk('../pics'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
np.random.seed(42)



src = ImageDataBunch.from_folder('../pics',train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))