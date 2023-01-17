!pip3 install --upgrade fastai
from fastai.vision.all import *

def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything()
root = Path("/kaggle/input/")

disk = root / 'drive-train' / 'training' / 'mask'

image = root / 'drive-train' / 'training'/ 'images'

bv = root / 'drive-train' / 'training'/ '1st_manual'
images = sorted(image.ls(), key=lambda x: x.name)

bvs = sorted(bv.ls(), key=lambda x: x.name)

disks = sorted(disk.ls(), key=lambda x: x.name)
im_plot = []

bv_plot = []

disk_plot = []



for j in images:

    im = Image.open(j)

    im_plot.append(im)



for j in bvs:

    im = Image.open(j)

    bv_plot.append(im)

    

for j in disks:

    im = Image.open(j)

    disk_plot.append(im)
num_images = 5

f, plots = plt.subplots(3, num_images, sharex='col', sharey='row', figsize=(15, 10),  constrained_layout=True)



for i in range(num_images):

    plots[0, i].axis('off')

    plots[0, i].imshow(im_plot[i])

    plots[1, i].axis('off')

    plots[1, i].imshow(bv_plot[i], cmap='gray')

    plots[2, i].axis('off')

    plots[2, i].imshow(disk_plot[i], cmap='gray')