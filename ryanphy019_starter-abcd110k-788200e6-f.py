%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *



import matplotlib.pyplot as plt
path = Path('../input/bdd100k_seg/bdd100k/seg/')

path.ls()
tmpp = path/'images'/'train'/'04a328c5-7d9c636f.jpg'
tmpp
tmpp.parts
img = open_image(path/'images'/'val'/'a00d3a96-00000000.jpg')

color_label = open_mask(path/'color_labels'/'val'/'a00d3a96-00000000_train_color.png')

label = open_image(path/'labels'/'val'/'a00d3a96-00000000_train_id.png')
img.show()

color_label.show()
color_label.data
import PIL.Image as PilImage



def getClassValues(label_names):



    containedValues = set([])



    for i in range(len(label_names)):

        tmp = open_mask(label_names[i])

        tmp = tmp.data.numpy().flatten()

        tmp = set(tmp)

        containedValues = containedValues.union(tmp)

    

    return list(containedValues)



def replaceMaskValuesFromZeroToN(mask, 

                                 containedValues):



    numberOfClasses = len(containedValues)

    newMask = np.zeros(mask.shape)



    for i in range(numberOfClasses):

        newMask[mask == containedValues[i]] = i

    

    return newMask



def convertMaskToPilAndSave(mask, 

                            saveTo):



    imageSize = mask.squeeze().shape



    im = PilImage.new('L',(imageSize[1],imageSize[0]))

    im.putdata(mask.astype('uint8').ravel())

    im.save(saveTo)



def convertMasksToGrayscaleZeroToN(pathToLabels,

                                   saveToPath):



    label_names = get_image_files(pathToLabels)

    containedValues = getClassValues(label_names)



    for currentFile in label_names:

        currentMask = open_mask(currentFile).data.numpy()

        convertedMask = replaceMaskValuesFromZeroToN(currentMask, containedValues)

        convertMaskToPilAndSave(convertedMask, saveToPath/f'{currentFile.name}')

    

    print('Conversion finished!')
pathToLabels = path/'color_labels/val'

saveToPath = Path('/kaggle/working/converted_masks/')

convertMasksToGrayscaleZeroToN(pathToLabels, saveToPath)
mask2=open_mask(saveToPath/'a00d3a96-00000000_train_color.png')

mask2.show()

mask2.data
def get_y_func(x):

    y = saveToPath/f'{x.stem}_train_color.png'

    return y
classes = ['banner',

'billboard',

'lane divider',

'parking sign',

'pole',

'polegroup',

'street light',

'traffic cone',

'traffic device',

'traffic light',

'traffic sign',

'sign frame',

'person',

'rider',

'bicycle',

'bus',

'car',

'caravan',

'motorcycle']

# 'trailer',

# 'train',

# 'truck',

# 'void']
src = (SegmentationItemList.from_folder(path/'images'/'val')

                            .split_by_rand_pct()

                            .label_from_func(get_y_func, classes=classes))
data = (src.transform(tfms=get_transforms(), size=128, tfm_y=True)

           .databunch(bs=4).normalize(imagenet_stats))
data.show_batch(rows=2, figsize=(10,7))
def acc_bdd(input, target):

    target = target.squeeze(1)

#     mask = target != void_code

    return (input.argmax(dim=1)==target).float().mean()
learn=unet_learner(data, models.resnet34,metrics=acc_bdd,path='/kaggle/working')
lr_find(learn)
learn.recorder.plot()
learn.fit_one_cycle(10,max_lr=2e-4)