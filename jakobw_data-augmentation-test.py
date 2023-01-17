!pip install opencv_transforms
import numpy as np

from torchvision import transforms



def data_augmentation(data):

    from opencv_transforms import transformscv

    

    ptrans=transforms.Compose([transforms.ColorJitter(saturation=0.5, hue=0.2)])

    cvtrans = transformscv.Compose([transformscv.RandomHorizontalFlip(),

                                    transformscv.RandomVerticalFlip(),

                                    transformscv.ColorJitter(brightness=0.5, contrast=0.5)])



    data=cvtrans(data)

    data=np.array(ptrans(Image.fromarray(data)))

    data=np.moveaxis(data, 2,0)

    

    return data
