import numpy as np
import pandas as pd
from PIL import Image
data = pd.read_csv('../input/analog-clocks/analog_clocks/label.csv')
def preprocess(im):
    
    im = im/255
    im -= .5
    return im
channel = 1
im_size = 100
path = '../input/analog-clocks/analog_clocks/images/'
def load_image_batch(ids, batch_size=32):
    
    image_batch = np.zeros((batch_size, im_size, im_size, channel))
    
    label_hour = np.zeros((batch_size, 1))
    label_min = np.zeros((batch_size, 1))
    batch_ids = np.random.choice(ids, batch_size)
    
    ind = 0
    for i in range(len(batch_ids)):
        
        if channel == 1:
            im = Image.open(path + str(batch_ids[i]) + '.jpg').convert('L')
        else:
            im = Image.open(path + str(batch_ids[i]) + '.jpg')
        im = im.resize((im_size,im_size), Image.ANTIALIAS)
        im = np.array(im)
        image_batch[ind] = preprocess(im).reshape((im_size, im_size, channel))
        label_hour[ind] = (data['hour'][data.index==batch_ids[i]])
        label_min[ind] = (data['minute'][data.index==batch_ids[i]])/60
        ind += 1
            
    return (np.array(image_batch), np.array(label_hour), np.array(label_min))
train_ids = np.arange(100)
test_ids = np.arange(20) + len(train_ids)

x_train, y1_train, y2_train = load_image_batch(train_ids, len(train_ids))

x_test, y1_test, y2_test = load_image_batch(test_ids, len(train_ids))