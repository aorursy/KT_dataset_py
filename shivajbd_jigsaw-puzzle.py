import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
path = '../input/jigsawpuzzle/data/'

data = pd.read_csv(path + 'label.csv')
def solve_jigsaw(im, label, cuts=3, dim=201, channel=3):
    
    '''
        This function rearange the image according to the label
    '''
    
    cut_len = dim//cuts
    
    new_im = np.zeros((dim,dim, channel))
    for i in range(cuts):

        hor_cut = im[i*cut_len:(i+1)*cut_len]

        for j in range(cuts):

            piece = hor_cut[:,j*cut_len:(j+1)*cut_len]      

            pos = label[i*cuts+j]
            x = pos//cuts
            y = pos%cuts

            new_im[x*cut_len:(x+1)*cut_len,y*cut_len:(y+1)*cut_len] = piece

    plt.imshow(new_im)
    plt.show()

id = 17

im = Image.open(path + 'images/' + data.iloc[id]['image'])
im = np.array(im)/255
plt.imshow(im)
plt.show()
label = data.iloc[id]['label']
label = [int(i) for i in label.split()]

solve_jigsaw(im, label)
def load_image():
        
    x = []
    y = []
    
    path = '../input/jigsawpuzzle/data/images/'
    
    for i in range(len(data)):
        
        im = Image.open(path + data.iloc[i]['image'])
        im = np.array(im)
        im = im/255
        x.append(im)
        
        label = data.iloc[i]['label']
        label = [int(i) for i in label.split()]
        y.append(label)
            
    return (np.array(x), np.array(y).reshape((-1,9,1)))
x, y = load_image()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)