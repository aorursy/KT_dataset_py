import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import gc
import os 
import pandas as pd
pdTrainMap = pd.read_csv("/kaggle/input/thai-mnist-classification/mnist.train.map.csv")        
trainImg = '../input/thai-mnist-classification/train'

plt.figure(figsize=(22, 22))

number = 10
img_list = []
df_map = pdTrainMap
training_dir = "/kaggle/input/thai-mnist-classification/train"
for i in range(number):
    temp = list(df_map[df_map['category'] == i]['id'][:25])
    img_list = img_list + temp

for index, file in enumerate(img_list):
    path = os.path.join(training_dir,file)
    plt.subplot(number,len(img_list)/number,index+1)
    img = mpimg.imread(path)
    plt.axis('off')
    plt.imshow(img)
    
gc.collect()
