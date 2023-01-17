import os,urllib3
import pandas as pd
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import requests
colors = ['red','green','blue','yellow']
DIR = "../HPAv18/"
v18_url = 'http://v18.proteinatlas.org/images/'
save_dir = '../' #Change the save path
os.listdir('../input')
imgList = pd.read_csv('../input/hpav18/HPAv18RBGY_wodpl.csv')
url_key = []
for i in imgList['Id'][74596:]: #Default download all data, for kernel example, I only download 10 image 
    img = i.split('_')
    for color in colors:
        img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
        img_name = i + "_" + color + ".jpg"
        img_url = v18_url + img_path
        url_key.append((img_name, img_url))

def DownloadImage(key_url):

    (key, url) = key_url
    filename = key
    r = requests.get(url, allow_redirects=True)
    img_save = Image.open(BytesIO(r.content)).resize((512, 512),Image.ANTIALIAS)
    if len(img_save.getbands())> 1:
        red, green, blue = img_save.split()
        if 'red' in filename:
            red.save(save_dir+filename[:-4]+'.png','png')
        if 'blue' in filename:  
            blue.save(save_dir+filename[:-4]+'.png','png')
        if 'green' in filename:
            green.save(save_dir+filename[:-4]+'.png','png')
        if 'yellow' in filename:
            Image.blend(red,green,0.5).save(save_dir+filename[:-4]+'.png','png')
    else:
        img_save.save(save_dir+filename[:-4]+'.png','png')

def Run():

  pool = ThreadPool(processes=100)

  with tqdm(total=len(url_key)) as bar:
    for _ in pool.imap_unordered(DownloadImage, url_key):
      bar.update(1)
if __name__ == '__main__':
  Run()
