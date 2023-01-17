%mkdir /kaggle/temp
%cd /kaggle/temp
!git clone https://github.com/ultralytics/yolov3  # clone
%cd yolov3
#!pip install -U -r requirements.txt

import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
from distutils.dir_util import copy_tree
import os

temp_input = '/kaggle/temp/input/'

image_path = os.path.join(temp_input, 'images')
copy_tree('/kaggle/input/synthetic-gloomhaven-monsters/images', image_path)

label_path = os.path.join(temp_input, 'labels')
copy_tree('/kaggle/input/synthetic-gloomhaven-monsters/labels', label_path)
label_path = os.path.join(temp_input, 'labels')
for dirname, _, filenames in os.walk(label_path):    
    if len(filenames) == 0:
        continue
        
    print('Processing', dirname, 'containing', len(filenames), 'labels')
    
    lines = []
    for filename in filenames:
        lines.append(os.path.join(dirname.replace("labels", "images"), filename.replace(".txt", ".png")))
    
    result_file = '/kaggle/working/{}_images.txt'.format(dirname.split("/")[-1])
    with open(result_file, 'w') as file:
        file.write('\n'.join(lines))
        
    print(result_file)
classes = ['ancient_artillery', 'bandit_archer', 'bandit_guard', 'black_imp', 'cave_bear', 'city_archer', 'city_guard',
           'cultist', 'deep_terror', 'earth_demon', 'flame_demon', 'forest_imp', 'frost_demon', 'giant_viper',
           'harrower_infester', 'hound', 'inox_archer', 'inox_guard', 'inox_shaman', 'living_bones', 'living_corpse',
           'living_spirit', 'lurker', 'night_demon', 'ooze', 'savvas_icestorm', 'savvas_lavaflow', 'spitting_drake',
           'stone_golem', 'sun_demon', 'vermling_scout', 'vermling_shaman', 'vicious_drake', 'wind_demon']

names_file = '/kaggle/working/monsters.names'
with open(names_file, 'w') as file:
    file.write('\n'.join(classes))
data_file = '/kaggle/working/monsters.data'
with open(data_file, 'w') as file:
    file.write("classes={}\n".format(len(classes)))
    file.write("train={}\n".format('/kaggle/working/train_images.txt'))
    file.write("valid={}\n".format('/kaggle/working/val_images.txt'))
    file.write("names={}".format(names_file))
nb_filters = (5 + len(classes)) * 3

config_file = 'cfg/yolov3-spp.cfg'

data = []
with open(config_file, 'r') as f:
    # read a list of lines into data
    data = f.readlines()

for idx, line in enumerate(data):
    if line == "filters=255\n":
        data[idx] = "filters={}\n".format(nb_filters)
    elif line == "classes=80\n":
        data[idx] = "classes={}\n".format(len(classes))
    else:
        continue
        
    print('Replaced line {} from "{}" to "{}"'.format(idx + 1, line, data[idx]).replace('\n', ''))

with open(config_file, 'w') as f:
    f.writelines(data)
!python3 train.py --epochs 10 --batch-size 16 --cfg yolov3-spp.cfg --data /kaggle/working/monsters.data --nosave --weights yolov3-spp-ultralytics.pt