import os

import PIL

import json

import numpy as np

from IPython.display import FileLink, display, clear_output



from PIL import ImageFont, ImageDraw, ImageEnhance



cat_list = ['bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'outerwear', 'pants', 'skirts', 'tops']



display_size = (480, 360)

base_folder = '/kaggle'

data_folder = os.path.join( base_folder, 'input', 'til2020-test' )



# Note: check if train or val first!

imgs_folder = os.path.join( data_folder, 'val', 'val' )



# CHANGE THIS TO THE ONE THAT YOU NEED TO CHECK

part = 'val-11'

json_annotations = os.path.join( data_folder, '{}.json'.format(part) )
with open(json_annotations, 'r') as f:

  gt_json = json.load(f)
master = {}

imgs = gt_json['images']

anns = gt_json['annotations']



for ann_dict in anns:

  img_id = ann_dict['image_id']

  if img_id not in master:

    master[img_id] = {'anns':[]}

  master[img_id]['anns'].append(ann_dict)

for img_dict in imgs:

  img_id = img_dict['id']

  if img_id in master:

    master[img_id]['file_name'] = img_dict['file_name']

# Now master looks like this:

# img_id -> {'file_name': img-filepath, 'anns': [ann1, ann2, ...]}

# We can iterate over it to check
good_img_ids = {}

master_list = list( master.items() )



keeplooping = True

curr_idx = 0

while keeplooping:

  while curr_idx < len(master_list):

    img_id, payload = master_list[curr_idx]

    anns = payload['anns']

    img_fp = os.path.join( imgs_folder, payload['file_name'] )

    pil_img = PIL.Image.open( img_fp )

    W,H = pil_img.size

    pil_img = pil_img.resize( display_size )



    draw_img = pil_img.copy()

    draw_img = ImageEnhance.Brightness(draw_img).enhance(0.5)

    draw = ImageDraw.Draw(draw_img)



    for ann in anns:

      left,top,width,height = ann['bbox']

      disp_left = (left/W) * display_size[0]

      disp_top = (top/H) * display_size[1]

      disp_width = (width/W) * display_size[0]

      disp_height = (height/H) * display_size[1]

    

      cat_id = ann['category_id']

      disp_right = disp_left + disp_width

      disp_bot = disp_top + disp_height

      disp_cenx = disp_left + disp_width/2.

      disp_ceny = disp_top + disp_height/2.



      cls_str = cat_list[ cat_id - 1 ]



      draw.rectangle(((disp_left, disp_top), (disp_right, disp_bot)), outline='cyan')

      draw.text((disp_cenx, disp_ceny), cls_str, fill='cyan')

    

    display(draw_img)

    

    print('"a" to accept, "b" to go back, any other key to reject.')



    key = input()

    if key == 'a':

      print('accepting img-id = {}'.format(img_id))

      good_img_ids[img_id] = True

      curr_idx += 1

    elif key == 'b':

      print('going back')

      curr_idx = (curr_idx - 1 + len(master_list)) % len(master_list)

    else:

      print('rejecting.. moving on.')

      if img_id in good_img_ids:

        del good_img_ids[img_id]

      curr_idx += 1

    clear_output(wait=True)

  print('All images inspected! Going to exit. Press "b" to go back')

  key = input()

  if key.lower() != 'b':

    keeplooping = False

  else:

    curr_idx = (curr_idx - 1 + len(master_list)) % len(master_list)
good_img_ids = list(good_img_ids.keys())

target_json = '{}-filtered.json'.format(part)

with open(target_json, 'w') as f:

  json.dump(good_img_ids, f)
os.chdir(r'/kaggle/working')

FileLink(r'{}'.format(target_json))