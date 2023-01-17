import os

import PIL

import json

import numpy as np

from IPython.display import FileLink, display, clear_output



from PIL import ImageFont, ImageDraw, ImageEnhance



cat_list = ['Hat', 'Trousers', 'Footwear', 'Skirt', 'Shorts', 'Dress', 'Shirt', 'Jacket', 'Backpack']



display_size = (320, 240)

base_folder = '/kaggle'

data_folder = os.path.join( base_folder, 'input', 'til2020-test' )



imgs_folder = os.path.join( data_folder, 'train_v2', '_'.join(cat_list) )



# CHANGE THIS TO THE ONE THAT YOU NEED TO CHECK

part = 'train_v2'

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

# Just keep first 5 for testing of this tool. TODO: remove

master_list = master_list[:5]



keeplooping = True

curr_idx = 0

while keeplooping:

  while curr_idx < len(master_list):

    img_id, payload = master_list[curr_idx]

#     print(img_id)

    anns = payload['anns']

    img_fp = os.path.join( imgs_folder, payload['file_name'] )

    pil_img = PIL.Image.open( img_fp )

    pil_img = pil_img.resize( display_size )

    W,H = display_size



    draw_img = pil_img.copy()

    draw_img = ImageEnhance.Brightness(draw_img).enhance(0.5)

    draw = ImageDraw.Draw(draw_img)



    for ann in anns:

      ln,tn,wn,hn = ann['bbox']

      cat_idx = ann['category_id']

      left = W * ln

      top = H * tn

      width = W * wn

      height = H * hn

      right = left + width

      bot = top + height

      cenx = left + width/2.

      ceny = top + height/2.

      cls_str = cat_list[ cat_idx ]

#       cls_str = cat_list[ cat_id - 1 ]



      draw.rectangle(((left, top), (right, bot)), outline='cyan')

      draw.text((cenx, ceny), cls_str, fill='cyan')

    

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