import os

import pandas as pd

import numpy as np

import cv2

import json



BASE_SIZE = 256

RAW_INPUT_DIR = '../input/quickdraw-doodle-recognition'
# Common methods for labeling and drawing images.

def draw_cv2(x_strokes, y_strokes, size=256, lw=6):

	img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)

	

	for x_stroke, y_stroke in zip(x_strokes, y_strokes):

		for i in range(len(x_stroke) - 1):

			_ = cv2.line(img, (x_stroke[i], y_stroke[i]), (x_stroke[i + 1], y_stroke[i + 1]), 255, lw)

	

	return img if size == BASE_SIZE else cv2.resize(img, (size, size))





def convert_vector_to_bitmap(image_vector):

	return draw_cv2([stroke[0] for stroke in image_vector], [stroke[1] for stroke in image_vector])





# Draw images from all raw files.

def draw_raw_data(target_dir, nrows=None, min_scale=0.5):

	file_names = os.listdir(target_dir)

	

	image_df_path = os.path.join('image_df.csv')

	pd.DataFrame(columns=['key_id', 'image', 'label']).to_csv(image_df_path, index=False)

	

	last_print_len = 0

	

	for i, fn in enumerate(file_names):

		status_message = 'Progress: {}/{} [Current:{}]'.format(i + 1, len(file_names), fn)

		print(chr(8) * last_print_len + status_message, end='')

		last_print_len = len(status_message)

		

		# print('Processing [{}]'.format(fn))

		df = pd.read_csv(os.path.join(target_dir, fn), nrows=nrows)  # give some limitation bruh :v!

		df['label'] = i

		df = df[['key_id', 'drawing', 'label']]

		df['drawing'] = df['drawing'].apply(json.loads)

		

		# perform standardize

		# TODO use standardize_vector_image()

		# draw image

		# print('Start drawing...')

		df['drawing'] = df['drawing'].apply(convert_vector_to_bitmap)

		

		# append to the integrated drawing DataFrame

		# image_df.append(df[['key_id', 'drawing', 'label']], ignore_index=True, sort=False)

		df.to_csv(image_df_path, mode='a', header=False, index=False)

	# print('{} has been appended.'.format(fn))

	print(chr(8) * last_print_len + 'Done.')

    

    

# Execute

draw_raw_data(os.path.join(RAW_INPUT_DIR, 'train_simplified'), nrows=25000)

raw_dataset = pd.read_csv('image_df.csv')

print(raw_dataset.head(10))