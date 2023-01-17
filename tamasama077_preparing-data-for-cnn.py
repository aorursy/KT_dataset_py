import os

import cv2

import numpy as np

import pickle

from tqdm import tqdm

from random import shuffle
DIR = '../input/asl_alphabet_train/asl_alphabet_train'

def storing_data(pic):

  data = []

  for i in range(len(pic)):

    for j in tqdm(os.listdir(f'{DIR}/{pic[i]}/')):

      img = cv2.imread(f'{DIR}/{pic[i]}/{j}')

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



      data.append([gray, i])

  return data
S_PIC = ['A', 'B', 'C']

B_PIC = list(chr(x) for x in range(65, 91))



s_data = storing_data(S_PIC)

b_data = storing_data(B_PIC)
shuffle(s_data)

shuffle(b_data)



s_x = np.array(list(x[0] for x in s_data)).reshape(-1, 200, 200, 1)

s_y = list(x[1] for x in s_data)



b_x = np.array(list(x[0] for x in b_data)).reshape(-1, 200, 200, 1)

b_y = list(x[1] for x in b_data)
sx_out = open('s_x.pickle', 'wb')

pickle.dump(s_x, sx_out)

sx_out.close()



sy_out = open('s_y.pickle', 'wb')

pickle.dump(s_y, sy_out)

sy_out.close()



bx_out = open('b_x.pickle', 'wb')

pickle.dump(b_x, bx_out)

bx_out.close()



by_out = open('b_y.pickle', 'wb')

pickle.dump(b_y, by_out)

by_out.close()
! ls -l