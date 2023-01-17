import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Specify the height and width of the image
w, h = 200, 200

# Specify real and imaginary range of image
re_min, re_max = -2.0, 2.0
im_min, im_max = -2.0, 2.0

# Generate evenly spaced values over real and imaginary ranges
real_range = np.arange(re_min, re_max, (re_max - re_min) / w)
imag_range = np.arange(im_max, im_min, (im_min-im_max) / h)
# Generate random real values of complex number
re_randos = []
for i in range(0,100):
	x = random.uniform(-2,2)
	re_randos.append(x)
	
# Generate random imaginary values of complex number	
im_randos = []
for i in range(0,100):
	x = random.uniform(-2,2)
	im_randos.append(x)
# Define a function which generates a Julia set for given tuple x,y
def julia_set(x,y):
	# Pick value of c
	c = complex(x,y)
	
	# Open output file and write PGM header info
	fout = open('julia_'+str(x)+'_'+ str(y)+'.pgm','w')
	fout.write('P2\n# Julia Set image\n' + str(w) + ' ' + str(h) + '\n255\n')
	
	# Generate pixel values and write to file
	for im in imag_range:
		for re in real_range:
			z = complex(re,im)
			n = 255
			while abs(z) < 10 and n >= 5:
				z = z*z + c
				n -= 5
			# Write pixel to file
			fout.write(str(n) + ' ')
		# End of row
		fout.write('\n')
	
	#Close file
	fout.close()
for i in range(0,100):
	a = re_randos[i]
	b = im_randos[i]
		
	julia_set(a,b)