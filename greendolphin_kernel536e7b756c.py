import os

from pathlib import Path

import psutil

import gc

import cv2



process = psutil.Process(os.getpid())



img_path = '../input/herbarium-2020-fgvc7/nybg2020/train/images'

images = (Path(path)/fn

          for path, subdirs, fns in os.walk(img_path) if not subdirs

          for fn in fns)



# Session Info: RAM 348.9MB, Max 16GB

# Python kernel: 0.15 GB

print(process.memory_info().rss / 1024 / 1024 / 1024, 'GB')



for i, image in enumerate(images):

    foo = cv2.imread(str(image))

    if i > 50000: break



# Session Info: RAM 3 GB

# Python kernel: 0.17 GB

print(process.memory_info().rss / 1024 / 1024 / 1024, 'GB')



gc.collect()



# Python kernel: 0.17 GB

print(process.memory_info().rss / 1024 / 1024 / 1024, 'GB') 
# As the loop is executed, RAM usage grows continuously, reaching 3 GB after 50 k images read.

# After about 500 k images, the kernel dies as it runs out of memory.

# If the kernel also writes something to disk it may die before with 

#     OSError: [Errno 28] No space left on device

# instead, which is not caused by a full disk but (presumably) some file cashing can't allocate mem.
# Not even ps shows the cause of memory drain:

!ps -Al | sort -rnk 10 | head