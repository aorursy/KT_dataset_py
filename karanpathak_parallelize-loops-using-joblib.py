!pip install joblib

!pip install Pillow
from joblib import Parallel, delayed
from PIL import ImageDraw, Image

import numpy as np

from pathlib import Path

from time import sleep, time

from multiprocessing import cpu_count
size_w = size_h = 512
def draw_rectangles(img_index, save_dir, n, m):

    image = Image.new(mode = 'RGB', size = (n, m), color = (255, 255, 255))

    draw = ImageDraw.Draw(image)

    sleep(3.0)

    x1 = np.random.randint(low=0, high=n//2)

    x2 = np.random.randint(low=n//2 + 1, high=n)

    

    y1 = np.random.randint(low=0, high=m//2)

    y2 = np.random.randint(low=m//2 + 1, high=m)

    

    draw.rectangle(xy=[(x1,y1), (x2,y2)], outline=(255, 0, 0))

    image_name = img_index + '.png'

    image.save(save_dir.joinpath(image_name).as_posix())

    return image_name
save_dir_no_parallel_process = Path('./no_parallel_process')

save_dir_no_parallel_process.mkdir(parents=True, exist_ok=True)
start_time = time()





for image_index in range(10):

    image_name = draw_rectangles(img_index=str(image_index+1), save_dir=save_dir_no_parallel_process, n=size_w, m=size_h)

    print("Image Name: ", image_name)





sequential_execution_time = time() - start_time





print("Execution Time: ", sequential_execution_time)
save_dir_parallel_process = Path('./parallel_process')

save_dir_parallel_process.mkdir(parents=True, exist_ok=True)
start_time = time()





print("Number of jobs: ",int(cpu_count()))



# Use multiple CPUs (Multi Processing)

image_filenames = Parallel(n_jobs=int(cpu_count()), prefer='processes')(

    delayed(draw_rectangles)(img_index=str(image_index+1), save_dir=save_dir_parallel_process, n=size_w, m=size_h) 

    for image_index in range(10)

)



parallel_execution_time = time() - start_time





print("Execution Time: ", parallel_execution_time)
for img_index in image_filenames:

    print(img_index)
save_dir_parallel_threads = Path('./parallel_threads')

save_dir_parallel_threads.mkdir(parents=True, exist_ok=True)
start_time = time()





print("Number of threads: ",10)



# Use multiple CPUs (Multi Processing)

image_filenames = Parallel(prefer='threads', n_jobs=10)(

    delayed(draw_rectangles)(img_index=str(image_index+1), save_dir=save_dir_parallel_threads, n=size_w, m=size_h) 

    for image_index in range(10)

)



parallel_execution_time_threading = time() - start_time





print("Execution Time: ", parallel_execution_time_threading)
shared_list = []
def add_to_list(x):

    sleep(3.0)

    shared_list.append(x)
start_time = time()



result = Parallel(n_jobs=cpu_count(), require='sharedmem')(delayed(add_to_list)(i) for i in range(10))



print("Execution Time: ", time()-start_time)
shared_list