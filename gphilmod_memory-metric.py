import psutil
def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
import cv2
import os
img_folders = [
    "../input/aptos2019-blindness-detection/train_images",
    "../input/aptos2019-blindness-detection/test_images",
    "../input/fashion-product-images-dataset/fashion-dataset/fashion-dataset/images",
]

image = None
last_mem = psutil.virtual_memory().used
i = 0
for n, folder in enumerate(img_folders):
    print(f"Folder: {folder}")
    (_, _, filelist) = next(os.walk(folder))
    for m, file in enumerate(filelist):
        if i % 10000 == 0:
            #print(f"Files read: {m}/{len(filelist)}    ", end="\r")
            print(f"Files read: {m}/{len(filelist)}    ")
            mem = psutil.virtual_memory().used
            print("Memory used: {} (increased by {}%)".format(get_size(mem), 100*round((mem-last_mem)/last_mem, 2)))
            print("Memory available: {}".format(get_size(psutil.virtual_memory().available)))
            print("Memory free: {}".format(get_size(psutil.virtual_memory().free)))
            last_mem = mem
        filepath = os.path.join(folder, file)
        image = cv2.imread(filepath)
        i += 1
    print(f"Files read: {len(filelist)}/{len(filelist)}    ")
!free -m