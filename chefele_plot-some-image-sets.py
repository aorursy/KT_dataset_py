import matplotlib.pyplot as plt
import glob, os

def plot_images(path, n_images):
    image_files = list(sorted(glob.glob(path)))[0:n_images]
    for image_file in image_files:
        print(image_file)
        im = plt.imread(image_file)
        plt.figure(figsize=(7,10))
        plt.imshow(im)
        plt.show()

IMAGES_PER_SET = 5
IMAGE_SETS = 10
N_IMAGES = IMAGE_SETS * IMAGES_PER_SET        

plot_images("../input/train_sm/*.jpeg", N_IMAGES)
plot_images("../input/test_sm/*.jpeg",  N_IMAGES)
