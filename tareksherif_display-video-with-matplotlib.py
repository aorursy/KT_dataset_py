!pip install imageio-ffmpeg
import imageio

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from skimage.transform import resize

from IPython.display import HTML

import warnings

warnings.filterwarnings("ignore")



 

def display(driving ):

    fig = plt.figure(figsize=(10, 6))



    ims = []

    for i in range(len(driving)):

        im = plt.imshow(driving[i], animated=True)

        plt.axis('off')

        ims.append([im])



    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)

    plt.close()

    return ani

    

reader = imageio.get_reader('../input/digitsinnoise-video/Test.mp4')

fps = reader.get_meta_data()['fps']

driving_video = []

try:

    for im in reader:

        driving_video.append(im)

except RuntimeError:

    pass

reader.close()



driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]







HTML(display(driving_video).to_html5_video())