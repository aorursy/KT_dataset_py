!git clone https://github.com/drjerk1/oneshot-deepfake

!rm -rf oneshot-deepfake/demos
!cp -r ../input/one-shot-deepfake-weights/* oneshot-deepfake/weights/
%cd oneshot-deepfake
!ls -alh
!ls -alh weights
!mkdir images
!wget 'https://upload.wikimedia.org/wikipedia/commons/0/0f/A._Schwarzenegger.jpg' -O images/arnold.jpg
!wget 'https://upload.wikimedia.org/wikipedia/commons/a/a0/Reuni%C3%A3o_com_o_ator_norte-americano_Keanu_Reeves_%28cropped%29.jpg?download' -O images/reeves.jpg
!python process_image.py --input-image images/arnold.jpg --output-image images/result.jpg --source-image images/reeves.jpg
from matplotlib import pyplot as plt

import imageio

fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(20, 10))

ax[0, 0].imshow(imageio.imread("images/arnold.jpg"))

ax[0, 1].imshow(imageio.imread("images/reeves.jpg"))

ax[0, 2].imshow(imageio.imread("images/result.jpg"))

plt.show()