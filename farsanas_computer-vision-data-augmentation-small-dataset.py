from IPython.display import YouTubeVideo      
YouTubeVideo('DbfpXtK4mLY')
#Note : In thw video, image upload and saving the outpt done in local desktop
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import matplotlib.pyplot as plt
pred_gen = ImageDataGenerator(
    rotation_range = 80,
    width_shift_range = 0.4,
    height_shift_range = 0.4,
    shear_range = 0.4,
    zoom_range = 0.4,
    horizontal_flip = True,
    fill_mode = 'nearest')
pred_img = load_img(r'../input/predator/predator.jpg')

plt.imshow(pred_img)
plt.show()
pred_img_array = img_to_array(pred_img)
pred_img_array = pred_img_array.reshape((1,) + pred_img_array.shape)
i = 0
for batch in pred_gen.flow(pred_img_array, save_to_dir= './', save_prefix='predator', save_format='jpg'):
    i += 1
    if i >= 14:
        break
