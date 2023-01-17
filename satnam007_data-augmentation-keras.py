import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
img = load_img('../input/face-expression/IMG_20200628_111907_2.jpg')  # this is a PIL image
print(img.size)


plt.imshow(img)
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
img.size
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='../input/output/', save_prefix='satnam', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project='data-augmentation-not-in-kaggle')