import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)
! git clone https://github.com/Nelson-Gon/cytounet.git
%cd cytounet
from cytounet.model import *
from cytounet.data import *
from cytounet.augmentation import *
!  ls examples/BBBC003_v1/
data_generator_args =  dict(rotation_range=0.1,
                      rescale = 1./255,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')
! if [ ! -d "aug" ]; then mkdir aug;fi
train_gen = generate_train_data(5, "examples/BBBC003_v1","images", "truth",aug_dict = data_generator_args,
                                 seed = 2, target_size = (512, 512), save_to_dir="aug")
for i, batch in enumerate(train_gen):
    if i>= 5:
        break
! ls aug | wc -l
! if [ ! -d  "aug/images" ]; then mkdir aug/images aug/masks;fi
! mv aug/image_* aug/images && mv aug/mask_* aug/masks && ls aug/masks | wc -l
def load_augmentations(image_path, mask_path, image_prefix="image", mask_prefix="mask"):
    image_name_arr = glob.glob(os.path.join(image_path, "{}*.png".format(image_prefix)))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = image.load_img(item, color_mode="grayscale", target_size = (512, 512))
        img = image.img_to_array(img)
       
        mask = image.load_img(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
                              color_mode="grayscale", target_size = (512, 512))
        mask = image.img_to_array(mask)
       
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr
images, masks = load_augmentations("aug/images","aug/masks")
show_images(images, number=10)
show_images(masks, number = 10)
model = unet(input_size = (512, 512, 1), learning_rate = 1e-4, metrics=["accuracy"],
            loss=["binary_crossentropy"])

history = train(model, train_gen, epochs = 5, steps_per_epoch=150, save_as="unet_embryo.hdf5")
results =  predict(model_object=unet(),test_path="aug/images", model_weights="unet_embryo.hdf5",
                  image_length=15, image_suffix="png")
show_images(results, number = 10)