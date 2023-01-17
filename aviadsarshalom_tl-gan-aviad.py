""" change working directory """

import os



if os.path.basename(os.getcwd()) == 'working':

    os.chdir('../input/tf-gan-code-20181007/transparent_latent_gan_kaggle_2018_1007/transparent_latent_gan_kaggle_2018_1007')

print('current working directory is {}'.format(os.getcwd()))

""" import packages """



import os

import glob

import sys

import numpy as np

from numpy.random import normal as nl

import pickle

import tensorflow as tf

from PIL import Image

import matplotlib.pyplot as plt

import PIL

import ipywidgets

import io



import src.tf_gan.generate_image as generate_image

import src.tf_gan.feature_axis as feature_axis

import src.tf_gan.feature_celeba_organize as feature_celeba_organize





""" load learnt feature axis directions """

path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'



pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]



with open(pathfile_feature_direction, 'rb') as f:

    feature_direction_name = pickle.load(f)



print(feature_direction_name)

feature_direction = feature_direction_name['direction']

feature_name = feature_direction_name['name']

num_feature = feature_direction.shape[1]



feature_name = feature_celeba_organize.feature_name_celeba_rename

feature_direction = feature_direction_name['direction']* feature_celeba_organize.feature_reverse[None, :]



""" ========== start tf session and load GAN model ========== """



# path to model code and weight

path_pg_gan_code = './src/model/pggan'

path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'

sys.path.append(path_pg_gan_code)





""" create tf session """

yn_CPU_only = False



if yn_CPU_only:

    config = tf.ConfigProto(device_count = {'GPU': 0}, allow_soft_placement=True)

else:

    config = tf.ConfigProto(allow_soft_placement=True)

    config.gpu_options.allow_growth = True



sess = tf.InteractiveSession(config=config)



try:

    with open(path_model, 'rb') as file:

        G, D, Gs = pickle.load(file)

except FileNotFoundError:

    print('before running the code, download pre-trained model to project_root/asset_model/')

    raise



len_z = Gs.input_shapes[0][1]

z_sample = np.random.randn(len_z)

x_sample = generate_image.gen_single_img(z_sample, Gs=Gs)
""" ========== ipywigets gui interface ========== """



def img_to_bytes(x_sample):

    """ tool funcion to code image for using ipywidgets.widgets.Image plotting function """

    imgObj = PIL.Image.fromarray(x_sample)

    imgByteArr = io.BytesIO()

    imgObj.save(imgByteArr, format='PNG')

    imgBytes = imgByteArr.getvalue()

    return imgBytes



# a random sample of latent space noise

z_sample = np.random.randn(len_z)

# the generated image using this noise patter

x_sample = generate_image.gen_single_img(z=z_sample, Gs=Gs)



w_img = ipywidgets.widgets.Image(value=img_to_bytes(x_sample), fromat='png', 

                                 width=512, height=512,

                                 layout=ipywidgets.Layout(height='512px', width='512px')

                                )



image_for_display = x_sample





# sl = 0.1 # sl = scale = standard deviation

# randomized_feature_values = [nl(Shadow, sl), nl(Arched_Eyebrows, sl), nl(Attractive, sl), nl(Eye_bags, sl), nl(Bald, sl), nl(Bangs, sl), nl(Big_Lips, sl),

#              nl(Big_Nose, sl), nl(Black_Hair, sl), nl(Blond_Hair, sl), nl(Blurry, sl), nl(Brown_Hair, sl), nl(Bushy_Eyebrows, sl), nl(Chubby, sl),

#              nl(Double_Chin, sl), nl(Eyeglasses, sl), nl(Goatee, sl), nl(Gray_Hair, sl), nl(Makeup, sl), nl(High_Cheekbones, sl), nl(Male, sl),

#              nl(Mouth_Open, sl), nl(Mustache, sl), nl(Narrow_Eyes, sl), nl(Beard, sl), nl(Oval_Face, sl), nl(Skin_Tone, sl), nl(Pointy_Nose, sl),

#              nl(Hairline, sl), nl(Rosy_Cheeks, sl), nl(Sideburns, sl), nl(Smiling, sl), nl(Straight_Hair, sl), nl(Wavy_Hair, sl), nl(Earrings, sl),

#              nl(Hat, sl), nl(Lipstick, sl), nl(Necklace, sl), nl(Necktie, sl), nl(Age, sl)]



# required_feature_values = [Shadow, Arched_Eyebrows, Attractive, Eye_bags, Bald, Bangs, Big_Lips,

#              Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby,

#              Double_Chin, Eyeglasses, Goatee, Gray_Hair, Makeup, High_Cheekbones, Male,

#              Mouth_Open, Mustache, Narrow_Eyes, Beard, Oval_Face, Skin_Tone, Pointy_Nose,

#              Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Earrings,

#              Hat, Lipstick, Necklace, Necktie, Age]





def scalar_projection(vector, vector_base):

    """

    returns the scalar projection of vector on vector_base

    """

    return np.dot(vector, vector_base) / np.dot(vector_base, vector_base)





class ImageManipulator(object):

    """ call back functions for button click behaviour """

    counter = 0

    

    def __init__(self):

        self.latents = z_sample

        self.image = generate_image.gen_single_img(z=self.latents, Gs=Gs)

        self.feature_direction = feature_direction

        self.feature_lock_status = np.zeros(num_feature).astype('bool')       

        self.feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(

            self.feature_direction, idx_base=np.flatnonzero(self.feature_lock_status))



    def random_gen(self):

        self.latents = np.random.randn(len_z)        

        self.update_img()

        

    def set_single_feature(self, idx_feature, required_feature_value):

        magnitude = scalar_projection(self.latents, self.feature_directoion_disentangled[:, idx_feature])        

        self.latents += self.feature_directoion_disentangled[:, idx_feature] * (required_feature_value - magnitude)        



    def set_features(self):

        

#         uncomment the features you want to set



#         self.set_single_feature(0, 1) # Shadow

#         self.set_single_feature(1, 1) # Arched_Eyebrows

#         self.set_single_feature(2, 1) # Attractive

#         self.set_single_feature(3, 1) # Eye_bags

#         self.set_single_feature(4, 1) # Bald

#         self.set_single_feature(5, 1) # Bangs

#         self.set_single_feature(6, 1) # Big_Lips

#         self.set_single_feature(7, 1) # Big_Nose

#         self.set_single_feature(8, 1) # Black_Hair

#         self.set_single_feature(9, 1) # Blond_Hair

#         self.set_single_feature(10, 1) # Blurry

#         self.set_single_feature(11, 1) # Brown_Hair

#         self.set_single_feature(12, 1) # Bushy_Eyebrows

#         self.set_single_feature(13, 3) # Chubby

#         self.set_single_feature(14, 1) # Double_Chin

#         self.set_single_feature(15, 1) # Eyeglasses

#         self.set_single_feature(16, 1) # Goatee

#         self.set_single_feature(17, 1) # Gray_Hair

#         self.set_single_feature(18, 1) # Makeup

#         self.set_single_feature(19, 1) # High_Cheekbones

#         self.set_single_feature(20, 1) # Male

#         self.set_single_feature(21, 1) # Mouth_Open

#         self.set_single_feature(22, 1) # Mustache

#         self.set_single_feature(23, 1) # Narrow_Eyes

#         self.set_single_feature(24, 1) # Beard

#         self.set_single_feature(25, 1) # Oval_Face

#         self.set_single_feature(26, 1) # Skin_Tone

#         self.set_single_feature(27, 1) # Pointy_Nose

#         self.set_single_feature(28, 1) # Hairline

#         self.set_single_feature(29, 1) # Rosy_Cheeks

#         self.set_single_feature(30, 1) # Sideburns

#         self.set_single_feature(31, 1) # Smiling

#         self.set_single_feature(32, 1) # Straight_Hair

#         self.set_single_feature(33, 1) # Wavy_Hair

#         self.set_single_feature(34, 1) # Earrings

#         self.set_single_feature(35, 1) # Hat

#         self.set_single_feature(36, 1) # Lipstick

#         self.set_single_feature(37, 1) # Necklace

#         self.set_single_feature(38, 1) # Necktie

#         self.set_single_feature(39, 1) # Age        

           

        self.update_img()



    def set_feature_lock(self, idx_feature, set_to=None):

        if set_to is None:

            self.feature_lock_status[idx_feature] = np.logical_not(self.feature_lock_status[idx_feature])

        else:

            self.feature_lock_status[idx_feature] = set_to        

        self.feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(

            self.feature_direction, idx_base=np.flatnonzero(self.feature_lock_status))

    

    def update_img(self):        

        self.image = generate_image.gen_single_img(z=self.latents, Gs=Gs)        

    

    def gen_save_and_show(self, image_idx=0):

        self.random_gen()

        self.set_features()

        image_for_save = Image.fromarray(self.image)        

        image_for_save.save('/kaggle/working/image' + str(image_idx) + '.png')

        plt.imshow(imageManipulator.image)

        plt.show()

        

    def gen_save_and_show_20_times(self):

        for image_idx in range(20):

            self.gen_save_and_show(image_idx)

            



imageManipulator = ImageManipulator()

imageManipulator.gen_save_and_show_20_times()


