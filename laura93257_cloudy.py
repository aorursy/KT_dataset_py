# Link for the script https://www.kaggle.com/dimitreoliveira/cloud-images-segmentation-utillity-script

from cloud_images_segmentation_utillity_script import *

from keras.models import load_model



!pip install tta-wrapper --quiet



seed = 0

seed_everything(seed)

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/understanding_cloud_organization/train.csv')

submission = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')



# Preprocecss data

train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])

submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])

test = pd.DataFrame(submission['image'].unique(), columns=['image'])



# Create one column for each mask

train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()

train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']



print('Compete set samples:', len(train_df))

print('Test samples:', len(submission))



display(train.head())
X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=seed)

X_train['set'] = 'train'

X_val['set'] = 'validation'

test['set'] = 'test'



print('Train samples: ', len(X_train))

print('Validation samples: ', len(X_val))
BACKBONE = 'resnet18'

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 3e-4

HEIGHT = 384

WIDTH = 480

CHANNELS = 3

N_CLASSES = 4

ES_PATIENCE = 5

RLROP_PATIENCE = 3

DECAY_DROP = 0.5

model_path = 'uNet_%s_%sx%s.h5' % (BACKBONE, HEIGHT, WIDTH)
preprocessing = sm.get_preprocessing(BACKBONE)



augmentation = None



#augmentation = albu.Compose([albu.HorizontalFlip(p=0.5),

                            # albu.VerticalFlip(p=0.5),

                            # albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1, p=0.5)

                            #])
train_base_path = '../input/understanding_cloud_organization/train_images/'

test_base_path = '../input/understanding_cloud_organization/test_images/'

train_images_dest_path = 'base_dir/train_images/'

validation_images_dest_path = 'base_dir/validation_images/'

test_images_dest_path = 'base_dir/test_images/'



# Making sure directories don't exist

if os.path.exists(train_images_dest_path):

    shutil.rmtree(train_images_dest_path)

if os.path.exists(validation_images_dest_path):

    shutil.rmtree(validation_images_dest_path)

if os.path.exists(test_images_dest_path):

    shutil.rmtree(test_images_dest_path)

    

# Creating train, validation and test directories

os.makedirs(train_images_dest_path)

os.makedirs(validation_images_dest_path)

os.makedirs(test_images_dest_path)



def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):

    '''

    This function needs to be defined here, because it will be called with no arguments, 

    and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)

    '''

    df = df.reset_index()

    for i in range(df.shape[0]):

        item = df.iloc[i]

        image_id = item['image']

        item_set = item['set']

        if item_set == 'train':

            preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)

        if item_set == 'validation':

            preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)

        if item_set == 'test':

            preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)



# Pre-procecss train set

pre_process_set(X_train, preprocess_data)



# Pre-procecss validation set

pre_process_set(X_val, preprocess_data)



# Pre-procecss test set

pre_process_set(test, preprocess_data)
train_generator = DataGenerator(

                  directory=train_images_dest_path,

                  dataframe=X_train,

                  target_df=train,

                  batch_size=BATCH_SIZE,

                  target_size=(HEIGHT, WIDTH),

                  n_channels=CHANNELS,

                  n_classes=N_CLASSES,

                  preprocessing=preprocessing,

                  augmentation=augmentation,

                  seed=seed)



valid_generator = DataGenerator(

                  directory=validation_images_dest_path,

                  dataframe=X_val,

                  target_df=train,

                  batch_size=BATCH_SIZE, 

                  target_size=(HEIGHT, WIDTH),

                  n_channels=CHANNELS,

                  n_classes=N_CLASSES,

                  preprocessing=preprocessing,

                  seed=seed)
from keras import backend as K

from keras.models import Model

from keras.layers import Input, BatchNormalization, Dropout

from keras.layers import Activation

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.losses import binary_crossentropy

from keras.optimizers import Adam, Nadam

from keras.callbacks import Callback, ModelCheckpoint





inputs = Input(shape=(HEIGHT, WIDTH, CHANNELS))



c1 = Conv2D(8, (3, 3), activation=None, padding='same') (inputs)

c1 = Conv2D(8, (3, 3), activation=None, padding='same') (c1)

bn1 = BatchNormalization()(c1)

a1 = Activation('elu')(bn1)

p1 = MaxPooling2D((2, 2), padding='same') (a1)

d1 = Dropout(0.8)(p1)



c2 = Conv2D(16, (3, 3), activation=None, padding='same') (d1)

c2 = Conv2D(16, (3, 3), activation=None, padding='same') (c2)

bn2 = BatchNormalization()(c2)

a2 = Activation('elu')(bn2)

p2 = MaxPooling2D((2, 2), padding='same') (a2)

d2 = Dropout(0.5)(p2)



c3 = Conv2D(32, (3, 3), activation=None, padding='same') (d2)

c3 = Conv2D(32, (3, 3), activation=None, padding='same') (c3)

bn3 = BatchNormalization()(c3)

a3 = Activation('elu')(bn3)

p3 = MaxPooling2D((2, 2), padding='same') (a3)

d3 = Dropout(0.5)(p3)



c4 = Conv2D(64, (3, 3), activation=None, padding='same') (d3)

c4 = Conv2D(64, (3, 3), activation=None, padding='same') (c4)

bn4 = BatchNormalization()(c4)

a4 = Activation('elu')(bn4)

p4 = MaxPooling2D((2, 2), padding='same') (a4)

d4 = Dropout(0.5)(p4)



c5 = Conv2D(64, (3, 3), activation=None, padding='same') (d4)

c5 = Conv2D(64, (3, 3), activation=None, padding='same') (c5)

bn5 = BatchNormalization()(c5)

a5 = Activation('elu')(bn5)

p5 = MaxPooling2D((2, 2), padding='same') (a5)

d5 = Dropout(0.5)(p5)



c55 = Conv2D(128, (3, 3), activation=None, padding='same') (d5)

c55 = Conv2D(128, (3, 3), activation=None, padding='same') (c55)

bn55 = BatchNormalization()(c55)

a55 = Activation('elu')(bn55)

d55 = Dropout(0.5)(a55)



u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (d55)

u6 = concatenate([u6, a5])

d6 = Dropout(0.5)(u6)

c6 = Conv2D(64, (3, 3), activation=None, padding='same') (d6)

c6 = Conv2D(64, (3, 3), activation=None, padding='same') (c6)

bn6 = BatchNormalization()(c6)

a6 = Activation('elu')(bn6)



u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (a6)

u71 = concatenate([u71, a4])

d71 = Dropout(0.5)(u71)

c71 = Conv2D(32, (3, 3), activation=None, padding='same') (d71)

c61 = Conv2D(32, (3, 3), activation=None, padding='same') (c71)

bn61 = BatchNormalization()(c61)

a61 = Activation('elu')(bn61)



u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (a61)

u7 = concatenate([u7, a3])

d7 = Dropout(0.5)(u7)

c7 = Conv2D(32, (3, 3), activation=None, padding='same') (d7)

c7 = Conv2D(32, (3, 3), activation=None, padding='same') (c7)

bn7 = BatchNormalization()(c7)

a7 = Activation('elu')(bn7)



u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (a7)

u8 = concatenate([u8, a2])

d8 = Dropout(0.5)(u8)

c8 = Conv2D(16, (3, 3), activation=None, padding='same') (d8)

c8 = Conv2D(16, (3, 3), activation=None, padding='same') (c8)

bn8 = BatchNormalization()(c8)

a8 = Activation('elu')(bn8)



u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (a8)

u9 = concatenate([u9, a1], axis=3)

d9 = Dropout(0.5)(u9)

c9 = Conv2D(8, (3, 3), activation=None, padding='same') (d9)

c9 = Conv2D(8, (3, 3), activation=None, padding='same') (c9)

bn9 = BatchNormalization()(c9)

a9 = Activation('elu')(bn9)



outputs = Conv2D(4, (1, 1), activation='sigmoid') (a9)





model = Model(inputs=[inputs], outputs=[outputs])
#model = sm.Unet(backbone_name=BACKBONE, 

               # encoder_weights='imagenet',

               # classes=N_CLASSES,

               # activation='sigmoid',

                #input_shape=(HEIGHT, WIDTH, CHANNELS))



checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)

rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)



metric_list = [dice_coef, sm.metrics.iou_score]

callback_list = [checkpoint, es, rlrop]

optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)



model.compile(optimizer=optimizer, loss=sm.losses.bce_dice_loss, metrics=metric_list)

model.summary()
STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE

STEP_SIZE_VALID = len(X_val)//BATCH_SIZE



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              callbacks=callback_list,

                              epochs=EPOCHS,

                              verbose=2).history
plot_metrics(history, metric_list=['loss', 'dice_coef', 'iou_score'])
# Load model trained longer

# model = load_model('../input/cloud-seg-resnet18-trainedlonger/resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':sm.losses.bce_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})
class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']

best_tresholds = [.5, .5, .5, .35]

best_masks = [25000, 20000, 22500, 15000]



for index, name in enumerate(class_names):

    print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))
train_metrics = get_metrics(model, train, X_train, train_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Train')

display(train_metrics)



validation_metrics = get_metrics(model, train, X_val, validation_images_dest_path, best_tresholds, best_masks, seed=seed, preprocessing=preprocessing, set_name='Validation')

display(validation_metrics)
from tta_wrapper import tta_segmentation



model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')
test_df = []



for i in range(0, test.shape[0], 300):

    batch_idx = list(range(i, min(test.shape[0], i + 300)))

    batch_set = test[batch_idx[0]: batch_idx[-1]+1]

    

    test_generator = DataGenerator(

                      directory=test_images_dest_path,

                      dataframe=batch_set,

                      target_df=submission,

                      batch_size=1, 

                      target_size=(HEIGHT, WIDTH),

                      n_channels=CHANNELS,

                      n_classes=N_CLASSES,

                      preprocessing=preprocessing,

                      seed=seed,

                      mode='predict',

                      shuffle=False)

    

    preds = model.predict_generator(test_generator)



    for index, b in enumerate(batch_idx):

        filename = test['image'].iloc[b]

        image_df = submission[submission['image'] == filename].copy()

        pred_masks = preds[index, ].round().astype(int)

        pred_rles = build_rles(pred_masks, reshape=(350, 525))

        image_df['EncodedPixels'] = pred_rles



        ### Post procecssing

        pred_masks_post = preds[index, ].astype('float32') 

        for class_index in range(N_CLASSES):

            pred_mask = pred_masks_post[...,class_index]

            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])

            pred_masks_post[...,class_index] = pred_mask



        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))

        image_df['EncodedPixels_post'] = pred_rles_post

        ###

        

        test_df.append(image_df)



sub_df = pd.concat(test_df)
# Choose 50 samples at random

images_to_inspect = np.random.choice(X_val['image'].unique(), 50, replace=False)

inspect_set = train[train['image'].isin(images_to_inspect)].copy()

inspect_set_temp = []



inspect_generator = DataGenerator(

                    directory=validation_images_dest_path,

                    dataframe=inspect_set,

                    target_df=train,

                    batch_size=1, 

                    target_size=(HEIGHT, WIDTH),

                    n_channels=CHANNELS,

                    n_classes=N_CLASSES,

                    preprocessing=preprocessing,

                    seed=seed,

                    mode='fit',

                    shuffle=False)



preds = model.predict_generator(inspect_generator)



for index, b in enumerate(range(len(preds))):

    filename = inspect_set['image'].iloc[b]

    image_df = inspect_set[inspect_set['image'] == filename].copy()

    pred_masks = preds[index, ].round().astype(int)

    pred_rles = build_rles(pred_masks, reshape=(350, 525))

    image_df['EncodedPixels_pred'] = pred_rles

    

    ### Post procecssing

    pred_masks_post = preds[index, ].astype('float32') 

    for class_index in range(N_CLASSES):

        pred_mask = pred_masks_post[...,class_index]

        pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])

        pred_masks_post[...,class_index] = pred_mask



    pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))

    image_df['EncodedPixels_pred_post'] = pred_rles_post

    ###

    inspect_set_temp.append(image_df)





inspect_set = pd.concat(inspect_set_temp)

inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred')
inspect_predictions(inspect_set, images_to_inspect, validation_images_dest_path, pred_col='EncodedPixels_pred_post')
# Choose 5 samples at random

images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)

inspect_predictions(sub_df, images_to_inspect_test, test_images_dest_path)
def get_score(model, target_df, df, df_images_dest_path, tresholds, min_mask_sizes, N_CLASSES=4, seed=0, preprocessing=None, set_name='Complete set'):

    class_names = ['Fish', 'Flower', 'Gravel', 'Sugar']

    metrics = []



    for class_name in class_names:

        metrics.append([class_name, 0, 0])



    metrics_df = pd.DataFrame(metrics, columns=['Class', 'Dice', 'Dice Post'])

    

    for i in range(0, df.shape[0], 300):

        batch_idx = list(range(i, min(df.shape[0], i + 300)))

        batch_set = df[batch_idx[0]: batch_idx[-1]+1]

        ratio = len(batch_set) / len(df)



        generator = DataGenerator(

                      directory=df_images_dest_path,

                      dataframe=batch_set,

                      target_df=target_df,

                      batch_size=len(batch_set), 

                      target_size=model.input_shape[1:3],

                      n_channels=model.input_shape[3],

                      n_classes=N_CLASSES,

                      preprocessing=preprocessing,

                      seed=seed,

                      mode='fit',

                      shuffle=False)



        x, y = generator.__getitem__(0)

        preds = model.predict(x)

        

        for class_index in range(N_CLASSES):

            class_score = []

            class_score_post = []

            mask_class = y[..., class_index]

            pred_class = preds[..., class_index]

            for index in range(len(batch_idx)):

                sample_mask = mask_class[index, ]

                sample_pred = pred_class[index, ]

                sample_pred_post = post_process(sample_pred, threshold=tresholds[class_index], min_size=min_mask_sizes[class_index])

                if (sample_mask.sum() == 0) & (sample_pred.sum() == 0):

                    dice_score = 1.

                else:

                    dice_score = dice_coefficient(sample_pred, sample_mask)

                if (sample_mask.sum() == 0) & (sample_pred_post.sum() == 0):

                    dice_score_post = 1.

                else:

                    dice_score_post = dice_coefficient(sample_pred_post, sample_mask)

                class_score.append(dice_score)

                class_score_post.append(dice_score_post)

           # metrics_df.loc[metrics_df['Class'] == class_names[class_index], 'Dice'] += np.mean(class_score) * ratio

           # metrics_df.loc[metrics_df['Class'] == class_names[class_index], 'Dice Post'] += np.mean(class_score_post) * ratio



   # metrics_df = metrics_df.append({'Class':set_name, 'Dice':np.mean(metrics_df['Dice'].values), 'Dice Post':np.mean(metrics_df['Dice Post'].values)}, ignore_index=True).set_index('Class')

    

  #  return metrics_df

    return class_score,class_score_post
# Cleaning created directories

if os.path.exists(train_images_dest_path):

    shutil.rmtree(train_images_dest_path)

if os.path.exists(validation_images_dest_path):

    shutil.rmtree(validation_images_dest_path)

if os.path.exists(test_images_dest_path):

    shutil.rmtree(test_images_dest_path)