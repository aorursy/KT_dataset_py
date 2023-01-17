!pip install tensorflow==2.1.0
!pip install segmentation-models
from kaggle_datasets import KaggleDatasets
GCS_PATH=KaggleDatasets().get_gcs_path("pneudemo")
print(GCS_PATH)

TFIDEN='PneuDTF'  # @param
IMG_DIM=256 
NB_CHANNEL=3 # @param
BATCH_SIZE=128 # @param
BUFFER_SIZE=1024 # @param
TRAIN_DATA=1024*20 # @param
EVAL_DATA=1024*5 # @param


GCS_PATH='{}/{}'.format(GCS_PATH,TFIDEN)
print(GCS_PATH)
import tensorflow as tf
import os 

print(tf.__version__)

def data_input_fn(mode): 
    
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'mask'   : tf.io.FixedLenFeature([],tf.string)
                    
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=NB_CHANNEL)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(IMG_DIM,IMG_DIM,NB_CHANNEL))

        target_raw=parsed_example['mask']
        target=tf.image.decode_png(target_raw,channels=1)
        target=tf.cast(target,tf.float32)/255.0
        target=tf.reshape(target,(IMG_DIM,IMG_DIM,1))
        
        
        return image,target

    gcs_pattern=os.path.join(GCS_PATH,mode,'*.tfrecord')
    file_paths = tf.io.gfile.glob(gcs_pattern)
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

eval_ds = data_input_fn("Eval")
train_ds = data_input_fn("Train")

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

eval_ds = data_input_fn("Eval")

for x,y in eval_ds.take(1):
    data=np.squeeze(x[0])
    plt.imshow(data)
    plt.show()
    print('Image Batch Shape:',x.shape)
    plt.imshow(np.squeeze(y[0]))
    plt.show()
    print('Target Batch Shape:',y.shape)
    print('Target Unique Values:',np.unique(np.squeeze(y[0])))
model_name='efficientnetb7' # @param
iden=str(model_name).upper()
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime;')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
import segmentation_models as sm
sm.set_framework('tf.keras')
with tpu_strategy.scope():
    model = sm.Unet(model_name,input_shape=(IMG_DIM,IMG_DIM,NB_CHANNEL), encoder_weights='imagenet')



model.compile(optimizer="Adam",
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score])
model.summary()
EPOCHS=250 # @param
TOTAL_DATA=TRAIN_DATA+EVAL_DATA
STEPS_PER_EPOCH = TOTAL_DATA//BATCH_SIZE
EVAL_STEPS      = EVAL_DATA//BATCH_SIZE

import os 
WEIGHT_PATH=os.path.join(os.getcwd(),f'{iden}_pneuDemo.h5') # @param
# reduces learning rate on plateau
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,
                               cooldown= 10,
                               patience=10,
                               verbose =1,
                               min_lr=0.1e-5)

mode_autosave = tf.keras.callbacks.ModelCheckpoint(WEIGHT_PATH,
                                                  monitor='val_iou', 
                                                  mode = 'max', 
                                                  save_best_only=True, 
                                                  verbose=1, 
                                                  period =10)

# stop learining as metric on validatopn stop increasing
early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, 
                               verbose=1, 
                               mode = 'auto') 

callbacks = [mode_autosave, lr_reducer,early_stopping ]




history = model.fit(train_ds,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=eval_ds,
                    validation_steps=EVAL_STEPS,
                    callbacks=callbacks)

model.save_weights(WEIGHT_PATH)


def plot_history(history):
    """
    Plots model training history 
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(history.epoch, history.history["iou"], label="Train IoU")
    ax_acc.plot(history.epoch, history.history["val_iou"], label="Validation IoU")
    ax_acc.legend()
    plt.show()

    # show history
plot_history(history)