'''

!conda install -y pillow

!conda install -y scikit-learn

!conda install -y scikit-image

!conda install -y tqdm

!conda install -c conda-forge -y gdcm

!pip install dicom-numpy

!pip install pydicom

'''
import os

from functools import partial

from functools import reduce



import numpy as np

import pandas as pd

from tqdm import tqdm

import tensorflow as tf



physical_devices = tf.config.list_physical_devices('GPU')

for gpu_instance in physical_devices:

    tf.config.experimental.set_memory_growth(gpu_instance, True)

    

from skimage.transform import resize

import pydicom
MIN_WEEK = -12

MAX_WEEK = 133

MAX_FVC = 4000

#../input/osic-pulmonary-fibrosis-progression

#INPUT_ROOT = '../input/osic-pulmonary-fibrosis-progression'

INPUT_ROOT = '/kaggle/input/osic-pulmonary-fibrosis-progression'



TRAIN_SCANS_ROOT = os.path.join(INPUT_ROOT, 'train')

TEST_SCANS_ROOT = os.path.join(INPUT_ROOT, 'test')

SCAN_DEPTH = 20

HEIGHT = 32

WIDTH = 32

LEARNING_RATE = 0.001



TRAIN_INPUT_FILE = os.path.join(INPUT_ROOT, 'train.csv')

TEST_INPUT_FILE = os.path.join(INPUT_ROOT, 'test.csv')

BATCH_SIZE = 192

SEQUENCE_LENGTH = 1



STEPS_PER_EPOCH = 1500 // BATCH_SIZE

VALIDATION_STEPS = 300 // BATCH_SIZE



EPOCHS = 50



TEST_OUTPUT = 'submission.csv'
def get_patient_scan(patient_dir, n_depth=5, rows=64, columns=64):

    patient_files = [os.path.join(e) for e in os.listdir(patient_dir)]

    patient_files.sort(key=lambda fname: int(fname.split('.')[0]))

    dcm_slices = [pydicom.read_file(os.path.join(patient_dir, f)) for f in patient_files]

    # Resample slices such that the depth of the CT scan is 'n_depth'

    slice_group = n_depth / len(patient_files)

    slice_indexes = [int(idx / slice_group) for idx in range(n_depth)]

    dcm_slices = [dcm_slices[i] for i in slice_indexes]

    # Merge slices

    shape = (rows, columns)

    shape = (n_depth, *shape)

    img = np.empty(shape, dtype='float32')

    for idx, dcm in enumerate(dcm_slices):

        # Rescale and shift in order to get accurate pixel values

        slope = float(dcm.RescaleSlope)

        intercept = float(dcm.RescaleIntercept)

        resized_img = resize(dcm.pixel_array.astype('float32'), (rows, columns), anti_aliasing=True)

        img[idx, ...] = resized_img * slope + intercept

    return img





def get_dicom_data(patients_root, n_depth=5, rows=64, columns=64):

    def gen(patients_root):

        for patient_dir in os.listdir(patients_root):

            patient_dir = os.path.join(patients_root, patient_dir)

            img = get_patient_scan(patient_dir, n_depth=n_depth, rows=rows, columns=columns)

            yield img



    return tf.data.Dataset.from_generator(partial(gen, patients_root), output_types=tf.float32)

def laplace_log_likelihood(y_true, y_pred):

    uncertainty_clipped = tf.maximum(y_pred[:, 1:2] * 1000.0, 70)

    prediction = y_pred[:, :1]

    delta = tf.minimum(tf.abs(y_true - prediction), 1000.0)

    metric = -np.sqrt(2.0) * delta / uncertainty_clipped - tf.math.log(np.sqrt(2.0) * uncertainty_clipped)

    return tf.reduce_mean(metric)



def laplace_log_likelihood_loss(y_true, y_pred):

    uncertainty_clipped = tf.maximum(y_pred[:, 1:2] * 1000.0, 70)

    prediction = y_pred[:, :1]

    delta = tf.minimum(tf.abs(y_true - prediction), 1000.0)

    metric = -np.sqrt(2.0) * delta / uncertainty_clipped - tf.math.log(np.sqrt(2.0) * uncertainty_clipped)

    return -tf.reduce_mean(metric)
def get3dcnn_model(width, height, depth):

    # Do we have to specify channels ?

    inputs = tf.keras.Input(shape=(width, height, depth, 1))

    x = tf.keras.layers.Conv3D(32, kernel_size=(5, 5, 5), activation='relu')(inputs)

    x = tf.keras.layers.MaxPool3D()(x)



    x = tf.keras.layers.Conv3D(64, kernel_size=(5, 5, 5), activation='relu')(x)

    x = tf.keras.layers.MaxPool3D()(x)



    return inputs, x







def get_combined_model(sequence_length, learning_rate, width, height, depth):

    rnn_inputs = tf.keras.Input(shape=(sequence_length, 2))

    x = tf.keras.layers.Masking(mask_value=-1, input_shape=(sequence_length, 1))(rnn_inputs)

    rnn_out = 4

    rnn_out = tf.keras.layers.GRU(rnn_out)(x)



    cnn3d_inputs, cnn3d_out = get3dcnn_model(width, height, depth)

    cnn3d_out_shape = reduce(lambda x, y: x*y, cnn3d_out.shape[1:])

    cnn3d_out = tf.keras.layers.Reshape((cnn3d_out_shape,))(cnn3d_out)



    combined_out = tf.keras.layers.concatenate([rnn_out, cnn3d_out])



    prediction_output = tf.keras.layers.Dense(1)(combined_out)

    uncertainty_output = tf.keras.layers.Dense(1, activation='sigmoid')(combined_out)



    outputs = tf.keras.layers.concatenate([prediction_output, uncertainty_output])



    model = tf.keras.Model(inputs=[rnn_inputs, cnn3d_inputs], outputs=outputs)



    metrics = [laplace_log_likelihood]



    model.compile(loss=laplace_log_likelihood_loss, optimizer=tf.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)



    return model
def get_combined_data(input_file, batch_size, sequence_length, scans_root, n_depth, rows, columns, max_fvc, split=0.8):

    train_data = pd.read_csv(input_file)

    n_features = 0

    train_data['Weeks'] = train_data['Weeks']

    n_features += 1

    train_data['FVC'] /= max_fvc

    n_features += 1

    grouped = train_data.groupby(train_data.Patient)

    n_data = len(train_data)



    def gen():

        for patient in tqdm(train_data['Patient'].unique()):

            patient_df = grouped.get_group(patient)

            FVC = patient_df['FVC'].iloc[:-1].tolist()

            weeks = patient_df['Weeks']

            Weeks = weeks.iloc[:-1]

            Weeks_next = weeks.iloc[1:]

            week_diff = (np.array(Weeks_next.tolist()) - np.array(Weeks.tolist())) / (MAX_WEEK - MIN_WEEK)

            converted_data = {'FVC': FVC, 'week_diff': week_diff}

            converted_df = pd.DataFrame.from_dict(converted_data)

            indexes = sorted(list(converted_df.index))

            patient_dir = os.path.join(scans_root, patient)

            img = np.expand_dims(get_patient_scan(patient_dir, n_depth=n_depth, rows=rows, columns=columns), axis=-1)



            for idx in indexes:

                prev_indexes = sorted(list(range(int(idx - sequence_length), int(idx))))

                if len(set(indexes).intersection(set(prev_indexes))):

                    sequence = np.empty((sequence_length, n_features))

                    for i, prev_idx in enumerate(prev_indexes):

                        if prev_idx in converted_df['FVC'].index:

                            sequence[i] = [converted_df['FVC'].loc[prev_idx], converted_df['week_diff'].loc[prev_idx]]

                        else:

                            sequence[i] = [-1, -1]

                    yield ((sequence, img), converted_df['FVC'].loc[idx])



    dataset = tf.data.Dataset.from_generator(gen, output_types=((tf.float32, tf.float32), tf.float32)).repeat(None).shuffle(n_data)



    train_size = int(split * n_data)

    train_dataset = dataset.take(train_size)

    val_dataset = dataset.skip(train_size)



    return train_dataset.batch(batch_size), val_dataset.batch(batch_size)

train_dataset, val_dataset = get_combined_data(TRAIN_INPUT_FILE, BATCH_SIZE, SEQUENCE_LENGTH, TRAIN_SCANS_ROOT, SCAN_DEPTH, HEIGHT, WIDTH, max_fvc=MAX_FVC)
model = get_combined_model(SEQUENCE_LENGTH, LEARNING_RATE, WIDTH, HEIGHT, SCAN_DEPTH)
model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS)
def get_input_data(FVC, img, sequence, week_diff):

    converted_data = {'FVC': FVC, 'week_diff': [week_diff]}

    converted_df = pd.DataFrame.from_dict(converted_data)

    sequence[0] = [converted_df['FVC'].loc[0], converted_df['week_diff'].loc[0]]

    data = (np.expand_dims(sequence, axis=0), np.expand_dims(img, axis=0))

    return data
    test_data = pd.read_csv(TEST_INPUT_FILE)

    n_features = 0

    test_data['Weeks'] = test_data['Weeks']

    n_features += 1

    test_data['FVC'] /= MAX_FVC

    n_features += 1

    grouped = test_data.groupby(test_data.Patient)



    prediction_data = {'Patient_Week': [], 'FVC': [], 'Confidence': []}

    all_weeks = set(list(range(MIN_WEEK, MAX_WEEK + 1)))



    for patient in tqdm(test_data['Patient'].unique()):

        patient_df = grouped.get_group(patient)

        FVC = patient_df['FVC'].iloc[:1].tolist()

        measurement_week = patient_df['Weeks'].iloc[0]

        prediction_weeks = sorted(all_weeks - set([measurement_week]))

        week_diffs = (np.array(prediction_weeks) - np.array(measurement_week)) / (MAX_WEEK - MIN_WEEK)

        patient_dir = os.path.join(TEST_SCANS_ROOT, patient)

        img = np.expand_dims(get_patient_scan(patient_dir, n_depth=SCAN_DEPTH, rows=HEIGHT, columns=WIDTH), axis=-1)

        sequence = np.empty((1, n_features))



        prediction_data['Patient_Week'].append(patient + '_' + str(measurement_week))

        prediction_data['FVC'].append(int(FVC[0] * MAX_FVC))

        prediction_data['Confidence'].append(100)



        for week, week_diff in zip(prediction_weeks, week_diffs):

            data = get_input_data(FVC, img, sequence, week_diff)

            prediction = model.predict([data])

            FVC = prediction[0][0]

            uncertainty = prediction[0][1]

            prediction_data['Patient_Week'].append(patient + '_' + str(week))

            prediction_data['FVC'].append(int(FVC * MAX_FVC))

            #confidence = 1/(uncertainty+1) * 100.0

            confidence = uncertainty

            prediction_data['Confidence'].append(confidence)



    indexes = list(range(len(prediction_data['Patient_Week'])))



    def get_key(patient_week):

        patient, week = patient_week.split('_')

        return int(week)



    sorted_data = sorted(zip(indexes, prediction_data['Patient_Week'], prediction_data['FVC'], prediction_data['Confidence']), key=lambda e: get_key(e[1]))

    prediction_data['Patient_Week'] = [e[1] for e in sorted_data]

    prediction_data['FVC'] = [e[2] for e in sorted_data]

    prediction_data['Confidence'] = [e[3] for e in sorted_data]



    df = pd.DataFrame.from_dict(prediction_data)

    df.to_csv(TEST_OUTPUT, index=False)