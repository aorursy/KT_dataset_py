import os

import numpy as np

from matplotlib import pyplot

import librosa

from keras.preprocessing.image import img_to_array, load_img

import IPython.display as ipd

%env JOBLIB_TEMP_FOLDER=/tmp
SONGS_FOLDER_PATH = '../input/gtzan-genre-collection/genres'

OUTPUT_FILE_NAME = 'song_arrays.npz'

SUBDIRECTORIES_LIST = sorted(os.listdir(SONGS_FOLDER_PATH))

AUDIO_SAMPLING_RATE = 16000

ARR_LENGTH = 448000 # 448980 # it's the length of the smallest audio file right now, so we cut all arrays to this length
def _load_song(filepath, sr):

    """

    Load an audio file into a NumPy array.

    """

    array, _ = librosa.load(filepath, sr=sr)

    return array



def load_songs_in_folder(folderpath, sr=AUDIO_SAMPLING_RATE):

    """

    Load all audio files in a folder into a FFT-applied NumPy array of NumPy arrays.

    """

    data_list = [

        _load_song(folderpath+"/"+filename, sr=sr)[:ARR_LENGTH]

        for filename in sorted(os.listdir(folderpath))

    ]

    return np.asarray(data_list)



def get_image(array):

    """

    Get an image file from an array.

    """

    x = librosa.stft(array)

    x = np.stack([np.real(x), np.imag(x)])

    return x.transpose()

 

def rebuild_array(image, original_array_len=ARR_LENGTH):

    """

    Get an array from its audio file.

    """

    image = image.T

    image = image[0] + (1j * image[1])

    array = librosa.istft(image, length=original_array_len)

    return array
arr_dict = {}



# Load all songs in the genre subdirectories

for subdir in SUBDIRECTORIES_LIST:

    arr = load_songs_in_folder(f"{SONGS_FOLDER_PATH}/{subdir}", sr=AUDIO_SAMPLING_RATE)

    arr_dict[subdir] = arr



print("Saving dataset...")

np.savez_compressed(OUTPUT_FILE_NAME, **arr_dict)

print(f"Dataset saved at {OUTPUT_FILE_NAME}")
def load_npz_file(folderpath="."):

    arr_dict = np.load(folderpath+"/"+OUTPUT_FILE_NAME)

    return arr_dict
songs_dict = load_npz_file()

blues_songs = songs_dict["blues"]

for k, v in songs_dict.items():

    print(k, len(v))
arr = _load_song("../input/gtzan-genre-collection/genres/blues/blues.00000.au", sr=AUDIO_SAMPLING_RATE)

image = get_image(arr)

reconstructed_arr = rebuild_array(image, arr.shape[0])



print(

    f"Image shape: {image.shape}\n"

    f"Array shape: {arr.shape}\n"

    f"Reconstructed array shape: {reconstructed_arr.shape}\n"

    f"Mean absolute error: {np.mean(abs(arr - reconstructed_arr))}"

)
# From original array

ipd.Audio(arr, rate=AUDIO_SAMPLING_RATE)
# From reconstructed array

ipd.Audio(reconstructed_arr, rate=AUDIO_SAMPLING_RATE)
arr = load_songs_in_folder("../input/gtzan-genre-collection/genres/blues", sr=AUDIO_SAMPLING_RATE)[0]

image = get_image(arr)

reconstructed_arr = rebuild_array(image, arr.shape[0])



print(

    f"Image shape: {image.shape}\n"

    f"Array shape: {arr.shape}\n"

    f"Reconstructed array shape: {reconstructed_arr.shape}\n"

    f"Mean absolute error: {np.mean(abs(arr - reconstructed_arr))}"

)
# From original array

ipd.Audio(arr, rate=AUDIO_SAMPLING_RATE)
# From reconstructed array

ipd.Audio(reconstructed_arr, rate=AUDIO_SAMPLING_RATE)
arr = blues_songs[0]

image = get_image(arr)

reconstructed_arr = rebuild_array(image, arr.shape[0])



print(

    f"Image shape: {image.shape}\n"

    f"Array shape: {arr.shape}\n"

    f"Reconstructed array shape: {reconstructed_arr.shape}\n"

    f"Mean absolute error: {np.mean(abs(arr - reconstructed_arr))}"

)
# From original array

ipd.Audio(arr, rate=AUDIO_SAMPLING_RATE)
# From reconstructed array

ipd.Audio(reconstructed_arr, rate=AUDIO_SAMPLING_RATE)