import os



# let's take a look at the number of 10-minute long wav files 

preprocessed_fns = os.listdir('../input/preprocessed14d')

len(preprocessed_fns)
# use 50% to add birdcall and remaining ones as no_call

backgrounds = preprocessed_fns[:47]

len(backgrounds)
from pydub import AudioSegment



def normalize(fn):

    '''

    function to read the audio

    and set the sampling rate to 32000

    '''

    audio = AudioSegment.from_file(fn)

    audio = audio.set_channels(1).set_frame_rate(32000)

    return audio
import pandas as pd

import numpy as np



train = pd.read_csv('../input/birdsong-recognition/train.csv')



np.random.seed(0)

msk = np.random.randn(len(train)) < 0.7



# only use 30% of the data for validation

train = train[~msk] # this is validation i forget and call it train

# less than 1 hour bird call clip to make sure we do not run out of memory

train = train[train.duration <= 60]

train.filename = '../input/birdsong-recognition/train_audio/' + train.ebird_code + '/' + train.filename

# make sure we have all the classes

train.ebird_code.value_counts()
def get_random_audio(df=train, length=20000):

    # list of all the classes

    classes = df.ebird_code.unique()

    # shuffle the classes

    np.random.shuffle(classes)

    # select a random class:

    random_class = np.random.choice(classes)

    # list of filenames from this class

    filenames = df[df.ebird_code == random_class].filename.tolist()

    # select a random file:

    fn = np.random.choice(filenames)

    # read the audio file:

    audio = normalize(fn)

    

    if len(audio) <= length:

        return fn, random_class, audio

    else:

        return get_random_audio(df, length)
!mkdir dataset
import numpy as np

class_check = []
# Set the random seed

np.random.seed(18)

for j in range(len(backgrounds)):

    background = normalize('../input/preprocessed14d/'+backgrounds[j])

    # Make background quieter

    background = background - 20

    # set input length of audio to be added

    input_length = 5000

    start = 0

    end = start + input_length

    file_list = []

    class_list = []

    seconds = []

    for i in range(int(len(background)/input_length)):

        # get a random audio and class to which it belongs

        fn, random_class, audio = get_random_audio()

        list_class = random_class

        class_check.append(random_class)

        list_fn = fn

        k = np.random.randint(int(len(background)/input_length))

        segment = int(end/1000)

        if k == i:

            random_class = 'no_call'

            fn = 'preprocessed/'+backgrounds[j]

            audio = background[start: end]

            list_class = random_class

            list_fn = fn

        elif len(audio) > input_length:

            max_offset = len(audio) - input_length

            offset = np.random.randint(max_offset)

            audio = audio[offset:(input_length+offset)]

            background = background.overlay(audio, position=start)

        elif input_length/2 > len(audio):

            background = background.overlay(audio, position=start)

            length = input_length - len(audio)

            fn, random_class, audio_ = get_random_audio(length=length)

            background = background.overlay(audio_, position=start + (input_length/2))

            list_fn += ',' + fn

            list_class += ',' + random_class

        else:

            background = background.overlay(audio, position=start)

        

        start = end

        end = end + input_length

        file_list.append(list_fn)

        class_list.append(list_class)

        seconds.append(segment)

    

    # Export new training example

    file_handle = background.export("dataset/"+backgrounds[j].split('.')[0] + ".wav", format="wav")

    print("File was saved in your directory (dataset) with number of unique classes:", len(np.unique(np.hstack(class_list))))

    

    pre_data = pd.DataFrame({'filename': file_list, 'classes': class_list, 'seconds':seconds})

    pre_data.to_csv("dataset/"+backgrounds[j].split('.')[0] + ".csv", index=False)
# sanity check

len(np.unique(class_check))
new_filenames = []

new_classes = []

new_site = []

new_time = []
# concatenate each dataframe to create a single csv file

for fn in backgrounds:

    fn_df = pd.read_csv('dataset/'+fn.split('.')[0]+'.csv')

    filename = fn

    classes = fn_df.classes.str.cat(sep=' ')

    seconds = fn_df.seconds.astype(str).str.cat(sep=' ')

    new_filenames.append(filename)

    new_classes.append(classes)

    new_time.append(seconds)



new_df = pd.DataFrame({'filename':new_filenames, 'ebird_code':new_classes, 'seconds':new_time})
new_df.head(15)
new_df.to_csv('dataset/val_label.csv', index=None)
import os



# 47 wav files + 47 csv files

len(os.listdir('dataset'))
%%time



import shutil

shutil.make_archive('dataset', 'zip', 'dataset')
!rm -r dataset
import os





def convert_bytes(num):

    """

    this function will convert bytes to MB.... GB... etc

    """

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:

        if num < 1024.0:

            return "%3.1f %s" % (num, x)

        num /= 1024.0





def file_size(file_path):

    """

    this function will return the file size

    """

    if os.path.isfile(file_path):

        file_info = os.stat(file_path)

        return convert_bytes(file_info.st_size)





# Lets check the file size

file_path = r"/kaggle/working/dataset.zip"

print(file_size(file_path))
!ls