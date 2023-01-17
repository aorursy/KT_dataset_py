!ls ../input/
import os

import librosa

import numpy as np

import IPython.display as ipd

import scipy.io.wavfile

from tqdm import tqdm



data_path = '../input/gtzan-dataset-music-genre-classification/Data/genres_original'



def generate_sample_paths():

    samples = dict()

    idx = len(data_path) + 1

    

    for dirname, _, filenames in os.walk(data_path):

        genre = dirname[idx:]

        sample_path = []

        for filename in filenames:

            sample_path.append(os.path.join(dirname, filename))

        samples[genre] = np.array(sorted(sample_path))



    del(samples[''])

    return samples
sample_paths = generate_sample_paths()

sample, sr = librosa.load(sample_paths['jazz'][42])

print(sample.shape)

print(sample)
sr = 220

sample, sr = librosa.load(sample_paths['jazz'][42], sr=sr)

print('Sample rate:', sr, '| Array shape:', sample.shape[0])



sr = 2205

sample, sr = librosa.load(sample_paths['jazz'][42], sr=sr)

print('Sample rate:', sr, '| Array shape:', sample.shape[0])
sample, sr = librosa.load(sample_paths['jazz'][42])

print('Shape:', sample.shape[0])

sample, sr = librosa.load(sample_paths['country'][1])

print('Shape:', sample.shape[0])

sample, sr = librosa.load(sample_paths['metal'][0])

print('Shape:', sample.shape[0])
def play(path=None, sample=None, sr=22050, **kwargs):

    if path != None:

        sample, sr = librosa.load(path, **kwargs)

    return ipd.Audio(sample, rate=sr)



play(sample_paths['disco'][7])
!wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl

!chmod a+rx /usr/local/bin/youtube-dl



!youtube-dl -qx --audio-format wav https://www.youtube.com/watch?v=d6rxGmvQPLU -o 'ochansensusu.%(ext)s'
play('ochansensusu.wav', offset=12, duration=30)
links = [

         'https://www.youtube.com/watch?v=P_B_GalsJrE',

         'https://www.youtube.com/watch?v=lln2NPx3aKw',

         'https://www.youtube.com/watch?v=fxsCwzsMOfU',

         'https://www.youtube.com/watch?v=RJ1YBbUKzvw',

         'https://www.youtube.com/watch?v=Tqpj9gmk8UI',

         'https://www.youtube.com/watch?v=PVn6gY1Jc7I',

         'https://www.youtube.com/watch?v=d6rxGmvQPLU',

         'https://www.youtube.com/watch?v=hvudfoL1EWU',

         'https://www.youtube.com/watch?v=-rZWdolJfgk',

         'https://www.youtube.com/watch?v=yU38oLPNpYk',

         'https://www.youtube.com/watch?v=C7NXYSklMbg',

         'https://www.youtube.com/watch?v=axV7NhKArV0',

         'https://www.youtube.com/watch?v=TZjTXh_zaXc',

         'https://www.youtube.com/watch?v=9oboWLb4I1Y',

         'https://www.youtube.com/watch?v=Bwq2M5T4dQo',

         'https://www.youtube.com/watch?v=Q77gbjfsVO8',

         'https://www.youtube.com/watch?v=3UjW3-0MsSI',

         'https://www.youtube.com/watch?v=iYrUwWq6KO8',

         'https://www.youtube.com/watch?v=8LIqn2FGYsg',

         'https://www.youtube.com/watch?v=F840uydN-Ps',

         'https://www.youtube.com/watch?v=18HPVYj_HnY',

         'https://www.youtube.com/watch?v=hSaiW1lJRhU',

         'https://www.youtube.com/watch?v=HGzrJjHwmBQ',

         'https://www.youtube.com/watch?v=0XWzY5SLTss',

         'https://www.youtube.com/watch?v=XWqua6rsEmw',

         'https://www.youtube.com/watch?v=JU-8Ikw5HL0',

         'https://www.youtube.com/watch?v=FLL8WPho6BI',

         'https://www.youtube.com/watch?v=iP-6wexQ0V0',

         'https://www.youtube.com/watch?v=uuMrQ6NMP0A',

         'https://www.youtube.com/watch?v=16Ep_2bLq0g',

         'https://www.youtube.com/watch?v=6qsXxKawUos',

         'https://www.youtube.com/watch?v=6Ey3jFf6vhA',

         'https://www.youtube.com/watch?v=RXGwVJCdV6A',

         'https://www.youtube.com/watch?v=rvNOFp6xFMc',

         'https://www.youtube.com/watch?v=FTxSXUzc96A',

         'https://www.youtube.com/watch?v=t0lbJRmlXW8',

         'https://www.youtube.com/watch?v=bKwPynQ7MLU',

         'https://www.youtube.com/watch?v=k-F1k6Nwlhk',

         'https://www.youtube.com/watch?v=Ohq_fzWyXfw',

         'https://www.youtube.com/watch?v=WISgltgMMR8'

]



print('Total math rock songs:', len(links))
def generate_sh(links, sh_name='download_list.sh', target_dir='./'):

    if not os.path.exists(target_dir):

        os.mkdir(target_dir)



    with open(os.path.join(sh_name), 'w') as f:

        for i in range(len(links)):

            link = links[i]

            cmd = "youtube-dl -qx --audio-format wav %s -o " % link

            cmd = cmd + ("'%s%03d." % (target_dir, i)) + "%(title)s.%(ext)s'\n"

            f.write(cmd)



generate_sh(links, target_dir='./downloads/')
!head -n 3 download_list.sh
!chmod +x download_list.sh

!./download_list.sh
!ls downloads
play('downloads/020.CHON - Knot - Audiotree Live.wav', duration=10)
def get_optimum_gap(chunk_length, audio_length):

    '''

    `chunk_length` and `audio_length` in ms.

    return gap length (in ms) which use maximum number of audio part.

    '''

    x = np.arange(chunk_length // 2)

    c = np.floor((audio_length - x) // (chunk_length - x))

    x = chunk_length + (c - 1) * (chunk_length - x)

    return np.argmax(x)



def partition(audio_path, chunk_length, sr=22050, min_amp=0.03, stats=False):

    '''

    Arguments:

        - audio_path : str, path to audio

        - chunk_length : int, length of each chunk in ms

        - sr : int, sample rate

        - min_amp : float, minimum amplitudes to start and finish and audio

        - stats : bool, whether return chunks only or also with the stats

    Returns:

        - If stats=False, a list of Numpy array (such result of `librosa.load`)

        - Else, return list also with gap duration of audio unused in ms, and the

          percentage of duration of audio used

    '''

    audio, _ = librosa.load(audio_path, sr=sr)

    s_time = np.argmax(audio > min_amp)

    e_time = len(audio) - np.argmax(audio[::-1] > min_amp)

    audio = audio[s_time:e_time]

    

    audio_length = 1000 * len(audio) // sr

    gap_length = get_optimum_gap(chunk_length, audio_length)

    num_chunks = (audio_length - gap_length) // (chunk_length - gap_length)

    

    counter, sample_length = 0, chunk_length * sr // 1000

    start, step = 0, (chunk_length - gap_length) * sr // 1000

    parts = []

    while counter < num_chunks:

        parts.append(audio[start : start + sample_length])

        start += step

        counter += 1



    total_length = chunk_length + (num_chunks - 1) * (chunk_length - gap_length)

    total_pct = total_length / audio_length

    loss = audio_length - total_length



    if stats:

        return parts, loss, total_pct

    return parts



def mass_partition(audio_path, save_path, chunk_length=30000, sr=22050, \

                   start_index=0, suffix='', **kwargs):

    if not os.path.exists(save_path):

        os.mkdir(save_path)

    filenames = os.listdir(audio_path)

    audio_path = sorted([os.path.join(audio_path, x) for x in filenames])



    file_id = start_index

    for path in tqdm(audio_path):

        audios = partition(path, chunk_length, **kwargs)

        chunk_id = 0

        for audio in audios:

            file_path = '%s.%03d%02d.wav' % (suffix, file_id, chunk_id)

            file_path = os.path.join(save_path, file_path)

            scipy.io.wavfile.write(file_path, sr, audio)

            chunk_id += 1

        file_id += 1
audios, loss, pct = partition('downloads/001.tricot「 爆裂パニエさん」（大反射祭Tour／2019.04.28 at '

                              'TSUTAYA O-EAST）YouTube Ver..wav', 30000, stats=True)

    

print(loss, pct, len(audios))
mass_partition(audio_path='downloads', \

               save_path='./math/', 

               chunk_length=30000, \

               sr=22050, \

               start_index=0, \

               suffix='math')
result = sorted(os.listdir('./math/'))

result = '\n'.join(result[:3] + result[-3:])

print(result)

play('./math/math.00105.wav')
!rm -rf math

!rm -rf downloads

!rm -rf ochansensusu.wav