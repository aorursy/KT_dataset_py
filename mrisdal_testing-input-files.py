from os import listdir

from os.path import isfile, join

from scipy.io import wavfile
path_to_files = "../input/notes"

sound_files = [f for f in listdir(path_to_files) if isfile(join(path_to_files, f))]



print(sound_files)
# testing reading in one of the wav files

sampFreq, X = wavfile.read(path_to_files + "/" + sound_files[1])