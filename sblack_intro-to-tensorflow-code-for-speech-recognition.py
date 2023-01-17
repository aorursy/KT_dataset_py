POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()

AUDIO_PATH = '../input/train/audio/'

AUDIO_PATHS = {}

for label in POSSIBLE_LABELS:

    AUDIO_PATHS[label] = AUDIO_PATH + label

print(AUDIO_PATHS)