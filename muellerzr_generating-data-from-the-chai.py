import pandas as pd

import re
f = open('../input/chai-time-data-science/Raw Subtitles/E75.txt')
f.readline()
f.readline()
f.readline()
def extract_transcript(fn, save=False, save_path=''):

    "Takes transcript and converts it to `DataFrame`"

    pat = r'([A-Za-z]|\s+)\s([0-9]{0,2}:{0,1}[0-9]{1,2}:[0-9][0-9])'

    f = open(fn, "r")

    t = True

    df = pd.DataFrame(columns = ['Time', 'Speaker', 'Text'])

    i = 0

    first = True

    while t:

        line = f.readline()

        if line == '': t = False

        i += 1

        line = re.split(pat, line[:-1])

        if len(line) == 4:

            is_new = 1

            speak = line[0]

            time = line[2]

        while is_new == 1:

            if first:

                line = f.readline()

                for i in range(6):

                    l_c = f.readline()

                    if speak not in l_c and time not in l_c:

                        line += l_c

                i += 1

                first = False

            else:

                line = f.readline()

                i += 1

            if len(line) > 2 and line != '\n':

                line = line[:-1]

                df.loc[i] = [time, speak, line]

                df.reset_index()

            else:

                is_new = 0

    df.reset_index(drop=True, inplace=True)

    df['Text'] = df['Text'].replace('\n', '')

    if save:

        df.to_csv(save_path+fn.name[:-3] + 'csv', index=False, sep='|')

    return df
df = extract_transcript('../input/chai-time-data-science/Raw Subtitles/E73.txt')
df.head()
df.tail()
from fastai.vision import *
path = Path('../input/chai-time-data-science/Raw Subtitles/')
path.ls()[:5]
for fn in path.ls():

    extract_transcript(fn, '../output/kaggle/working/')
df = pd.read_csv('./E70.csv', delimiter='|')
df.head()
df.tail()