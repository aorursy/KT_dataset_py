# Import libraries

import os
import re
import matplotlib.pyplot as plt
# Load data
transcripts = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file = os.path.join(dirname,filename)
        f = open(file)
        transcripts[os.path.basename(file[:-4])] = f.read()
# Get word count of each episode
episode_word_count = {episode: len(re.sub('\\n',' ',transcripts[episode]).split()) for episode in transcripts.keys()}
plt.hist(list(episode_word_count.values()),bins='auto')
plt.show()