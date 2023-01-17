! pip install -U pip

! pip install -U tensorflow

! pip install -U tensorflow_hub
import itertools



import torch



import tensorflow as tf

import tensorflow_hub as hub



import numpy as np



from sklearn.metrics.pairwise import cosine_similarity



def cos(a, b):

    return cosine_similarity(torch.tensor(a).view(1, -1), torch.tensor(b).view(1, -1))[0][0]





# Printing candies, make sure that arrays 

# are ellipsis and humanly readable.

np.set_printoptions(precision=4, threshold=10)
# The URL that hosts the Transformer model for Universal Sentence Encoder 

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"



# The URL that hosts the DAN model for Universal Sentence Encoder 

##module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"



# On a local machine, uncomment the last two lines in this cell,

# so that the mmodule don't get redownloaded multiple times.

# when you run the notebook in different sessions.

#

# By setting `TFHUB_CACHE_DIR` environment variable,

# it sets the directory where tf_hub will save the model.

#



##import os

##os.environ["TFHUB_CACHE_DIR"] = os.getcwd() + "/tfhub_models/"
# Import the Universal Sentence Encoder's TF Hub module

# This will take some time to download the model for the first time...

embed = hub.Module(module_url)
bulbasaur = """A strange seed was planted on its back at birth. The plant sprouts and grows with this POKéMON."""

ivysaur = """When the bulb on its back grows large, it appears to lose the ability to stand on its hind legs."""

venusaur = """The plant blooms when it is absorbing solar energy. It stays on the move to seek sunlight."""



charmander = """Obviously prefers hot places. When it rains, steam is said to spout from the tip of its tail."""

charmeleon = """When it swings its burning tail, it elevates the temperature to unbearably high levels."""

charizard = """Spits fire that is hot enough to melt boulders. Known to cause forest fires unintentionally."""



input_texts = [bulbasaur, ivysaur, venusaur, 

              charmander, charmeleon, charizard]

with tf.Session() as session:

    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    sentence_embeddings = session.run(embed(input_texts))
names = ['bulbasaur', 'ivysaur  ', 'venusaur', 

         'charmander', 'charmeleon', 'charizard']
for (mon1, vec1), (mon2, vec2) in itertools.product(zip(names, sentence_embeddings), repeat=2):

    print('\t'.join(map(str, [mon1, mon2, cos(vec1, vec2)])))