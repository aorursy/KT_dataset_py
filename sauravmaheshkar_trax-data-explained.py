## Install the Latest Version of Trax

!pip install --upgrade trax
import trax
data_pipeline = trax.data.Serial(

    trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),

    trax.data.Tokenize(vocab_dir='gs://trax-ml/vocabs/', vocab_file='en_8k.subword', keys=[0]),

    trax.data.Log(only_shapes=False)

  )

example = data_pipeline()

print(next(example))
data_pipeline = trax.data.Serial(

    trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),

    trax.data.Tokenize(vocab_dir='gs://trax-ml/vocabs/', vocab_file='en_8k.subword', keys=[0]),

    trax.data.Log(only_shapes=True)

  )

example = data_pipeline()

print(next(example))
sentence = ['Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?',

            'But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure. To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure?',

            'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum',

            'At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum facilis est et expedita distinctio. Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae. Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat.']



def sample_generator(x):

  for i in x:

    yield i



example_shuffle = list(trax.data.inputs.shuffle(sample_generator(sentence), queue_size = 2))

example_shuffle
import numpy as np



tensors = np.array([(1.,2.),

           ((3.,4.),(5.,6.))])

padded_tensors = trax.data.inputs.pad_to_max_dims(tensors=tensors, boundary=3)

padded_tensors
data_pipeline = trax.data.Serial(

    trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),

    trax.data.Tokenize(vocab_dir='gs://trax-ml/vocabs/', vocab_file='en_8k.subword', keys=[0]),

    trax.data.BucketByLength(boundaries=[32, 128, 512, 2048],

                             batch_sizes=[512, 128,  32,    8, 1],

                             length_keys=[0]),

    trax.data.Log(only_shapes=True)

  )

example = data_pipeline()

print(next(example))
Filtered = trax.data.Serial(

    trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),

    trax.data.Tokenize(vocab_dir='gs://trax-ml/vocabs/', vocab_file='en_8k.subword', keys=[0]),

    trax.data.BucketByLength(boundaries=[32, 128, 512, 2048],

                             batch_sizes=[512, 128,  32,    8, 1],

                             length_keys=[0]),

    trax.data.FilterByLength(max_length=2048, length_keys=[0]),

    trax.data.Log(only_shapes=True)

  )

filtered_example = Filtered()

print(next(filtered_example))
data_pipeline = trax.data.Serial(

    trax.data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),

    trax.data.Tokenize(vocab_dir='gs://trax-ml/vocabs/', vocab_file='en_8k.subword', keys=[0]),

    trax.data.Shuffle(),

    trax.data.FilterByLength(max_length=2048, length_keys=[0]),

    trax.data.BucketByLength(boundaries=[  32, 128, 512, 2048],

                             batch_sizes=[512, 128,  32,    8, 1],

                             length_keys=[0]),

    trax.data.AddLossWeights(),

    trax.data.Log(only_shapes=True)

  )



example = data_pipeline()

print(next(example))