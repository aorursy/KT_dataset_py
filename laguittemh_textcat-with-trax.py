! pip install trax
import os

import trax
from trax import data
from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as np
# you need to run this cell twice in Kaggle
# Reference: https://www.tensorflow.org/datasets/catalog/ag_news_subset
train_stream = data.TFDS('ag_news_subset', keys=('description', 'label'), train=True)()
eval_stream = data.TFDS('ag_news_subset', keys=('description', 'label'), train=False)()
print(next(train_stream))
data_pipeline = data.Serial(
    data.Tokenize(vocab_file='en_8k.subword', keys=[0]),
    data.Shuffle(),
    data.FilterByLength(max_length=2048, length_keys=[0]),
    data.BucketByLength(boundaries=[  32, 128, 512, 2048],
                             batch_sizes=[512, 128,  32,    8, 1],
                             length_keys=[0]),
    data.AddLossWeights()
)
train_batches_stream = data_pipeline(train_stream)
eval_batches_stream = data_pipeline(eval_stream)
example_batch = next(train_batches_stream)
print(f'shapes = {[x.shape for x in example_batch]}')
model = tl.Serial(
    tl.Embedding(vocab_size=8192, d_feature=50),
    tl.Mean(axis=1),
    tl.Dense(4),
    tl.LogSoftmax()
)
model
# Training task.
train_task = training.TrainTask(
    labeled_data=train_batches_stream,
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=500,
)

# Evaluaton task.
eval_task = training.EvalTask(
    labeled_data=eval_batches_stream,
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    n_eval_batches=20  # For less variance in eval numbers.
)

# Training loop saves checkpoints to output_dir.
output_dir = os.path.expanduser('~/output-dir/')
!rm -rf {output_dir}
training_loop = training.Loop(model,
                              train_task,
                              eval_tasks=[eval_task],
                              output_dir=output_dir)

# Run 2000 steps (batches).
training_loop.run(2000)
inputs, targets, weights = next(eval_batches_stream)
example_input = inputs[0]
expected_class = targets[0]
example_input_str = trax.data.detokenize(example_input, vocab_file='en_8k.subword')
print(f'example input_str: {example_input_str}')
sentiment_log_probs = model(example_input[None, :])  # Add batch dimension.
print(f'Model returned sentiment probabilities: {np.exp(sentiment_log_probs)}')
print(f'Expected class: {expected_class}')
