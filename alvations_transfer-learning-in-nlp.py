# Jupyter Candies 



# Run the notebook readable w/o wanrings.

# P/S: Not a good habit to do this, but squelching warnings 

#      so that the notebook is easier to demo.

import warnings

warnings.filterwarnings('ignore')
# Install the additional libaries.

! pip install -U git+https://github.com/huggingface/transformers

## ! pip install torch
from itertools import chain

from collections import namedtuple



import numpy as np

import torch

from transformers import BertTokenizer, BertModel, BertForMaskedLM



device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import BertTokenizer, BertModel, BertForMaskedLM



# Load pre-trained model tokenizer (vocabulary)

# A tokenizer will split the text into the appropriate sub-parts (aka. tokens).

# Depending on how the pre-trained model is trained, the tokenizers defers.

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')



# Example of a tokenized input after WordPiece Tokenization.

text = "[CLS] my dog is cute [SEP] he likes playing [SEP]"

print(tokenizer.wordpiece_tokenizer.tokenize(text))
"playing" in tokenizer.wordpiece_tokenizer.vocab
print("slacking" in tokenizer.wordpiece_tokenizer.vocab)
text = "[CLS] my dog is cute [SEP] he likes slacking [SEP]"

tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(text) # There, we see the ##ing token!

print(tokenized_text)
token_indices = tokenizer.convert_tokens_to_ids(tokenized_text)

token_indices
import numpy as np



# We need to create an array that indicates the end of sentences, delimited by [SEP]

text = "[CLS] my dog is cute [SEP] he likes slacking [SEP]"

tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(text)  # There, we see the ##ing token!



# First we find the indices of `[SEP]`, and incrementally adds it up. 

# Here's some Numpy gymnastics... 

# Thanks to @divakar https://stackoverflow.com/a/58316889/610569

m = np.asarray(tokenized_text) == "[SEP]"

segments_ids = m.cumsum()-m



tokens_tensor, segments_tensors = torch.tensor([token_indices]), torch.tensor([segments_ids])



# See the type change?

print(tokens_tensor.shape, type(token_indices), type(tokens_tensor))

print(segments_tensors.shape, type(segments_ids), type(segments_tensors))
tokens_tensor, segments_tensors = torch.tensor([token_indices]), torch.tensor([segments_ids])



# See the type change?

print(tokens_tensor.shape, type(token_indices), type(tokens_tensor))

print(segments_tensors.shape, type(segments_ids), type(segments_tensors))
import torch 



device = 'cuda' if torch.cuda.is_available() else 'cpu'



# When using the BERT model for "encoding", i.e. convert string to array of floats, 

# we use the `BertModel` object from pytorch transformer library.

model = BertModel.from_pretrained('bert-base-uncased')

model.eval(); model.to(device)
# Predict hidden states features for each layer

with torch.no_grad():

    encoded_layers, _ = model(tokens_tensor.to(device), segments_tensors.to(device))

    

print(encoded_layers)
encoded_layers.shape
# Load the model.

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

model.eval(); model.to(device)
# We need to create an array that indicates the end of sentences, delimited by [SEP]

text = "[CLS] please don't let the [MASK] out of the [MASK] . [SEP]"

tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(text)

token_indices = tokenizer.convert_tokens_to_ids(tokenized_text)



# Create the segment indices.

m = np.asarray(tokenized_text) == "[SEP]"

segments_ids = m.cumsum()-m



# Convert them to the arrays to pytorch tensors.

tokens_tensor, segments_tensors = torch.tensor([token_indices]), torch.tensor([segments_ids])



# Apply the model to the inputs.

with torch.no_grad(): # You can take this context manager to mean that we're not training.

    outputs, *_ = model(tokens_tensor.to(device), 

                        token_type_ids=segments_tensors.to(device))



outputs.shape
# Apply the model to the inputs.

with torch.no_grad(): # You can take this context manager to mean that we're not training.

    outputs, *_ = model(tokens_tensor.to(device), token_type_ids=segments_tensors.to(device))
outputs.shape
print(tokenized_text)
# Lets remember our original masked sentence.

print(tokenized_text)

# We have to check where the masked token is from the original text. 

mask_index = tokenized_text.index('[MASK]') 

assert mask_index == 7 # The 7th token.



# Then we fetch the vector for the 7th value, 

# The [0, mask_index] refers to accessing vector of vocab_size for

# the 0th sentence, mask_index-th token.

output_value = outputs[0, mask_index]



# As a sanity check we can see that the shape of the output_value

# is the same as the `vocab_size` from the outputs' shape.

assert int(output_value.shape[0]) == len(tokenizer.wordpiece_tokenizer.vocab)
# Lets recap the original sentence with the masked word.

print(text)



# We have to check where the first masked token is from the original text. 

mask_index = tokenized_text.index('[MASK]') 

output_value = outputs[0, mask_index]



## We use torch.argmax to get the index with the highest value.

mask_word_in_vocab = int(torch.argmax(output_value))

print(tokenizer.convert_ids_to_tokens([mask_word_in_vocab]))
# Lets recap the original sentence with the masked word.

print(text)



# We have to check where the masked tokens are from the original text. 

for mask_index, token in enumerate(tokenized_text):

    if token == '[MASK]':

        output_value = outputs[0, mask_index]

        mask_word_in_vocab = int(torch.argmax(output_value))

        print(tokenizer.convert_ids_to_tokens([mask_word_in_vocab]))
def fill_in_the_blanks(text, model, tokenizer, return_str=False):

    tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(text)

    token_indices = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segment indices.

    m = np.asarray(tokenized_text) == "[SEP]"

    segments_ids = m.cumsum()-m

    # Convert them to the arrays to pytorch tensors.

    tokens_tensor = torch.tensor([token_indices]).to(device)

    segments_tensors = torch.tensor([segments_ids]).to(device)

    

    # Apply the model to the inputs.

    with torch.no_grad(): # You can take this context manager to mean that we're not training.

        outputs, *_ = model(tokens_tensor, token_type_ids=segments_tensors)

    

    output_tokens = []

    for mask_index, token_id in enumerate(token_indices):

        token = tokenizer.convert_ids_to_tokens([token_id])[0]

        if token == '[MASK]':

            output_value = outputs[0, mask_index]

            # The masked word index in the vocab.

            mask_word_in_vocab = int(torch.argmax(output_value))

            token = tokenizer.convert_ids_to_tokens([mask_word_in_vocab])[0]

        output_tokens.append(token)

        

    return " ".join(output_tokens).replace(" ##", "").replace(" ' t ", "'t ") if return_str else output_tokens
# Load the model.

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

model.eval(); model.to(device)



text = "[CLS] please don't let the [MASK] out of the [MASK] . [SEP]"

print(fill_in_the_blanks(text, model, tokenizer, return_str=True))
text = "[CLS] i like to drink beer and eat [MASK] . [SEP]"

print(fill_in_the_blanks(text, model,tokenizer, return_str=True))
text = "[CLS] i like to drink coffee and eat [MASK] . [SEP]"

print(fill_in_the_blanks(text, model, tokenizer, return_str=True))
phoenix_turtle = """Truth may seem but cannot be;\nBeauty brag but ’tis not she;\nTruth and beauty buried be."""

sonnet20 = """A woman’s face with Nature’s own hand painted\nHast thou, the master-mistress of my passion;\nA woman’s gentle heart, but not acquainte\nWith shifting change, as is false women’s fashion;"""

sonnet1 = """From fairest creatures we desire increase,\nThat thereby beauty’s rose might never die,\nBut as the riper should by time decease,\nHis tender heir might bear his memory:"""

sonnet73 = """In me thou see’st the glowing of such fire,\nThat on the ashes of his youth doth lie,\nAs the death-bed whereon it must expire,\nConsum’d with that which it was nourish’d by."""

venus_adonis = """It shall be cause of war and dire events,\nAnd set dissension ‘twixt the son and sire;\nSubject and servile to all discontents,\nAs dry combustious matter is to fire:\nSith in his prime Death doth my love destroy,\nThey that love best their loves shall not enjoy\n"""

sonnet29 = """When, in disgrace with fortune and men’s eyes,\nI all alone beweep my outcast state,\nAnd trouble deaf heaven with my bootless cries,\nAnd look upon myself and curse my fate,"""

sonnet130 = """I have seen roses damask’d, red and white,\nBut no such roses see I in her cheeks;\nAnd in some perfumes is there more delight\nThan in the breath that from my mistress reeks."""

sonnet116 = """Love’s not Time’s fool, though rosy lips and cheeks\nWithin his bending sickle’s compass come;\nLove alters not with his brief hours and weeks,\nBut bears it out even to the edge of doom."""

sonnet18 = """But thy eternal summer shall not fade\nNor lose possession of that fair thou ow’st;\nNor shall Death brag thou wander’st in his shade,\nWhen in eternal lines to time thou grow’st;\nSo long as men can breathe or eyes can see,\nSo long lives this, and this gives life to thee."""

anthony_cleo = """She made great Caesar lay his sword to bed;\nHe plowed her, and she cropped."""



shakespeare = [phoenix_turtle, sonnet20, sonnet1, sonnet73, venus_adonis,

              sonnet29, sonnet130, sonnet116, sonnet18, anthony_cleo]
from transformers import BertConfig, BertForMaskedLM, BertTokenizer



# Load the BERT model.

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

model.eval()

model.to(device)

# Load the BERT Tokenizer.

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Load the BERT Config.

config = BertConfig.from_pretrained('bert-large-uncased')
truth = "that on the ashes of his youth doth lie"

masked_text = "[CLS] that on the ashes of his youth [MASK] lie"

print(fill_in_the_blanks(masked_text, model, tokenizer, return_str=True))
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

import torch.nn.functional as F



class TextDataset(Dataset):

    def __init__(self, texts, tokenizer):

        """

        :param texts: A list of documents, each document is a list of strings.

        :rtype texts: list(string)

        """

        tokenization_process = lambda s: tokenizer.build_inputs_with_special_tokens(

                                             tokenizer.convert_tokens_to_ids(

                                                 tokenizer.tokenize(s.lower())))

        pad_sent = lambda x: np.pad(x, (0,tokenizer.max_len_single_sentence - len(x)), 'constant', 

                                    constant_values=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))

        self.examples = torch.tensor([pad_sent(tokenization_process(doc)) for doc in texts])



    def __len__(self):

        return len(self.examples)



    def __getitem__(self, item):

        return torch.tensor(self.examples[item])

# Initialize the Dataset object.

train_dataset = TextDataset(shakespeare, tokenizer)

# Initalize the DataLoader object, `batch_size=2` means reads 2 poems at a time.

dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=2)
# 10 poems with 510 tokens per poems, 

# if poem has <510, pad with the 0th index.

train_dataset.examples.shape



# For each batch, we read 2 poems at a time.

print(next(iter(dataloader)).shape)
# An example of a batch.

next(iter(dataloader))
def mask_tokens(inputs, tokenizer, mlm_probability=0.8):

    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """



    if tokenizer.mask_token is None:

        raise ValueError(

            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."

        )



    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    probability_matrix = torch.full(labels.shape, mlm_probability)

    special_tokens_mask = [

        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()

    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    if tokenizer._pad_token is not None:

        padding_mask = labels.eq(tokenizer.pad_token_id)

        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # We only compute loss on masked tokens



    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)



    # 10% of the time, we replace masked input tokens with random word

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

    inputs[indices_random] = random_words[indices_random]



    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return inputs, labels
from transformers.optimization import AdamW

from transformers.optimization import get_linear_schedule_with_warmup as WarmupLinearSchedule



Arguments = namedtuple('Arguments', ['learning_rate', 'weight_decay', 'adam_epsilon', 'num_warmup_steps', 

                                     'max_steps', 'num_train_epochs'])



args = Arguments(learning_rate=5e-3, weight_decay=0.0, adam_epsilon=1e-8, num_warmup_steps=0, # Optimizer arguments

                 max_steps=20, num_train_epochs=50  # Training routine arugments

                )  



# Prepare optimizer and schedule (linear warmup and decay)

no_decay = ['bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [

    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},

    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

    ]



optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_steps)
for _e in range(args.num_train_epochs):

    print('Epoch:', _e, '\t' ,end='')

    for step, batch in enumerate(iter(dataloader)):

        print(step, end=', ')



        optimizer.zero_grad()

        # Randomly mask the tokens 80% of the time. 

        inputs, labels = mask_tokens(batch, tokenizer)

        inputs, labels = inputs.to(device), labels.to(device)

        # Initialize the model to train mode.

        model.train()

        # Feed forward the inputs through the models.

        loss, _ = model(inputs, masked_lm_labels=labels)



        # Backpropagate the loss.

        loss.backward()

        # Step through the optimizer.

        optimizer.step()

    print()
truth = "That on the ashes of his youth doth lie"

masked_text = "[CLS] That on the ashes of his youth [MASK] lie [SEP]"

print(fill_in_the_blanks(masked_text, model, tokenizer, return_str=True))
labels
labels.to(device)
inputs[1][0]