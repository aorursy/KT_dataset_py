!pip install -U transformers
import torch

from transformers import AutoModel, AutoTokenizer, BertTokenizer



torch.set_grad_enabled(False)
# Store the model we want to use

MODEL_NAME = "bert-base-cased"



# We need to create the model and tokenizer

model = AutoModel.from_pretrained(MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Tokens comes from a process that splits the input into sub-entities with interesting linguistic properties. 

tokens = tokenizer.tokenize("This is an input example")

print("Tokens: {}".format(tokens))



# This is not sufficient for the model, as it requires integers as input, 

# not a problem, let's convert tokens to ids.

tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens id: {}".format(tokens_ids))



# Add the required special tokens

tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)



# We need to convert to a Deep Learning framework specific format, let's use PyTorch for now.

tokens_pt = torch.tensor([tokens_ids])

print("Tokens PyTorch: {}".format(tokens_pt))



# Now we're ready to go through BERT with out input

outputs, pooled = model(tokens_pt)

print("Tokenw   ise output: {}, Pooled output: {}".format(outputs.shape, pooled.shape))
# tokens = tokenizer.tokenize("This is an input example")

# tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

# tokens_pt = torch.tensor([tokens_ids])



# This code can be factored into one-line as follow

tokens_pt2 = tokenizer.encode_plus("This is an input example", return_tensors="pt")



for key, value in tokens_pt2.items():

    print("{}:\n\t{}".format(key, value))



outputs2, pooled2 = model(**tokens_pt2)

print("Difference with previous code: ({}, {})".format((outputs2 - outputs).sum(), (pooled2 - pooled).sum()))
# Single segment input

single_seg_input = tokenizer.encode_plus("This is a sample input")



# Multiple segment input

multi_seg_input = tokenizer.encode_plus("This is segment A", "This is segment B")



print("Single segment token (str): {}".format(tokenizer.convert_ids_to_tokens(single_seg_input['input_ids'])))

print("Single segment token (int): {}".format(single_seg_input['input_ids']))

print("Single segment type       : {}".format(single_seg_input['token_type_ids']))



# Segments are concatened in the input to the model, with 

print()

print("Multi segment token (str): {}".format(tokenizer.convert_ids_to_tokens(multi_seg_input['input_ids'])))

print("Multi segment token (int): {}".format(multi_seg_input['input_ids']))

print("Multi segment type       : {}".format(multi_seg_input['token_type_ids']))
# Padding highlight

tokens = tokenizer.batch_encode_plus(

    ["This is a sample", "This is another longer sample text"], 

    pad_to_max_length=True  # First sentence will have some PADDED tokens to match second sequence length

)



for i in range(2):

    print("Tokens (int)      : {}".format(tokens['input_ids'][i]))

    print("Tokens (str)      : {}".format([tokenizer.convert_ids_to_tokens(s) for s in tokens['input_ids'][i]]))

    print("Tokens (attn_mask): {}".format(tokens['attention_mask'][i]))

    print()
from transformers import TFBertModel, BertModel



# Let's load a BERT model for TensorFlow and PyTorch

model_tf = TFBertModel.from_pretrained('bert-base-cased')

model_pt = BertModel.from_pretrained('bert-base-cased')
# transformers generates a ready to use dictionary with all the required parameters for the specific framework.

input_tf = tokenizer.encode_plus("This is a sample input", return_tensors="tf")

input_pt = tokenizer.encode_plus("This is a sample input", return_tensors="pt")



# Let's compare the outputs

output_tf, output_pt = model_tf(input_tf), model_pt(**input_pt)



# Models outputs 2 values (The value for each tokens, the pooled representation of the input sentence)

# Here we compare the output differences between PyTorch and TensorFlow.

for name, o_tf, o_pt in zip(["output", "pooled"], output_tf, output_pt):

    print("{} differences: {}".format(name, (o_tf.numpy() - o_pt.numpy()).sum()))
from transformers import DistilBertModel



bert_distil = DistilBertModel.from_pretrained('distilbert-base-cased')

input_pt = tokenizer.encode_plus(

    'This is a sample input to demonstrate performance of distiled models especially inference time', 

    return_tensors="pt"

)





%time _ = bert_distil(input_pt['input_ids'])

%time _ = model_pt(input_pt['input_ids'])
# Let's load German BERT from the Bavarian State Library

de_bert = BertModel.from_pretrained("dbmdz/bert-base-german-cased")

de_tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-cased")



de_input = de_tokenizer.encode_plus(

    "Hugging Face ist einen französische Firma Mitarbeitern in New-York.",

    return_tensors="pt"

)

output_de, pooled_de = de_bert(**de_input)