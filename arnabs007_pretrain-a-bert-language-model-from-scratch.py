!pip install torch
!pip install tokenizers
!pip install transformers
# Train a tokenizer
import tokenizers
 
bwpt = tokenizers.BertWordPieceTokenizer(vocab_file=None)
 
filepath = "input file directory"

bwpt.train(
    files=[filepath],
    vocab_size=50000,
    min_frequency=3,
    limit_alphabet=1000
)

bwpt.save('/kaggle/working/', 'name')
# Load the tokenizer
from transformers import BertTokenizer, LineByLineTextDataset

vocab_file_dir = '/kaggle/input/bert-bangla/bangla-vocab.txt' 

tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)

sentence = 'শেষ দিকে সেনাবাহিনীর সদস্যরা এসব ঘর তাঁর প্রশাসনের কাছে হস্তান্তর করেন'

encoded_input = tokenizer.tokenize(sentence)
print(encoded_input)
# print(encoded_input['input_ids'])
%%time

'''
transformers has a predefined class LineByLineTextDataset()
which reads your text line by line and converts them to tokens
'''

dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = '/kaggle/input/bert-bangla/raw_bangla_for_BERT.txt',
    block_size = 128  # maximum sequence length
)

print('No. of lines: ', len(dataset)) # No of lines in your datset
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

config = BertConfig(
    vocab_size=50000,
    hidden_size=768, 
    num_hidden_layers=6, 
    num_attention_heads=12,
    max_position_embeddings=512
)
 
model = BertForMaskedLM(config)
print('No of parameters: ', model.num_parameters())


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='/kaggle/working/',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)
%%time
trainer.train()
trainer.save_model('/kaggle/working/')
from transformers import pipeline

model = BertForMaskedLM.from_pretrained('/kaggle/working/')

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)
fill_mask('লাশ উদ্ধার করে ময়নাতদন্তের জন্য কক্সবাজার [MASK] মর্গে পাঠিয়েছে পুলিশ')
fill_mask('১৯৭১ সালে বাংলাদেশ ৯ মাস মুক্তিযুদ্ধ করে [MASK] অর্জন করে')