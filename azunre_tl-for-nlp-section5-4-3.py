!ls ../input/jw300entw/jw300.en-tw.en
!pip freeze > kaggle_image_requirements.txt
from tokenizers import BertWordPieceTokenizer
paths = ['../input/jw300entw/jw300.en-tw.tw']



tokenizer = BertWordPieceTokenizer() # Initialize a tokenizer
# Customize training and carry it out

tokenizer.train(

    paths,

    vocab_size=10000,

    min_frequency=2,

    show_progress=True,

    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], # standard BERT special tokens

    limit_alphabet=1000,

    wordpieces_prefix="##",

)



# Save tokenizer to disk

!mkdir twibert

tokenizer.save("twibert")
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("twibert", max_len=512) #  use the language-specific tokenizer we just trained
from transformers import BertForMaskedLM, BertConfig



model = BertForMaskedLM(BertConfig()) # Don't initialize to pretrained, create a fresh one



print("Number of parameters in mBERT model:")

print(model.num_parameters())
from transformers import LineByLineTextDataset



dataset = LineByLineTextDataset(

    tokenizer=tokenizer,

    file_path="../input/jw300entw/jw300.en-tw.tw",

    block_size=128,

)
from transformers import DataCollatorForLanguageModeling



data_collator = DataCollatorForLanguageModeling(

    tokenizer=tokenizer,

    mlm=True, mlm_probability=0.15

)
from transformers import Trainer, TrainingArguments



training_args = TrainingArguments(

    output_dir="twimbert",

    overwrite_output_dir=True,

    num_train_epochs=2, # how about 2 epochs?

    per_gpu_train_batch_size=16,

    save_total_limit=1,

)
trainer = Trainer(

    model=model,

    args=training_args,

    data_collator=data_collator,

    train_dataset=dataset,

    prediction_loss_only=True,

)
import time

start = time.time()

trainer.train()

end = time.time()

print("Number of seconds for training:")

print((end-start))
trainer.save_model("twimbert") # save model
from transformers import pipeline # test model



fill_mask = pipeline(

    "fill-mask",

    model="twimbert",

    tokenizer=tokenizer

)
# same example as before

print(fill_mask("Eyi de ɔhaw kɛse baa [MASK] hɔ."))