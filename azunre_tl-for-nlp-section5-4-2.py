!ls ../input/jw300entw/jw300.en-tw.en
!pip freeze > kaggle_image_requirements.txt
from transformers import BertTokenizerFast # this is just a faster version of BertTokenizer, which you could use instead
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased") # use pre-trained mBERT tokenizer
from transformers import BertForMaskedLM # use masked language modeling



model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased") # initialize to mBERT checkpoint



print("Number of parameters in mBERT model:")

print(model.num_parameters())
from transformers import LineByLineTextDataset



dataset = LineByLineTextDataset(

    tokenizer=tokenizer,

    file_path="../input/jw300entw/jw300.en-tw.tw",

    block_size=128, # how many lines to read at a time 

)
from transformers import DataCollatorForLanguageModeling



data_collator = DataCollatorForLanguageModeling(

    tokenizer=tokenizer,

    mlm=True, mlm_probability=0.15) # use masked language modeling, and mask words with probability of 0.15
from transformers import Trainer, TrainingArguments



training_args = TrainingArguments(

    output_dir="twimbert",

    overwrite_output_dir=True,

    num_train_epochs=1,

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
# Define fill-in-the-blanks pipeline

from transformers import pipeline



fill_mask = pipeline(

    "fill-mask",

    model="twimbert",

    tokenizer=tokenizer

)
# We modified a sentences as "Eyi de ɔhaw kɛse baa sukuu hɔ." => "Eyi de ɔhaw kɛse baa [MASK] hɔ."

# Predict masked token 

print(fill_mask("Eyi de ɔhaw kɛse baa [MASK] hɔ."))