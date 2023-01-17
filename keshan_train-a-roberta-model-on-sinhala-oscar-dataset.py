!pip install tokenizers transformers > /dev/null
from tokenizers import ByteLevelBPETokenizer

paths = ["../input/si_dedup.txt"]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
!mkdir SinhalaBERTo
tokenizer.save_model("SinhalaBERTo")
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "SinhalaBERTo/vocab.json",
    "SinhalaBERTo/merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM

tokenizer = RobertaTokenizerFast.from_pretrained("SinhalaBERTo", max_len=512)
model = RobertaForMaskedLM(config=config)
model.num_parameters()
%%time

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../input/si_dedup.txt",
    block_size=128,
)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="SinhalaBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
%%time
trainer.train()
trainer.save_model("SinhalaBERTo")
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="SinhalaBERTo",
    tokenizer="SinhalaBERTo"
)
fill_mask("මම ගෙදර <mask>.")