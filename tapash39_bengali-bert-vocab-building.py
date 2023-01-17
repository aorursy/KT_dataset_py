from tokenizers import BertWordPieceTokenizer
# Initialize an empty BERT tokenizer
tokenizer = BertWordPieceTokenizer(
  clean_text=False,
  handle_chinese_chars=False,
  strip_accents=False,
  lowercase=True,
)
files = ['../input/bengali-oscar-corpus/bn_dedup.txt']
files
# train BERT tokenizer
tokenizer.train(
  files,
  vocab_size=32000,
  min_frequency=2,
  show_progress=True,
  special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
  limit_alphabet=1000,
  wordpieces_prefix="##"
)
tokenizer.save('./')
# tokenizer.save_model(path)
!head -20 ./vocab.txt
!tail -20 ./vocab.txt
