from pathlib import Path



import sentencepiece as spm

import pandas as pd
lang_data = pd.read_csv('../input/tamil-wiki-data-extraction/filtered_data.csv.tar.gz', index_col=[0])

lang_data.head()
lang_data.info()
# Initialize directories

OUTPUT_DIR = Path('/kaggle/working')

TEXTS_DIR = OUTPUT_DIR/'texts'

TOK_DIR = OUTPUT_DIR/'tokenizer'



# Create directories

TOK_DIR.mkdir()

TEXTS_DIR.mkdir()

# Save all article texts in seperate files



for t in lang_data.itertuples():

    file_name = Path(TEXTS_DIR/f'text_{t.Index}.txt')

    file_name.touch()

    with file_name.open('w') as f:

        f.write(t.text)
# Check files in directory

len([t for t in TEXTS_DIR.iterdir()]), lang_data.shape[0]
files = ','.join([str(t) for t in TEXTS_DIR.iterdir()])

files[:100]
for v in 8000, 16000, 20000, 30000:

    api_str = f"""--input={files} --vocab_size={v} --model_type=unigram --character_coverage=0.9995 --model_prefix={str(TOK_DIR)}/tok_{v}_size --max_sentence_length=20000"""

    print("Training with vocab set as:", v)

    spm.SentencePieceTrainer.train(api_str)
!rm -rf /kaggle/working/texts/