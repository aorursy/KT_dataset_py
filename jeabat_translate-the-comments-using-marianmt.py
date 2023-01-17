!pip install -U transformers
import pandas as pd

from tqdm.notebook import tqdm
df = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

df = df.sample(100, random_state=12)

df.head(3)
df['lang'].unique()
from transformers import MarianMTModel, MarianTokenizer
df['content_english'] = ''
for i, lang in tqdm(enumerate(['es', 'it', 'tr'])):

    if lang in ['es', 'it']:

        model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'

        df_lang = df.loc[df['lang']==lang, 'comment_text'].apply(lambda x: '>>{}<< '.format(lang) + x)

    else:

        model_name = 'Helsinki-NLP/opus-mt-{}-en'.format(lang)

        df_lang = df.loc[df['lang']==lang, 'comment_text']

    

    tokenizer = MarianTokenizer.from_pretrained(model_name)

    model = MarianMTModel.from_pretrained(model_name, output_loading_info=False)

        

    batch = tokenizer.prepare_translation_batch(df_lang.values,

                                               max_length=192,

                                               pad_to_max_length=True)

    translated = model.generate(**batch)



    df.loc[df['lang']==lang, 'content_english'] = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
df.head(3)
df.to_csv("df_translated.csv")