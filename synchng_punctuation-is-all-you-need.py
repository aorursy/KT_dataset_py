!pip install sacrebleu
# product_title_translation_eval_script.py

"""Sample evaluation script for product title translation."""



import re

from typing import List



import regex

from sacrebleu import corpus_bleu



OTHERS_PATTERN: re.Pattern = regex.compile(r'\p{So}')





def eval(preds: List[str], refs: List[str]) -> float:

    """BLEU score computation.



    Strips all characters belonging to the unicode category "So".

    Tokenize with standard WMT "13a" tokenizer.

    Compute 4-BLEU.



    Args:

        preds (List[str]): List of translated texts.

        refs (List[str]): List of target reference texts.

    """

    preds = [OTHERS_PATTERN.sub(' ', text) for text in preds]

    refs = [OTHERS_PATTERN.sub(' ', text) for text in refs]

    return corpus_bleu(

        preds, [refs],

        lowercase=True,

        tokenize='13a',

        use_effective_order= False

    ).score
# Better eval function that can take single strings

def better_eval(preds, refs):

    if not isinstance(preds, list):

        preds = [preds]

    if not isinstance(refs, list):

        refs = [refs]

    return eval(preds, refs)
import pandas as pd

import re
!ls ../input/student-shopee-code-league-product-translation
dev_en_df = pd.read_csv('../input/student-shopee-code-league-product-translation/dev_en.csv')

dev_tcn_df = pd.read_csv('../input/student-shopee-code-league-product-translation/dev_tcn.csv')
dev_df = pd.DataFrame({'text': dev_tcn_df['text'], 'translation_output': dev_en_df['translation_output']})

print(dev_df.shape)

dev_df.head()
better_eval(dev_df['text'].to_list(), dev_df['translation_output'].to_list())
# https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex

zh_pattern = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
# text charset

"".join(sorted(list(set("".join(dev_df['text'].to_list())))))
# translation_output charset

"".join(sorted(list(set("".join(dev_df['translation_output'].to_list())))))
OTHERS_PATTERN.sub('', "‘’◢◤《》『』【】・🎀")
def test_symbol(symbol):

    with_symbol = f'The quick fox jumped {symbol} over the lazy dog'

    without_symbol =  f'The quick fox jumped over the lazy dog'

    return symbol, better_eval(without_symbol, with_symbol)
for symbol in "‘’◢◤《》『』【】・🎀":

    print(test_symbol(symbol))
better_eval('The quick fox jumped over the lazy lazy lazy dog', 'The quick fox jumped over the lazy dog')
masked_character_pattern= re.compile(r'[^ -~×è‘’◢◤《》『』【】・🎀]')
dev_df['text_reduced_chars'] = dev_df['text'].str.replace(masked_character_pattern, ' ').str.replace(' +', ' ')

dev_df.head()
better_eval(dev_df['text_reduced_chars'].to_list(), dev_df['translation_output'].to_list())
dev_df.loc[dev_df['text'].str.contains('【')]
# Note the missing spacing                    !     !                                   !    !

better_eval("IFairies zircon tassel earrings 【33531】", "IFairies zircon tassel earrings 【 33531 】")
funny_punctuation_pattern = re.compile(r'([×è‘’◢◤《》『』【】・🎀])')
dev_df['text_spaced_punctuation'] = dev_df['text_reduced_chars'].str.replace(funny_punctuation_pattern, lambda m: f' {m.groups(1)[0]} ').str.replace(' +', ' ')

dev_df.head()
dev_df.loc[dev_df['text'].str.contains('【')]
better_eval(dev_df['text_spaced_punctuation'].to_list(), dev_df['translation_output'].to_list())
from itertools import chain
# Add some padding to punctuation

raw_replaces = [

    ('#', ' # '),

    ('＆', ' & '),

    (r'\(', ' ('),

    (r'\)', ') '),

    (r'\*', ' * '),

    (r'\+', ' + '),

    ('-', ' - '),

    ('/', ' / '),

    (':', ': '),

    ('<', ' < '),

    ('>', ' > '),

    ('@', ' @ '),

    ('_', ' _ '),

    ('~', ' ~ '),

    ('è', ' è '),

    ('=', ' = '), 

    (r'\\', r' \ '),

    ('Ⅱ', ' Ii '),

    ('。', '. '),

    ('・', ' ・ '),

    (' ㎝ ', ' Cm '),

    ('◢', ' ◢ '),

    ('◤', ' ◤ '),

    ('【', ' 【 '),

    ('】', ' 】 '),

    ('Ⅲ', ' Iii '),

    ('Ⅳ', ' Iv '),

    ('（', ' ('),

    ('）', ') '),

    ('＊', ' * '),

    ('，', ', '),

    ('－', ' - '),

    ('．', '. '),

    ('／', ' / '),

    ('：', ': '),

    ('︱', ' | '),

    ('！', '! '),

    ('＂', '"'),

    ('＃', ' # '),

    ('％', '% '),

    (' ＆ ', ' & '),

    ('＋', ' + '),

    ('＜', ' < '),

    ('＞', ' > '),

    ('？', ' ? '),

    ('～', ' ~ '),

    ('＆', ' & '),

]



# Convert fullwidth characters to english

fullwidth_offset = ord('Ａ') - ord('A')

fullwidth_replaces = [(chr(i + fullwidth_offset), chr(i)) for i in chain(range(ord('0'), ord('9') + 1), range(ord('A'), ord('Z') + 1), range(ord('a'), ord('z') + 1))]



# Allow only ASCII and chinese characters

blacklist_pattern = re.compile(u'[^ -:<-\]_a-z|~×è◢◤〈〉【】⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)

blacklist_replaces = [(blacklist_pattern, ' ')]
cleaning_text = dev_df['text']

for pat, sub in chain(raw_replaces, fullwidth_replaces, blacklist_replaces):

    cleaning_text = cleaning_text.str.replace(pat, sub)

dev_df['text_cleaned'] = cleaning_text

dev_df.head()
dev_df['text_cleaned_reduced_chars'] = dev_df['text_cleaned'].str.replace(masked_character_pattern, ' ').str.replace(' +', ' ')

dev_df.head()
better_eval(dev_df['text_cleaned'].to_list(), dev_df['translation_output'].to_list())
better_eval(dev_df['text_cleaned_reduced_chars'].to_list(), dev_df['translation_output'].to_list())