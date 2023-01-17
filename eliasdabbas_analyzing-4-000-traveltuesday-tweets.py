import advertools as adv

import pandas as pd

pd.options.display.max_columns = None

adv.__version__
app_key = 'YOUR_APP_KEY'

app_secret = 'YOUR_APP_SECRET'

oauth_token = 'YOUR_OAUTH_TOKEN'

oauth_token_secret = 'YOUR_OAUTH_TOKEN_SECRET'



auth_params = {

    'app_key': app_key,

    'app_secret': app_secret,

    'oauth_token': oauth_token,

    'oauth_token_secret': oauth_token_secret,

}

adv.twitter.set_auth_params(**auth_params)
# code to get the data from the Twitter API 

# ttues = adv.twitter.search(q='#TravelTuesday -filter:retweets', count=5000, tweet_mode='extended')
ttues = pd.read_csv('../input/ttues.csv')
hashtag_summary = adv.extract_hashtags(ttues['tweet_full_text'])

hashtag_summary.keys()
hashtag_summary['overview']
hashtag_summary['hashtag_freq'][:10]
hashtag_summary['top_hashtags'][:20]
mention_summary = adv.extract_mentions(ttues['tweet_full_text'])

mention_summary.keys()
mention_summary['overview']
mention_summary['top_mentions'][:20]
emoji_summary = adv.extract_emoji(ttues['tweet_full_text'])

emoji_summary.keys()
emoji_summary['overview']
emoji_summary['top_emoji'][:20]
emoji_summary['top_emoji_text'][:20]
emoji_summary['top_emoji_groups']
emoji_summary['top_emoji_sub_groups'][:20]
[(i, f) for i, f in enumerate(dir(adv), -42) if f.startswith('extract')]
currency_summary = adv.extract_currency(ttues['tweet_full_text'])
currency_summary.keys()
currency_summary['overview']
currency_summary['top_currency_symbols']
print(*adv.regex.CURRENCY_RAW, sep=' ')
[x for x in  currency_summary['currency_symbol_names'] if x][:20]
[x for x in currency_summary['surrounding_text'] if x][:20]
[x for x in adv.extract_currency(ttues['tweet_full_text'],

                                 left_chars=0, 

                                 right_chars=15)['surrounding_text'] if x][:20]
exclamation_summary = adv.extract_exclamations(ttues['tweet_full_text'])

exclamation_summary.keys()
exclamation_summary['overview']
exclamation_summary['exclamation_mark_freq']
exclamation_summary['top_exclamation_marks']
[x for x in exclamation_summary['exclamation_text'] if x][:10]
intense_summary = adv.extract_intense_words(ttues['tweet_full_text'], min_reps=3)
intense_summary.keys()
intense_summary['overview']
intense_summary['intense_word_freq']
intense_summary['intense_words_flat'][20:50]
for i, tweet in enumerate(intense_summary['intense_words'][:100]):

    if tweet:

        print('Tweet original text:')

        print(ttues['tweet_full_text'][i])

        print()

        print('Extracted intense words:')

        print(tweet)

        print('\n===============\n')
print(*adv.regex.QUESTION_MARK_RAW, sep='  ')
'what?' == 'what?'
';' == ';'
from unicodedata import name

name(';')
name('¿')
cervantes = """Por Dios, hermano, que agora me acabo de desengañar de un engaño en 

que he estado todo el mucho tiempo que ha que os conozco, en el cual siempre os he 

tenido por discreto y prudente en todas vuestras aciones. Pero agora veo que estáis 

tan lejos de serlo como lo está el cielo de la tierra. ¿Cómo que es posible que 

cosas de tan poco momento y tan fáciles de remediar puedan tener fuerzas de 

suspender y absortar un ingenio tan maduro como el vuestro, y tan hecho a romper 

y atropellar por otras dificultades mayores? A la fe, esto no nace de falta de 

habilidad, sino de sobra de pereza y penuria de discurso. ¿Queréis ver si es verdad 

lo que digo? Pues estadme atento y veréis cómo, en un abrir y cerrar de ojos, confundo 

todas vuestras dificultades y remedio todas las faltas que decís que os suspenden y 

acobardan para dejar de sacar a la luz del mundo la historia de vuestro famoso 

don Quijote, luz y espejo de toda la caballería andante.  Decid -le repliqué yo, 

oyendo lo que me decía-: ¿de qué modo pensáis llenar el vacío de mi temor y reducir 

a claridad el caos de mi confusión?"""
preguntas = adv.extract_questions([cervantes])
print(*preguntas['question_text'][0], sep='\n\n')
name('؟')
arabic = ['مرحباً. ما اسمك؟', 'كيف حالك؟ ماذا تفعل؟']
adv.extract_questions(arabic)['question_text']
hebrew = ['שלום. מה שלומך? מה השם שלך?']
adv.extract_questions(hebrew)['question_text']
print('\N{Interrobang}')
for i, q in enumerate(adv.QUESTION_MARK_RAW[1:-1], 1):

    print(f'{i:>2}: {q:^4} {name(q).title()}')
question_summary = adv.extract_questions(ttues['tweet_full_text'])
question_summary.keys()
question_summary['overview']
question_summary['top_question_marks']
question_summary['question_mark_freq']
for i, tweet in enumerate(ttues.tweet_full_text[:40]):

    if '?' in tweet:

        print('Tweet original text:')

        print(tweet)

        print()

        print('Extracted question(s):')

        print(question_summary['question_text'][i])

        print('\n==============\n')
question_summary['question_text'][:30]
photo_words = adv.extract_words(ttues['tweet_full_text'], ['photo', 'image', 'img', 'camera'])
photo_words['overview']
photo_words.keys()
photo_words['top_words'][:20]
adv.extract_words(ttues['tweet_full_text'],['photo', 'image', 'img', 'camera'], 

                 entire_words_only=True)['top_words']
exclamation_summary = adv.extract(ttues['tweet_full_text'], regex='\S+!', 

                                  key_name='exclamation')
exclamation_summary['overview']
exclamation_summary['top_exclamations'][:10]
url_summary = adv.extract_urls(ttues['tweet_full_text'])

url_summary.keys()
url_summary['overview']
url_summary['url_freq']
url_summary['top_domains']
url_summary['top_tlds']
text_list = ['one two three', 'four five six']

adv.word_tokenize(text_list, phrase_len=1)
adv.word_tokenize(text_list, phrase_len=2)
adv.word_tokenize(text_list, phrase_len=3)
word_freq = adv.word_frequency(ttues['tweet_full_text'], 

                               ttues['user_followers_count'], 

                               rm_words=list(adv.stopwords['english'])+["it's", '–', '',

                                                                        ' ', '—', '-', '&amp'])
word_freq.head(20).style.format({'wtd_freq': '{:,}', 'rel_value': '{:,}'})
word_freq2 = adv.word_frequency(ttues['tweet_full_text'], 

                               ttues['user_followers_count'], 

                                phrase_len=2,

                               rm_words=list(adv.stopwords['english'])+["it's", '–', '',

                                                                        ' ', '—', '-', '&amp']+

                               ['to be', 'of the', 'it is', 'is the', 'in the', 'if you', 

                                'to travel', 'for a', 'i have', 'to the', 'at the', 'as a', 'is to'])
word_freq2.head(20).style.format({'wtd_freq': '{:,}', 'rel_value': '{:,}'})
hash_frequency = adv.word_frequency(ttues['tweet_full_text'], 

                                    ttues['user_followers_count'], 

                                    regex=adv.regex.HASHTAG_RAW)

hashtag_summary['top_hashtags'][:10]
hash_frequency.head(20).style.format({'wtd_freq': '{:,}', 'rel_value': '{:,}'})
mention_freq = adv.word_frequency(ttues['tweet_full_text'], 

                                  ttues['user_followers_count'],

                                  regex=adv.regex.MENTION_RAW)
mention_freq.head(15).style.format({'wtd_freq': '{:,}', 'rel_value': '{:,}'})
currency_freq = adv.word_frequency(ttues['tweet_full_text'], 

                                  ttues['user_followers_count'],

                                  regex=adv.regex.CURRENCY_RAW)
currency_freq.head().style.format({'wtd_freq': '{:,}', 'rel_value': '{:,}'})
emoji_freq = adv.word_frequency(ttues['tweet_full_text'], 

                                ttues['user_followers_count'],

                                regex=adv.emoji.EMOJI.pattern)

emoji_freq.head(20)
(emoji_freq

 .assign(emoji_text=[adv.emoji.EMOJI_ENTRIES[x].name if x != '️' else '' for x in emoji_freq['word']],

         emoji_group=[adv.emoji.EMOJI_ENTRIES[x].group if x != '️' else '' for x in emoji_freq['word']],

         emoji_sub_group=[adv.emoji.EMOJI_ENTRIES[x].sub_group if x != '️' else '' for x in emoji_freq['word']])

 .head(30)

 .style.format({'wtd_freq': '{:,}', 'rel_value': '{:,}'}))