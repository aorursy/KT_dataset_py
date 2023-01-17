#import and preview the data

import pandas as pd

df = pd.read_csv('../input/lyrics.csv')

df.head()
df.info()
#replace carriage returns

df = df.replace({'\n': ' '}, regex=True)

df.head()
#count the words in each song

df['word_count'] = df['lyrics'].str.split().str.len()

df.head()
#check the word counts by genre

df['word_count'].groupby(df['genre']).describe()
#let's see what the songs with 1 word look like

df1 = df.loc[df['word_count'] == 1]

df1
#elimintate the 1-word songs and review the data again

df = df[df['word_count'] != 1]

df['word_count'].groupby(df['genre']).describe()
#There are still some outliers on the low end. Reviewing songs with less than 100 words.

df100 = df.loc[df['word_count'] <= 100]

df100
#let's check on the high end

df1000 = df.loc[df['word_count'] >= 1000]

df1000
#let's get rid of the outliers on the low and high end using somewhat randomly selected points

del df1, df100, df1000 

df_clean = df[df['word_count'] >= 100]

df_clean = df[df['word_count'] <= 1000]

df_clean['word_count'].groupby(df_clean['genre']).describe()
#let's see how much smaller the data set is now

df.info()
#check the overall distribution of the cleaned dataset

import seaborn as sns

sns.violinplot(x=df_clean["word_count"])
#compare wordcounts by genre

import matplotlib as mpl

mpl.rc("figure", figsize=(12, 6))

sns.boxplot(x="genre", y="word_count", data=df_clean)
from fastai.text import *
df_clean = df_clean[~df_clean.isna().any(axis=1)]

df_clean['train'] = np.random.choice(a=[True,False], size=len(df_clean), p=[0.8,0.2])
df_clean.groupby('train').size()

df_full = df_clean.copy()

df_clean = df_clean.sample(frac=0.1, random_state=1)
data = TextClasDataBunch.from_df(path='.', 

                             train_df=df_clean[df_clean['train']], 

                             valid_df=df_clean[~df_clean['train']], 

                             text_cols = 'lyrics',

                             label_cols = 'genre'

                            )
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=0.5)

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.save('first', return_path=True)
learn.load('first')
learn.load('first')

# learn.load('first');

learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(2e-2/(2.6**4),2e-2), moms=(0.8,0.7))
learn.save('second')
data_lm = TextLMDataBunch.from_df(path='.', 

                             train_df=df_clean[df_clean['train']], 

                             valid_df=df_clean[~df_clean['train']], 

                             text_cols = 'lyrics'

                            )
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
learn.fit_one_cycle(5, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned');
TEXT = "Take me to the place where you go"

N_WORDS = 50

N_SENTENCES = 2

print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
learn.save_encoder('fine_tuned_enc')
learn = text_classifier_learner(data, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.predict("I've been living with a shadow overhead I've been sleeping with a cloud above my bed I've been lonely for so long Trapped in the past, I just can't seem to move on I've been hiding all my hopes and dreams away Just in case I ever need 'em again someday I've been setting aside time To clear a little space in the corners of my mind All I wanna do is find a way back into love I can't make it through without a way back into love I've been watching, but the stars refuse to shine I've been searching, but I just don't see the signs I know that it's out there There's got to be something for my soul somewhere I've been looking for someone to shed some light Not somebody just to get me through the night I could use some direction And I'm open to your suggestions All I wanna do is find a way back into love I can't make it through without a way back into love And if I open my heart again I guess I'm hoping you'll be there for me in the end There are moments when I don't know if it's real Or if anybody feels the way I feel I need inspiration Not just another negotiation All I wanna do is find a way back into love I can't make it through without a way back into love And if I open my heart to you I'm hoping you'll show me what to do And if you help me to start again You know that I'll be there for you in the end")

old_town_road = "Yeah, I'm gonna take my horse to the old town road I'm gonna ride 'til I can't no more I'm gonna take my horse to the old town road I'm gonna ride 'til I can't no more (Kio, Kio) I got the horses in the back Horse tack is attached Hat is matte black Got the boots that's black to match Ridin' on a horse, ha You can whip your Porsche I been in the valley You ain't been up off that porch, now Can't nobody tell me nothin' You can't tell me nothin' Can't nobody tell me nothin' You can't tell me nothin' Ridin' on a tractor Lean all in my bladder Cheated on my baby You can go and ask her My life is a movie Bull ridin' and boobies Cowboy hat from Gucci Wrangler on my booty Can't nobody tell me nothin' You can't tell me nothin' Can't nobody tell me nothin' You can't tell me nothin' Yeah, I'm gonna take my horse to the old town road I'm gonna ride 'til I can't no more I'm gonna take my horse to the old town road I'm gonna ride 'til I can't no more Hat down, cross town, livin' like a rockstar Spent a lot of money on my brand new guitar Baby's got a habit, diamond rings and Fendi sports bras Ridin' down Rodeo in my Maserati sports car Got no stress, I've been through all that I'm like a Marlboro Man so I kick on back Wish I could roll on back to that old town road I wanna ride 'til I can't no more Yeah, I'm gonna take my horse to the old town road I'm gonna ride 'til I can't no more I'm gonna take my horse to the old town road I'm gonna ride 'til I can't no more"

learn.predict(old_town_road)
nine = "Who am I to say What any of this means? I have been sleepwalking Since I was fourteen Now, as I write my song I retrace my steps Honestly, it’s easier To let myself forget Still, I check my vital signs Choked up, I realize I’ve been less than half myself For more than half my life Wake up Fall in love again Wage war on gravity There’s so much worth fighting for, you’ll see Another domino falls Either way It looks like empathy To understand all sides But I’m just trying to find myself through Someone else’s eyes So show me what to do To restart this heart of mine How do I forgive myself For losing so much time? Wake up Roll up your sleeves There’s a chain reaction in your heart Muscle memory remembering who you are Stand up And fall in love again and again and again Wage war on gravity There’s so much worth fighting for, you’ll see Another domino falls And another domino falls A little at a time I feel more alive I let the scale tip and feel all of it It’s uncomfortable, but right And we were born to try To see each other through To know and love ourselves and others as well Is the most difficult and meaningful work We’ll ever do"

learn.predict(nine)
boyfriend = "I'm a motherfuckin' train wreck I don't wanna be too much But I don't wanna miss your touch And you don't seem to give a fuck I don't wanna keep you waiting But I do just what I have to do And I might not be the one for you But you ain't about to have no boo 'Cause I know we be so complicated But we be so smitten, it's crazy I can't have what I want, but neither can you You ain't my boyfriend (boyfriend) And I ain't your girlfriend (girlfriend) But you don't want me to see nobody else And I don't want you to see nobody But you ain't my boyfriend (boyfriend) And I ain't your girlfriend (girlfriend) But you don't want me to touch nobody else Baby, we ain't gotta tell nobody Even though you ain't mine, I promise the way we fight Make me honestly feel like we just in love 'Cause, baby, when push comes to shove Damn, baby, I'm a train wreck, too (too) I lose my mind when it comes to you I take time with the ones I choose And I don't want a smile if it ain't from you, yeah I know we be so complicated Lovin' you sometimes drive me crazy 'Cause I can't have what I want and neither can you You ain't my boyfriend (boyfriend) And I ain't your girlfriend (girlfriend) But you don't want me to see nobody else And I don't want you to see nobody But you ain't my boyfriend (boyfriend) And I ain't your girlfriend (girlfriend) But you don't want me to touch nobody else Baby, we ain't gotta tell nobody [Scootie & Ariana Grande] I wanna kiss you (yeah), don't wanna miss you (yeah) But I can't be with you 'cause I got issues Yeah, on the surface seem like it's easy Careful with words, but it's still hard to read me Stress high when the trust low Bad vibes, where'd the fun go? (Oh) Try to open up and love more (love more) Try to open up and love more If you were my boyfriend And you were my girlfriend I probably wouldn't see nobody else But I can't guarantee that by myself You ain't my boyfriend (boyfriend, you ain't my boyfriend) And I ain't your girlfriend (girlfriend, I ain't your girlfriend) But you don't want me to see nobody else (nobody) And I don't want you to see nobody But you ain't my boyfriend (boyfriend, you know you ain't my boyfriend) And I ain't your girlfriend (girlfriend, yeah, mmm) But you don't want me to touch nobody else (nobody) Baby, we ain't gotta tell nobody (oh, yeah) You ain't my boyfriend (boyfriend) And I ain't your girlfriend (girlfriend) But you don't want me to see nobody else And I don't want you to see nobody But you ain't my boyfriend (boyfriend) And I ain't your girlfriend (girlfriend, yeah) But you don't want me to touch nobody else (nobody) Baby, we ain't gotta tell nobody"

learn.predict(boyfriend)
data.classes
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
df_full.groupby('genre').size().plot.pie()