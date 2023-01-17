import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
#print(negative_words)

#print(positive_words)

#print(happiness_words)

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test_df

positive_emoticons = [';‑)',';)','*-)','*)',';‑]',';]',';^)',':‑,',';D',':‑P',':P','X‑P','XP','x‑p','xp',':‑p',':p',':‑Þ',':Þ',':‑þ',':þ',':‑b',':b','d:','=p','>:P',':-*',':*',':×',":'‑)",":')",':‑)',':)',':-]',':]',':-3',':3',':->',':>','8-)','8)',':-}',':}',':o)',':c)',':^)','=]','=)',':‑D',':D','8‑D','8D','x‑D','xD','X‑D','XD','=D','=3','B^D',':-))']

negative_emoticons = [':‑(',':(',':‑c',':c',':‑<',':<',':‑[',':[',':-||','>:[',':{',':@','>:(',';(',":'‑(",":'(","D‑':",'D:<','D:','D8','D;','D=','DX']                      

negative_words = ['abysmal', 'adverse', 'alarming', 'angry', 'annoy', 'anxious', 'apathy', 'appalling', 'atrocious', 'awful', 'bad', 'banal', 'barbed', 'belligerent', 'bemoan', 'beneath', 'boring', 'broken', 'callous', "can't", 'clumsy', 'coarse', 'cold', 'cold-hearted', 'collapse', 'confused', 'contradictory', 'contrary', 'corrosive', 'corrupt', 'crazy', 'creepy', 'criminal', 'cruel', 'cry', 'cutting', 'damage', 'damaging', 'dastardly', 'dead', 'decaying', 'deformed', 'deny', 'deplorable', 'depressed', 'deprived', 'despicable', 'detrimental', 'dirty', 'disease', 'disgusting', 'disheveled', 'dishonest', 'dishonorable', 'dismal', 'distress', "don't", 'dreadful', 'dreary', 'enraged', 'eroding', 'evil', 'fail', 'faulty', 'fear', 'feeble', 'fight', 'filthy', 'foul', 'frighten', 'frightful', 'gawky', 'ghastly', 'grave', 'greed', 'grim', 'grimace', 'gross', 'grotesque', 'gruesome', 'guilty', 'haggard', 'hard', 'hard-hearted', 'harmful', 'hate', 'hideous', 'homely', 'horrendous', 'horrible', 'hostile', 'hurt', 'hurtful', 'icky', 'ignorant', 'ignore', 'ill', 'immature', 'imperfect', 'impossible', 'inane', 'inelegant', 'infernal', 'injure', 'injurious', 'insane', 'insidious', 'insipid', 'jealous', 'junky', 'lose', 'lousy', 'lumpy', 'malicious', 'mean', 'menacing', 'messy', 'misshapen', 'missing', 'misunderstood', 'moan', 'moldy', 'monstrous', 'naive', 'nasty', 'naughty', 'negate', 'negative', 'never', 'objectionable', 'odious', 'offensive', 'old', 'oppressive', 'pain', 'perturb', 'pessimistic', 'petty', 'plain', 'poisonous', 'poor', 'prejudice', 'questionable', 'quirky', 'quit', 'reject', 'renege', 'repellant', 'reptilian', 'repugnant', 'repulsive', 'revenge', 'revolting', 'rocky', 'rotten', 'rude', 'ruthless', 'sad', 'savage', 'scare', 'scary', 'scream', 'severe', 'shocking', 'shoddy', 'sick', 'sickening', 'sinister', 'slimy', 'smelly', 'sobbing', 'sorry', 'spiteful', 'sticky', 'stinky', 'stormy', 'stressful', 'stuck', 'stupid', 'substandard', 'suspect', 'suspicious', 'tense', 'terrible', 'terrifying', 'threatening', 'ugly', 'undermine', 'unfair', 'unfavorable', 'unhappy', 'unhealthy', 'unjust', 'unlucky', 'unpleasant', 'unsatisfactory', 'unsightly', 'untoward', 'unwanted', 'unwelcome', 'unwholesome', 'unwieldy', 'unwise', 'upset', 'vice', 'vicious', 'vile', 'villainous', 'vindictive', 'wary', 'weary', 'wicked', 'woeful', 'worthless', 'wound', 'yell', 'yucky', 'zero']

positive_words = ['absolutely', 'accepted', 'acclaimed', 'accomplish', 'accomplishment', 'achievement', 'action', 'active', 'admire', 'adorable', 'adventure', 'affirmative', 'affluent', 'agree', 'agreeable', 'amazing', 'angelic', 'appealing', 'approve', 'aptitude', 'attractive', 'awesome', 'beaming', 'beautiful', 'believe', 'beneficial', 'bliss', 'bountiful', 'bounty', 'brave', 'bravo', 'brilliant', 'bubbly', 'calm', 'celebrated', 'certain', 'champ', 'champion', 'charming', 'cheery', 'choice', 'classic', 'classical', 'clean', 'commend', 'composed', 'congratulation', 'constant', 'cool', 'courageous', 'creative', 'cute', 'dazzling', 'delight', 'delightful', 'distinguished', 'divine', 'earnest', 'easy', 'ecstatic', 'effective', 'effervescent', 'efficient', 'effortless', 'electrifying', 'elegant', 'enchanting', 'encouraging', 'endorsed', 'energetic', 'energized', 'engaging', 'enthusiastic', 'essential', 'esteemed', 'ethical', 'excellent', 'exciting', 'exquisite', 'fabulous', 'fair', 'familiar', 'famous', 'fantastic', 'favorable', 'fetching', 'fine', 'fitting', 'flourishing', 'fortunate', 'free', 'fresh', 'friendly', 'fun', 'funny', 'generous', 'genius', 'genuine', 'giving', 'glamorous', 'glowing', 'good', 'gorgeous', 'graceful', 'great', 'green', 'grin', 'growing', 'handsome', 'happy', 'harmonious', 'healing', 'healthy', 'hearty', 'heavenly', 'honest', 'honorable', 'honored', 'hug', 'idea', 'ideal', 'imaginative', 'imagine', 'impressive', 'independent', 'innovate', 'innovative', 'instant', 'instantaneous', 'instinctive', 'intellectual', 'intelligent', 'intuitive', 'inventive', 'jovial', 'joy', 'jubilant', 'keen', 'kind', 'knowing', 'knowledgeable', 'laugh', 'learned', 'legendary', 'light', 'lively', 'lovely', 'lucid', 'lucky', 'luminous', 'marvelous', 'masterful', 'meaningful', 'merit', 'meritorious', 'miraculous', 'motivating', 'moving', 'natural', 'nice', 'novel', 'now', 'nurturing', 'nutritious', 'okay', 'one', 'one-hundred percent', 'open', 'optimistic', 'paradise', 'perfect', 'phenomenal', 'pleasant', 'pleasurable', 'plentiful', 'poised', 'polished', 'popular', 'positive', 'powerful', 'prepared', 'pretty', 'principled', 'productive', 'progress', 'prominent', 'protected', 'proud', 'quality', 'quick', 'quiet', 'ready', 'reassuring', 'refined', 'refreshing', 'rejoice', 'reliable', 'remarkable', 'resounding', 'respected', 'restored', 'reward', 'rewarding', 'right', 'robust', 'safe', 'satisfactory', 'secure', 'seemly', 'simple', 'skilled', 'skillful', 'smile', 'soulful', 'sparkling', 'special', 'spirited', 'spiritual', 'stirring', 'stunning', 'stupendous', 'success', 'successful', 'sunny', 'super', 'superb', 'supporting', 'surprising', 'terrific', 'thorough', 'thrilling', 'thriving', 'tops', 'tranquil', 'transformative', 'transforming', 'trusting', 'truthful', 'unreal', 'unwavering', 'valued', 'vibrant', 'victorious', 'victory', 'vigorous', 'virtuous', 'vital', 'vivacious', 'wealthy', 'welcome', 'well', 'whole', 'wholesome', 'willing', 'wonderful', 'wondrous', 'worthy', 'wow', 'yes', 'yummy', 'zeal', 'zealous']

happiness_words = ['adore', 'affable', 'agreeable', 'amiable', 'amusing', 'animated', 'appealing', 'as happy as a clam', 'beaming', 'beatific', 'beautiful', 'bliss', 'blissful', 'blithe', 'bowl over', 'buoyant', 'carefree', 'charming', 'cheerful', 'cheeriness', 'cheery', 'chipper', 'chirpy', 'content', 'contented', 'delight', 'delighted', 'delightful', 'diverting', 'droll', 'ebullient', 'ecstasy', 'ecstatic', 'elated', 'elation', 'enchanting', 'endearing', 'energized', 'engaging', 'enjoyable', 'entertaining', 'euphoria', 'euphoric', 'excited', 'exhilarated', 'exuberance', 'exultant', 'exultation', 'favorable', 'fine', 'friendly', 'fulfilled', 'fun', 'funny', 'genial', 'get a kick out of', 'glad', 'gladden', 'glee', 'glorious', 'glory', 'glory in', 'good', 'good humored', 'good mood', 'good natured', 'grateful', 'gratified', 'gratify', 'gratifying', 'great', 'grinning', 'happiness', 'happy', 'happy as a clam', 'heartening', 'heartwarming', 'heavenly', 'high', 'high spirits', 'hilarious', 'hopeful', 'in a good mood', 'in good spirits', 'in seventh heaven', 'invigorated', 'jocular', 'joie de vivre', 'jollity', 'joy', 'joyfulness', 'joyous', 'jubilation', 'jumping for joy', 'lap up', 'lighthearted', 'likable', 'looking on the bright side', 'lovable', 'lovely', 'lucky', 'luxuriate in', 'merriment', 'merry', 'mirth', 'mirthful', 'never been better', 'nice', 'obliging', 'on cloud nine', 'on top of the world', 'open', 'opportune', 'optimistic', 'over the moon', 'overjoyed', 'paradise', 'perkiness', 'perky', 'pleasant', 'please greatly', 'pleased', 'pleasure', 'precious', 'radiant', 'rapture', 'rapturous', 'relaxed', 'relish', 'revel in', 'satisfied', 'savor', 'seventh heaven', 'simpatico', 'smiling', 'source of pleasure', 'sparkle', 'stimulated', 'sunniness', 'sunny', 'sweet', 'take pleasure in', 'tears of joy', 'thrill', 'thrilled', 'tickled pink', 'touching', 'treat', 'triumph', 'untroubled', 'upbeat', 'uplifting', 'vitalized', 'vivacity', 'walking on air', 'welcoming', 'willing', 'wondrous', 'zest for life']
result_arr = []

for ind in test_df.index:

    score = 0

    result = 'neutral'

    selected_word = ''

    #print(test_df['text'][ind])

    for word in test_df['text'][ind].split():

        if(word in negative_words or word in negative_emoticons):

            score = score - 1;

            selected_word = word

        if(word in positive_words or word in happiness_words or word in positive_emoticons):

            score = score + 1;

            selected_word = word

    if score > 0:

        result = 'positive'

    if score < 0:

        result = 'negative'

    if result != 'neutral':

        result_arr.append(selected_word)

        #print(selected_word)

    else:

        result_arr.append(test_df['text'][ind])

        #print(test_df['text'][ind])



test_df['selected_text']=result_arr

test_df[['textID','selected_text']].to_csv('submission.csv',index=False)