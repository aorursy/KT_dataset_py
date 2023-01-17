import re

from collections import namedtuple, Counter



with open('../input/emoji-data-descriptions-codepoints/emoji-test.txt', 'rt') as file:

    emoji_raw = file.read()

print(emoji_raw[:2800])
print('\U00000063')  # the lower-cae letter "c" for example
print('\U0001F44D')
len('👍'), len('👍🏿')
print('👍🏿'[0], '👍🏿'[1])
import unicodedata

unicodedata.name('👍'), unicodedata.name('👍🏿'[0]), unicodedata.name('👍🏿'[1])
s = 'The rest of my friends are at the restaurant.'

regex = re.compile('rest|restaurant')

regex.findall(s)
regex2 = re.compile('restaurant|rest')

regex2.findall(s)
thumbs_sentence = 'This is thumbs up: 👍, and this is thumbs up with dark skin tone: 👍🏿'

thumbs_regex = re.compile('👍|👍🏿')



thumbs_regex.findall(thumbs_sentence)
thumbs_regex2 = re.compile('👍🏿|👍')

thumbs_regex2.findall(thumbs_sentence)
print('\U0001F44D', '\U0001F44D\U0001F3FF')  # the U0001F44D code point exists in both
for i, line in enumerate(emoji_raw.splitlines()):

    if '; component' in line:

        print(i, line)
EmojiEntry = namedtuple('EmojiEntry', ['codepoint', 'status', 'emoji', 'name', 'group', 'sub_group'])
E_regex = re.compile(r' ?E\d+\.\d+ ') # remove the pattern E<digit(s)>.<digit(s)>

emoji_entries = []



for line in emoji_raw.splitlines()[32:]:  # skip the explanation lines

    if line == '# Status Counts':  # the last line in the document

        break

    if 'subtotal:' in line:  # these are lines showing statistics about each group, not needed

        continue

    if not line:  # if it's a blank line

        continue

    if line.startswith('#'):  # these lines contain group and/or sub-group names

        if '# group:' in line:

            group = line.split(':')[-1].strip()

        if '# subgroup:' in line:

            subgroup = line.split(':')[-1].strip()

    if group == 'Component':  # skin tones, and hair types, skip, as mentioned above

        continue

    if re.search('^[0-9A-F]{3,}', line):  # if the line starts with a hexadecimal number (an emoji code point)

        # here we define all the elements that will go into emoji entries

        codepoint = line.split(';')[0].strip()  # in some cases it is one and in others multiple code points

        status = line.split(';')[-1].split()[0].strip() # status: fully-qualified, minimally-qualified, unqualified

        if line[-1] == '#':

            # The special case where the emoji is actually the hash sign "#". In this case manually assign the emoji

            if 'fully-qualified' in line:

                emoji = '#️⃣'

            else:

                emoji = '#⃣'  # they look the same, but are actually different 

        else:  # the default case

            emoji = line.split('#')[-1].split()[0].strip()  # the emoji character itself

        if line[-1] == '#':  # (the special case)

            name = '#'

        else:  # extract the emoji name

            split_hash = line.split('#')[1]

            rm_capital_E = E_regex.split(split_hash)[1]

            name = rm_capital_E

        templine = EmojiEntry(codepoint=codepoint,

                              status=status,

                              emoji=emoji,

                              name=name,

                              group=group,

                              sub_group=subgroup)

        emoji_entries.append(templine)

emoji_dict = {x.emoji: x for x in emoji_entries}
emoji_dict['😆'].emoji
emoji_entries[0]
emoji_entries[0].emoji
emoji_entries[0].group, emoji_entries[0].sub_group
Counter([x.group for x in emoji_entries])
sorted(Counter([x.sub_group for x in emoji_entries]).items(), key=lambda x: x[1], reverse=True)[:30]
Counter([' | '.join([x.group, x.sub_group]) for x in emoji_entries])
multi_codepoint_emoji = []



for code in [c.codepoint.split() for c in emoji_entries]:

    if len(code) > 1:

        # turn to a hexadecimal number zfilled to 8 zeros e.g: '\U0001F44D'

        hexified_codes = [r'\U' + x.zfill(8) for x in code]  

        hexified_codes = ''.join(hexified_codes)  # join all hexadecimal components 

        multi_codepoint_emoji.append(hexified_codes)



# sorting by length in decreasing order is extremely important as demonstrated above

multi_codepoint_emoji_sorted = sorted(multi_codepoint_emoji, key=len, reverse=True)



# join with a "|" to function as an "or" in the regex

multi_codepoint_emoji_joined = '|'.join(multi_codepoint_emoji_sorted)  

multi_codepoint_emoji_joined[:400]  # sample
single_codepoint_emoji = []



for code in [c.codepoint.split() for c in emoji_entries]:

    if len(code) == 1:

        single_codepoint_emoji.append(code[0])
def get_ranges(nums):

    """Reduce a list of integers to tuples of local maximums and minimums.



    :param nums: List of integers.

    :return ranges: List of tuples showing local minimums and maximums

    """

    nums = sorted(nums)

    lows = [nums[0]]

    highs = []

    if nums[1] - nums[0] > 1:

        highs.append(nums[0])

    for i in range(1, len(nums)-1):

        if (nums[i] - nums[i-1]) > 1:

            lows.append(nums[i])

        if (nums[i + 1] - nums[i]) > 1:

            highs.append(nums[i])

    highs.append(nums[-1])

    if len(highs) > len(lows):

        lows.append(highs[-1])

    return [(l, h) for l, h in zip(lows, highs)]
# We first convert single_codepoint_emoji to integers to make calculations easier

single_codepoint_emoji_int = [int(x, base=16) for x in single_codepoint_emoji]

single_codepoint_emoji_ranges = get_ranges(single_codepoint_emoji_int)

single_codepoint_emoji_ranges[:10]
single_codepoint_emoji_raw = r''  # start with an empty raw string

for code in single_codepoint_emoji_ranges:

    if code[0] == code[1]:  # in this case make it a single hexadecimal character

        temp_regex =  r'\U' + hex(code[0])[2:].zfill(8)

        single_codepoint_emoji_raw += temp_regex

    else:

        # otherwise create a character range, joined by '-'

        temp_regex = '-'.join([r'\U' + hex(code[0])[2:].zfill(8), r'\U' + hex(code[1])[2:].zfill(8)])

        single_codepoint_emoji_raw += temp_regex



single_codepoint_emoji_raw[:100]  # sample
all_emoji_regex = re.compile(multi_codepoint_emoji_joined + '|' +  r'[' + single_codepoint_emoji_raw + r']')

all_emoji_regex.pattern[:500], all_emoji_regex.pattern[-500:]
all_emoji_regex.findall(' '.join([x.emoji for x in emoji_entries])).__len__()
count = 0

found_emoji = set()

for line in emoji_raw.splitlines()[30:]:

    match = all_emoji_regex.findall(line)

    if match:

        if len(match) > 1:

            break

        count += 1

        found_emoji.add(match[0])

        temp_name = [x.name for x in emoji_entries if x.emoji == match[0]][0]

        assert temp_name in line



count, found_emoji.__len__()
with open('emoji_df.csv', 'wt') as file:

    print('emoji;name;group;sub_group;codepoints', file=file)

    for i, em in enumerate(emoji_entries):

        print(f"{em.emoji};{em.name};{em.group};{em.sub_group};{em.codepoint}", file=file)
import pandas as pd

pd.options.display.max_columns = None



emoji_df = pd.read_csv('emoji_df.csv', sep=';')

emoji_df.to_csv('emoji_df.csv', index=False)

emoji_df = pd.read_csv('emoji_df.csv')

emoji_df[:35]
justdoit = pd.read_csv('../input/5000-justdoit-tweets-dataset/justdoit_tweets_2018_09_07_2.csv')

justdoit.head(3)
import advertools as adv

justdoit_emoji_freq = (adv.word_frequency(justdoit['tweet_full_text'],

                                          justdoit['user_followers_count'],

                                          regex=all_emoji_regex.pattern))

justdoit_emoji_freq.head(15)
justdoit_emoji_freq['name'] = [emoji_dict[word].name if word != '️' else '' for word in justdoit_emoji_freq['word']]

justdoit_emoji_freq['group'] = [emoji_dict[word].group if word != '️' else '' for word in justdoit_emoji_freq['word']]

justdoit_emoji_freq['sub_group'] = [emoji_dict[word].sub_group if word != '️' else '' for word in justdoit_emoji_freq['word']]

justdoit_emoji_freq[:40]
(justdoit_emoji_freq

 .groupby('group')

 .agg({'abs_freq': 'sum', 'wtd_freq': 'sum'})

 .sort_values('wtd_freq', ascending=False)

 .style.format({'wtd_freq': '{:,.0f}'}))
(justdoit_emoji_freq

 .groupby('sub_group')

 .agg({'abs_freq': 'sum', 'wtd_freq': 'sum'})

 .sort_values('wtd_freq', ascending=False)

 .head(20)

 .style.format({'wtd_freq': '{:,.0f}'}))