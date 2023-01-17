example = "example-1	I saw Oratile slap Rorisang on her left shoulder.	her	31	Oratile	6	False	Rorisang	19	True"

line = example.split('\t')
max_seq_length = 12 #For the cometition, I used 64 and 128, but keeping it very short for clarity.
text = line[1]

P_offset = int(line[3])

A_offset = int(line[5])

B_offset = int(line[8])
char_off = sorted([

  [P_offset, 0],

  [A_offset, 1],

  [B_offset, 2]

], key=lambda x: x[0])

char_off
text_segments = [text[:char_off[0][0]], 

text[char_off[0][0]:char_off[1][0]], 

text[char_off[1][0]:char_off[2][0]], 

text[char_off[2][0]:]]

text_segments
!cp ../input/bertfiles/tokenization.py .

import tokenization

tokenizer = tokenization.FullTokenizer(vocab_file='../input/bertfiles/vocab.txt', do_lower_case=True)

token_segments = []

num_tokens = []

for segment in text_segments:

    token_segment = tokenizer.tokenize(segment)

    token_segments.append(token_segment)

    num_tokens.append(len(token_segment))

token_segments
import numpy as np

while np.sum(num_tokens) > (max_seq_length - 2):

    index = np.argmax([num_tokens[0] * 2, num_tokens[1], num_tokens[2], num_tokens[3] * 2])

    if index == 0:

        token_segments[index] = token_segments[index][1:]

    elif index == 3:

        token_segments[index] = token_segments[index][:-1]

    else: #middle segments

        middle = num_tokens[index] // 2

        token_segments[index] = token_segments[index][:middle] + token_segments[index][middle + 1:]

    num_tokens[index] -= 1

token_segments
tokens = []

tokens.append("[CLS]")

for segment in token_segments:

    temp = ''

    for token in segment:

        tokens.append(token)

tokens.append("[SEP]")
offset = 1 #to account for "[CLS]"

for i, row in enumerate(char_off):

    offset += num_tokens[i]

    row[0] = offset



token_off = sorted(char_off, key=lambda x: x[1])

token_off
P_mask = [0] * max_seq_length

A_mask = [0] * max_seq_length

B_mask = [0] * max_seq_length



P_mask[token_off[0][0]] = 1

A_mask[token_off[1][0]] = 1

B_mask[token_off[2][0]] = 1



print(P_mask)

print(A_mask)

print(B_mask)

print(tokens)