import random,copy,re
text = (

    'Hello, how are you? I am Romeo.\n'

    'Hello, Romeo My name is Juliet. Nice to meet you.\n'

    'Nice meet you too. How are you today?\n'

    'Great. My baseball team won the competition.\n'

    'Oh Congratulations, Juliet\n'

    'Thanks you Romeo'

)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')     # 去除 '.', ',', '?', '!'

print(sentences)
tokens_a_index, tokens_b_index= random.randrange(len(sentences)), random.randrange(len(sentences)) # sample random index in sentences

print(tokens_a_index,"\n")  #tokens_a来自第几句话

print(tokens_b_index,"\n")  #tokens_b来自第几句话
word_list = list(set(" ".join(sentences).split()))

word_dict = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}   



for i, w in enumerate(word_list):

    word_dict[w] = i + 4

number_dict = {i: w for i, w in enumerate(word_dict)}

vocab_size = len(word_dict)



print("word_dict:",word_dict,"\n")

print("number_dict:",number_dict,"\n")   #句子中所有的词+标记组成的词典
token_list = list()

for sentence in sentences:

    arr = [word_dict[s] for s in sentence.split()]

    token_list.append(arr)



print("token_list:",token_list)    # 每句话对应的词id列表
tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]

input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']] 

segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)  



input_ids_origin = copy.deepcopy(input_ids)  # 列表深拷贝，保留原始的input_ids列表

#print("input_ids_origin:",input_ids_origin,"\n")
# MASK LM

max_pred = 5 # max tokens of prediction

n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15))))          # 15 % of tokens in one sentence

cand_maked_pos = [i for i, token in enumerate(input_ids)                     # 取出input_ids中的非标记ids

                  if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]

random.shuffle(cand_maked_pos)

masked_tokens, masked_pos = [], []

for pos in cand_maked_pos[:n_pred]:   # 对cand_maked_pos[:n_pred]范围内的ids进行mask

    masked_pos.append(pos)

    masked_tokens.append(input_ids[pos])

    if random.random() < 0.8:  # 80%

        input_ids[pos] = word_dict['[MASK]'] # make mask

    elif random.random() < 0.5:  # 10%,  为了展示效果，这里设置为0.5

        index = random.randint(0, vocab_size - 1) # random index in vocabulary

        input_ids[pos] = word_dict[number_dict[index]] # replace
print("word_dict:",word_dict,"\n")

print("input_ids_origin:",input_ids_origin,"\n")

print("input_ids_masked",input_ids,"\n")
input_ids_origin_words = [number_dict[i] for i in input_ids_origin ]

input_ids_masked_words = [number_dict[i] for i in input_ids ]

print("input_ids_origin_words:",input_ids_origin_words,"\n")

print("input_ids_masked_words",input_ids_masked_words,"\n")