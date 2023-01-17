import torch

from matplotlib import pyplot as plt

import seaborn as sns
ENCODER_WORDS = 2

DECODER_WORDS = 3

EMBEDDING_DIM = 4
encoder_hidden = torch.rand(ENCODER_WORDS, EMBEDDING_DIM)

decoder_hidden = torch.rand(DECODER_WORDS, EMBEDDING_DIM)

decoder_hidden.shape, encoder_hidden.shape
encoder_hidden
decoder_hidden
x1 = torch.Tensor([1, 2, 3])

x2 = torch.Tensor([2, 2, 4])

x3 = torch.Tensor([-2, 2, -4])



print('Скалярное произведение между первым и вторым векторами: {}'.format((x1 * x2).sum().item()))

print('Скалярное произведение между первым и третьим векторами: {}'.format((x1 * x3).sum().item()))

print('Скалярное произведение между вторым и третьим векторами: {}'.format((x2 * x3).sum().item()))

print()

print('Скалярное произведение первого вектора с самим собой: {}'.format((x1 * x1).sum().item()))

print('Скалярное произведение второго вектора с самим собой: {}'.format((x2 * x2).sum().item()))

print('Скалярное произведение третьего вектора с самим собой: {}'.format((x3 * x3).sum().item()))
print('Косинусная близость между первым и вторым векторами: {}'.format((x1 * x2).sum()/(x1.norm()*x2.norm()).item()))

print('Косинусная близость между первым и первым векторами: {}'.format((x1 * x1).sum()/(x1.norm()*x1.norm()).item()))
attention_scores = torch.matmul(decoder_hidden, encoder_hidden.t())
attention_scores
plt.figure(figsize=(12, 12))

plt.title('Attention Scores')

sns.heatmap(attention_scores.numpy(), annot=True, cmap="YlGnBu")

plt.xlabel('Слова энкодера')

plt.ylabel('Слова декодера')
attention_scores.shape
attention_scores[0]
attention_distribution = torch.softmax(attention_scores, 1)
attention_distribution_2 = torch.softmax(attention_scores / EMBEDDING_DIM ** 0.5, 1)
attention_distribution
# в сумме должны получить вектор из единиц размером количество слов в декодере

attention_distribution.sum(1)
plt.figure(figsize=(12, 12))

plt.title('Attention Distribution')

sns.heatmap(attention_distribution.numpy(), annot=True, cmap="YlGnBu")

plt.xlabel('Слова энкодера')

plt.ylabel('Слова декодера')
plt.figure(figsize=(12, 12))

plt.title('Attention Distribution')

sns.heatmap(attention_distribution_2.numpy(), annot=True, cmap="YlGnBu")

plt.xlabel('Слова энкодера')

plt.ylabel('Слова декодера')
attention_distribution
attention_distribution_2
x = torch.rand(4, 128)
attn_scores = torch.matmul(x, x.t())
attn_dist = torch.softmax(attn_scores, 1)
plt.figure(figsize=(12, 12))

plt.title('Attention Distribution')

sns.heatmap(attn_dist.numpy(), annot=True, cmap="YlGnBu")

plt.xlabel('Слова энкодера')

plt.ylabel('Слова декодера')
attn_dist = torch.softmax(torch.matmul(x, x.t()) / 128 ** 0.5 , 1)
attn_dist
plt.figure(figsize=(12, 12))

plt.title('Attention Distribution')

sns.heatmap(attn_dist.numpy(), annot=True, cmap="YlGnBu")

plt.xlabel('Слова энкодера')

plt.ylabel('Слова декодера')
c00 = attention_distribution[0][0].item()

c01 = attention_distribution[0][1].item()



print('Доля памяти нулевого слова энкодера для нулевого слова из декодера: {:.2f} %'.format(c00 * 100))

print('Доля памяти первого слова энкодера для нулевого слова из декодера: {:.2f} %'.format(c01 * 100))



print()



c10 = attention_distribution[1][0].item()

c11 = attention_distribution[1][1].item()



print('Доля памяти нулевого слова энкодера для первого слова из декодера: {:.2f} %'.format(c10 * 100))

print('Доля памяти первого слова энкодера для первого слова из декодера: {:.2f} %'.format(c11 * 100))
print('Для первого слова декодера нужно вспомнить {:.2f} % нулевого слова энкодера и {:.2f} % первого слова энкодера'.format(

    c10 * 100, c11 * 100))

print('Или')

print('Для первого слова декодера нулевое слова энкодера релевантно на {:.2f} %, а первое слово энкодера на {:.2f} %'.format(

    c10 * 100, c11 * 100))
# наша таблицу с весами

attention_distribution
# наши вектора слов энкодера

encoder_hidden
# наши вектора слов декодера

decoder_hidden
# веса слов энкодера для нулевого слова декодера

attention_distribution[0]
# два веса на каждое из слов в энкодере

attention_distribution[0][0], attention_distribution[0][1]
# домножим вектор нулевого слова энкодера на его вес (какую долю информации мы хотим взять от этого слова)

attention_distribution[0][0] * encoder_hidden[0]
# тоже самое для первого слова в энкодере

attention_distribution[0][1] * encoder_hidden[1]
print('Для нулевого слова декодера нужно вспомнить {:.2f} % нулевого слова энкодера и {:.2f} % первого слова энкодера'.format(

    c00 * 100, c01 * 100))
# теперь давайте эту информацию сложим

# то есть возьмем часть от нулевого слова и часть от первого

# эти части зависят от того насколько для текущего слова декодера нам важные слова энкодера

attn_vec_0 = attention_distribution[0][0] * encoder_hidden[0] + attention_distribution[0][1] * encoder_hidden[1]
# получился вектор внимания нулевого слова декодера относительно слов энкодера

attn_vec_0
# сделаем тоже самое для остальных слов декодера

# их у нас все еще 3

attn_vec_1 = attention_distribution[1][0] * encoder_hidden[0] + attention_distribution[1][1] * encoder_hidden[1]

attn_vec_2 = attention_distribution[2][0] * encoder_hidden[0] + attention_distribution[2][1] * encoder_hidden[1]
# сделаем массив для векторов внимания, в нем должно быть количество векторов равное количеству слов в декодере

# потому что мы считаем вектора внимания для каждого слова в декодере, а "обращать внимание" будем на слова в энкодере

attention_vectors = [attn_vec_0, attn_vec_1, attn_vec_2]
# теперь давайте переведем массив в матрицу

# добавим в каждый вектор нулевую размерность

# и по этой размерности сконкатенируем

attention_vectors = torch.cat([vec.unsqueeze(0) for vec in attention_vectors], dim=0)
# вот что у нас получилось

# это просто несколько комбинаций векторов энкодера

attention_vectors
# посмотрим размерность

attention_vectors.shape
# убедимся, что она такая же, что и у декодера

decoder_hidden.shape == attention_vectors.shape

# этого мы и хотели добиться
# можно либо конкатенировать

output_vectors = torch.cat([decoder_hidden, attention_vectors], dim=1)
# теперь размерность эмбеддинга каждого слова декодера стала в два раза больше

output_vectors.shape
# либо складывать

output_vectors = decoder_hidden + attention_vectors
output_vectors.shape
attention_vectors_new = torch.matmul(attention_distribution, encoder_hidden)
# мы получили ту же самую матрицу

attention_vectors_new, (attention_vectors_new - attention_vectors).sum().item()



# вы можете выписать на листок перемножение матриц attention_distribution и encoder_hidden

# и проверить
# есть представления слов энкодера

encoder_hidden



# есть представления слов декодера

decoder_hidden



# мы хотим понять насколько каждое слово в декодере похоже на каждое слово в энкодере

# сделаем это через скалярное произведение матриц

attention_scores = torch.matmul(decoder_hidden, encoder_hidden.t())



# далее для каждого слова из декодера (каждой строки матрицы attention_scores)

# посчитаем "вклад"/"долю"/"важность"/"вес" с помощью софтмакса по строкам

# то есть для каждой строки (для каждого слова декодера) у нас будет распределение важности слов энкодера

attention_distribution = torch.softmax(attention_scores, 1)



# возьмем взвешенную сумму вектором энкодера

# веса мы посчитали в attention_distribution

# если вы не понимаете почему у нас это получилось

# вернитесь выше или распишите на листочке

attention_vectors = torch.matmul(attention_distribution, encoder_hidden)



# скомбинируем вектора декодера и вектора внимания слов энкодера к этим словам декодера

decoder_with_attention = torch.cat([decoder_hidden, attention_vectors], dim=1)
decoder_with_attention.shape
# для батчей нужно использовать bmm вместо matmul

# Performs a batch matrix-matrix product of matrices stored in :attr:`input` and :attr:`mat2`



BATCH_SIZE = 32



encoder_hidden = torch.rand(BATCH_SIZE, ENCODER_WORDS, EMBEDDING_DIM)

decoder_hidden = torch.rand(BATCH_SIZE, DECODER_WORDS, EMBEDDING_DIM)



# .transpose(1, 2) тоже транспонирование, только явно указываем 

attention_scores = torch.bmm(decoder_hidden, encoder_hidden.transpose(1, 2))



# заметим, что у нас добавилась одна размерность и поэтому чуть меняем софтмакс

attention_distribution = torch.softmax(attention_scores, 2)

attention_vectors = torch.bmm(attention_distribution, encoder_hidden)



decoder_with_attention = torch.cat([decoder_hidden, attention_vectors], dim=-1)
decoder_with_attention.shape