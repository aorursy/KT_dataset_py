# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Natural Language Toolkit: Some Portuguese texts for exploration in chapter 1 of the book

#

# Copyright (C) 2001-2014 NLTK Project

# Author: Steven Bird <stevenbird1@gmail.com>

# URL: <http://nltk.org/>

# For license information, see LICENSE.TXT

from __future__ import print_function, unicode_literals



from nltk.corpus import machado, mac_morpho, floresta, genesis

from nltk.text import Text

from nltk.probability import FreqDist

from nltk.util import bigrams

from nltk.misc import babelize_shell



print("*** Exemplos introdutórios do livro NLTK em Português ***")

print("Carregando ptext1...ptext4 e psent1...ptext4")

print("Use 'texts()' ou 'sents()' para listar os materiais de texto ou sentença.")



ptext1 = Text(machado.words('romance/marm05.txt'), name="Memórias Póstumas de Brás Cubas (1881)")

ptext2 = Text(machado.words('romance/marm08.txt'), name="Dom Casmurro (1899)")

ptext3 = Text(genesis.words('portuguese.txt'), name="Gênesis")

ptext4 = Text(mac_morpho.words('mu94se01.txt'), name="Folha de Sao Paulo (1994)")



def texts():

    print("ptext1:", ptext1.name)

    print("ptext2:", ptext2.name)

    print("ptext3:", ptext3.name)

    print("ptext4:", ptext4.name)



psent1 = "o amor da glória era a coisa mais verdadeiramente humana que há no homem , e , conseqüentemente , a sua mais genuína feição .".split()

psent2 = "Não consultes dicionários .".split()

psent3 = "No princípio, criou Deus os céus e a terra.".split()

psent4 = "A Cáritas acredita que outros cubanos devem chegar ao Brasil .".split()



def sents():

    print("psent1:", " ".join(psent1))

    print("psent2:", " ".join(psent2))

    print("psent3:", " ".join(psent3))

    print("psent4:", " ".join(psent4))
# Vejamos então um de nossos textos

ptext2
# E uma das sentença

psent3
# Agora, todos os textos

texts()
# E todas as sentenças

sents()
# O método concordance permite ver palavras em um contexto

ptext1.concordance('olhos')
# Para uma dada palavras, é possível encontrar palavras com distribuição de texto similar

print(f"{ptext1.name}:")

ptext1.similar('chegar')

print(f"\n{ptext3.name}:")

ptext3.similar('chegar')
# É possível procurar no texto por colocações significativamente significantes

ptext1.collocations()
# Expressões regulares também podem ser usadas para procurar palavras em contexto

ptext1.findall("<olhos> (<.*>)")
# Gerar texto randomicamente baseado em um dado texto? Sim, dá para fazer isso. Digo, não. Esquece isso

ptext1.generate(words=None)
# Como vimos nos primeiro passos, algumas sentenças já foram definidas

sents()
psent1
# Perceba que a sentença foi tokenizada. Cada token é representado como uma string, inclusive, a

# pontuação, a.k.a "ponto" e "vírgula". Perceba então que os token são combinados em uma lista,

# cujo tamanho é...

len(psent1)
# Vejamos qual o vocabulário dessa sentença

sorted(set(psent1))
# Para cada palavra do vocabulário da sentença, vamos imprimir algumas informações básicas. Mas dessa

# vez, vamos excluir a pontuação

pontuacao = [".", ",", ";", "-", ":"]

vocabulario = [v for v in sorted(set(psent1)) if not v in pontuacao]

for w in vocabulario:

    print(f"{w} = tamanho: {len(w)}, último caractere: {w[-1]}")
# OK, que tal brincar um pouco mais com list comprehension?

tudo_maiusculo = [w.upper() for w in psent2]

print(f"tudo maiúsculo: {tudo_maiusculo}\n")



palavras_terminadas_em_a = [w for w in psent1 if w.endswith('a')]

print(f"palavras terminadas em a: {palavras_terminadas_em_a}\n")



palavas_com_mais_de_15_caracteres = [w for w in ptext4 if len(w) > 15]

print(f"palavas com mais de 15 caracteres: {palavas_com_mais_de_15_caracteres}")
# Podemos inspecionar a frequências em que as palavras aparecem no texto

fd1 = FreqDist(ptext1)

print(f"{fd1}")

fd1
# OK, vamos ver algumas palavras em especifico

fd1["olhos"]
# Qual a palavra que mais aparece no texto? Advinha...

fd1.max()
# Não tem o método samples() na classe FreqDist
# NLTK tem os trabalhos completos de Machado Assis, sendo que cada arquivo contém um de seus trabalhos

from nltk.corpus import machado

machado.fileids()
# Você pode ter mais informações dando uma lida no readme do corpus

print(machado.readme())
# Vamos dar uma olhada em Memórias Póstumas de Brás Cubas.

# É possível acessar o texto como uma lista de caracteres. Por exemplo, vejamos os 200 caracteres a

# partir da posição 10000.

raw_text = machado.raw('romance/marm05.txt')

raw_text[10000:10200]
# No entanto, essa não é uma maneira muito útil de trabalhar com texto. Pensamos nos textos, geralmente,

# como uma sequencia de palavras e pontuações, não de caracteres

text1 = machado.words('romance/marm05.txt')

text1
# Quantidade de palavras

print(f"quantidade de palavras: {len(text1)}")



# Quantidade de palavras, excluindo pontuação

print(f"quantidade de palavras excluindo pontuação: {len([p for p in text1 if not p in pontuacao])}")



# Quantidade de palavras únicas

print(f"quantidade de palavras únicas: {len(set(text1))}")



# Quantidade de palavras únicas, excluindo pontuação

print(f"quantidade de palavras únicas excluindo pontuação: {len([p for p in set(text1) if not p in pontuacao])}")
# Procurando os ngrams mais comuns, que contenham uma palavra alvo em particular

# NOTA: O livro usa a função ingrams, mas no NLTK 3, ela tornou-se apenas ngram

from nltk import ngrams, FreqDist

target_word = 'olhos'

fd = FreqDist(ng for ng in ngrams(text1, 5) if target_word in ng)

fd
# Vejamos uma sequencia de palavras:

from nltk.corpus import mac_morpho

mac_morpho.words()
# Suas sentenças

mac_morpho.sents()
# Suas palavras taggeadas

mac_morpho.tagged_words()
# Também é possível acessar em forma de setenças com suas palavras taggeadas

mac_morpho.tagged_sents()
# Bônus: Taggeamento part-of-speeach (POS)

import nltk

nltk.pos_tag(mac_morpho.sents()[0])
# Bônus: Quer saber o que significa cada tag?

palavra_pos_taggeado = nltk.pos_tag(mac_morpho.sents()[0])

tags = set([tag for (texto, tag) in palavra_pos_taggeado])

for tag in tags:

    nltk.help.upenn_tagset(tag)
# Podemos acessar o corpus como uma sentença de palavras ou palavras taggeadas

from nltk.corpus import floresta

floresta.words()
# Palavras taggeadas

floresta.tagged_words()
# As tags consistem em alguma informação sintatica, seguida por um sinal "+", seguido por uma tag

# convencional de part-of-speech (a.k.a grammatical tagging, a.k.a. análise/classificação gramatical,

# a.k.a. noun, verb, article, adjective, preposition, pronoun, adverb, conjunction, and interjection,

# a.k.a. substantivo, verbo, artigo, adjetivo, preposição, pronome, adverbo, conjução e interjeição)



# OK, vamos cortar fora a parte antes do sinal de adição

def simplify_tag(t):

    if "+" in t:

        return t[t.index("+") + 1:]

    else:

        return t



twords = floresta.tagged_words()

twords = [(w.lower(), simplify_tag(t)) for (w, t) in twords]

twords[:10]
# Pretty print?

print('\n'.join(word + ' = ' + tag for (word, tag) in twords[:10]))
# Vamos contar a quantidade de tokens e tipos

words = floresta.words()

print(f"quantidade de palavras: {len(words)}")



# Inspecionar a distribuição de palavras

fd = nltk.FreqDist(words)

print(f"distribuição de palavras: {len(fd)}")



# A palavra mais frequente

print(f"palavra mais frequente: {fd.max()}")
# Quais são as 20 tags mais comuns, em ordem decrescente de frequencia?

tags = [simplify_tag(tag) for (word,tag) in floresta.tagged_words()]

fd = nltk.FreqDist(tags)

fd_keys = [x for x in fd.keys()]

fd_keys[:20]
# É possível também acessar o corpus agrupado por sentença

floresta.sents()
# Dá para fazer isso com a sentença taggeada

floresta.tagged_sents()
# Sentença parseada, como uma árvore

floresta.parsed_sents()
# Vamos ver isso mais bonitinho -- Meh. It fails so bad

from IPython.core.display import display

psents = floresta.parsed_sents()

display(psents)
raw_text = machado.raw('romance/marm05.txt')

raw_text[:200]
# Vamos criar uma função que, dado uma palavra e um tamanho de contexto (em número de caracteres),

# gere a concordância da tal palavra dentro de um conjunto de sentenças

def concordance(word, context=30):

    print(f"palavra: {word}, contexto: {context} caracteres")

    for sent in floresta.sents():

        if word in sent:

            pos = sent.index(word)

            left = " ".join(sent[:pos])

            right = " ".join(sent[pos + 1:])

            print(f"{left[-context:]} '{word}' {right[:context]}")



concordance("dar")
# Mais uma? Que tal vender?

concordance("vender")
# OK, vamos primeiro pegar as tags das sentenças e simplificá-las, como já fizemos anteriormente

from nltk.corpus import floresta

tsents = floresta.tagged_sents()

tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]
# Então criamos duas listas de sentenças, um para trainamento e outra para teste

train_tsents = tsents[100:]

test_tsents = tsents[:100]



print(f"train_tsents: {len(train_tsents)}")

print(f"test_tsents: {len(test_tsents)}")
# Sabemos que "n" é a tag mais comum, então podemos settar uma tag como default que marque todas as

# palavras como um substantivo e ver como isso performa

# NOTA: Por conta das mudanças na API do NLTK, o exemplo do livro precisa ser reescrito...

from nltk import DefaultTagger

tagger0 = DefaultTagger("n")

tagger0.evaluate(test_tsents)
# Evidentemente, 1 em 6 palavras são substantivos. Vamos melhorar isso então treinando um taggeador do

# tipo unigram

from nltk import UnigramTagger

tagger1 = UnigramTagger(train_tsents, backoff=tagger0)

tagger1.evaluate(test_tsents)
# E agora, para finalizar, um taggeador bigram

from nltk import BigramTagger

tagger2 = BigramTagger(train_tsents, backoff=tagger1)

tagger2.evaluate(test_tsents)
# Bônus: Vamos fazer uma avaliação de acuracidade com TnT (Trigrams'n'Tag), que é bom, mas ainda é um

# pouco menos performatico que o BigramTagger

from nltk.tag import tnt

tnt_pos_tagger = tnt.TnT()

tnt_pos_tagger.train(train_tsents)

tnt_pos_tagger.evaluate(test_tsents)
# Punkt é uma ferramenta de segumentação de linguagem-neutra

from nltk import data

sent_tokenizer = data.load("tokenizers/punkt/portuguese.pickle")

raw_text = machado.raw("romance/marm05.txt")

sentences = sent_tokenizer.tokenize(raw_text)

for sent in sentences[1000:1005]:

    print("<<", sent, ">>")
# Um sentence tokenizer pode ser treinado para ser aplicado em outros textos. O Floresta Portuguese

# Treebank contem uma sentença por linha, então vamos convenientemente usá-lo para criar dois textos,

# um de treinamento e um de teste

import os, nltk.test



text = floresta.raw()

lines = text.split('\n')

train_text = " ".join(lines[20:])

test_text = " ".join(lines[:20])



print(f"total de linhas no arquivo: {len(lines)}")

print(f"train_text: {len(train_text)} caracteres")

print(f"test_text: {len(test_text)} caracteres")



# Agora, vamos treinar o seguimentador de sentenças, a.k.a. sentence tokenizer, e usá-lo em nosso

# texto de teste

from nltk import PunktSentenceTokenizer

stok = PunktSentenceTokenizer(train_text)



tok_text = stok.tokenize(test_text)

print(f"texto de teste segmentado: {len(tok_text)}")

tok_text
# NLTK já inclui um modelo treinado para a geração de setenças em português, que pode ser carregado

# a partir de um arquivo .pickle

stok = data.load("tokenizers/punkt/portuguese.pickle")

stok
# Stopwords do português

stopwords = nltk.corpus.stopwords.words('portuguese')

stopwords[:10]
# Podemos então usar isso para filtrar texto

words = [w.lower() for w in floresta.words() if w.lower() not in stopwords]

fd = nltk.FreqDist(words)

fd_keys = [x for x in fd.keys()]



print("distribuição de frequência das palavras que não são stopwords:")

for word in fd_keys[:20]:

    print(f"{word} = {fd[word]}")