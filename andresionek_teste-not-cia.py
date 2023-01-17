from collections import Counter

file = open('../input/noticia.txt', encoding='utf-8').read()
words = file.split()
dwords = dict(Counter(words))


def is_possible_recursive(phrase):
    word = phrase.copy().pop()
    if word not in dwords.keys():
        return False
    elif dwords[word] < 1:
        return False
    elif len(phrase) == 0:
        return True
    else:
        dwords[word] -= 1
        return is_possible_recursive(phrase)
    
    
def is_possible_loop(phrase):
    for word in phrase:
        if word not in dwords.keys():
            return False
        elif dwords[word] < 1:
            return False
        else:
            dwords[word] -= 1
    return True
# Frase existente
PHRASE = 'Flávio não tem inteligência'.split()
%%timeit
is_possible_recursive(PHRASE)
%%timeit
is_possible_loop(PHRASE)
# Frase palavra não existente
PHRASE = 'Frávio não tem inteligência'.split()
%%timeit
is_possible_recursive(PHRASE)
%%timeit
is_possible_loop(PHRASE)
# Frase palavra esgotada
PHRASE = 'Flávio não tem inteligência inteligência inteligência inteligência inteligência'.split()
%%timeit
is_possible_recursive(PHRASE)
%%timeit
is_possible_loop(PHRASE)
