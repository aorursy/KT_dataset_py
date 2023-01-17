import pandas as pd

import string

import markovify
hhgroups = pd.read_csv('../input/hhgroups_merge_28_05.csv')

hhgroups.head()
letras = list(hhgroups['letra'])



# Eliminamos canciones sin letra

for letra in list(letras):

    if "¿Tienes ya la letra para este tema? Ayúdanos y ¡Envíanosla!" in letra:

        letras.remove(letra)



with open('corpus_markov_hhgroups.txt', 'wb') as file:

    for i in range(len(letras)):

        for linea in letras[i].split("\n"):

            if ("[" not in linea and "(" not in linea and linea != ""):

                 file.write((linea + "\n").encode("utf-8"))
with open('corpus_markov_hhgroups.txt', encoding="utf-8") as f:

    text = f.read()

    model1 = markovify.NewlineText(text, state_size=1)

    model2 = markovify.NewlineText(text, state_size=2)

    model3 = markovify.NewlineText(text, state_size=3)
def mapavocal(frase):

    palabra = frase.split(" ")[-1:][0]

    voc = "aeiouyáéíóú"

    s = ""

    for letra in palabra:

        if letra in voc:

            s += letra

    return s



def generarFraseRima(frase, model):

    mv = mapavocal(frase)

    frase_temp = None

    while frase_temp == None:

        frase_temp = model.make_sentence()

    mv2 = mapavocal(frase_temp)

    while mv2 == "" or (mv[-3:] != mv2[-3:] and mv[-2:] != mv2[-2:] and mv != mv2):

        frase_temp = None

        while frase_temp == None:

            frase_temp = model.make_sentence()

        mv2 = mapavocal(frase_temp)

    return frase_temp



def rapear(barras, model):

    frase = model.make_sentence()

    print(frase)

    for i in range(barras-1):

        print(generarFraseRima(frase,model))



def responder(frase, model, barras):

    print(frase)

    for i in range(barras-1):

        print(generarFraseRima(frase, model))

for i in range(4):

    print(model1.make_sentence())
for i in range(4):

    print(model2.make_sentence())
for i in range(4):

    print(model3.make_sentence())
responder("Kaggle es interesante", model1, 4)
responder("Este bot rapea un poco raro", model2, 4)
responder("Rapeas bastante mejor que muchos raperos", model3, 4)
rapear(4, model3)