!pip install stanfordnlp
!pip install conllu
import stanfordnlp

import warnings

from conllu import parse
warnings.simplefilter('ignore')
stanfordnlp.download('es', force=True)
textcorpus = ''



corpuspath = '/kaggle/input/{}.txt'



for i in range(1, 202):

    textcorpus += (open(corpuspath.format(i)).read().split('Contenido:')[-1])
textcorpus[:100]
textcorpus = '_ _ ' + textcorpus.replace('.', ' _ _')

textcorpus = textcorpus.replace(',', ' _ _ ')
nlp = stanfordnlp.Pipeline(processors='tokenize,lemma,pos',lang='es')
%%time

doc = nlp(textcorpus)

doc.write_conll_to_file('/kaggle/working/news_corpus.conll')
%%time

with open('/kaggle/working/news_corpus.conll') as fd:

    sentences = parse(fd.read())
def get_tagkey(word):

    txtkey = ''

    

    try:

        txtkey += word['upostag']

        

        for feat in word['feats'].values():

            txtkey += ' ' + feat

    except:

        txtkey = '_'

        

    return txtkey
def get_trigrams(sentences):

    # Lista para almacenar los trigramas

    trigrams = []

    tag_trigrams = []

    

    # Recorrer la lista de palabras

    for sent in sentences:

        for i in range(len(sent)):

            try:

                w1, w2, w3 = sent[i:i+3]

            except:

                try:

                    w1, w2 = sent[i:i+2]

                    w3 = {'form': '.'}

                except:

                    w1 = sent[i]

                    w2 = w3 = {'form': '_'}

            

            # Agregar trigrama a la lista

            trigrams.append((w1['form'], w2['form'], w3['form']))

            

            tag_trigrams.append((

                get_tagkey(w1),

                get_tagkey(w2),

                get_tagkey(w3)

            ))

    

    return trigrams, tag_trigrams
%%time

word_trigrams, tag_trigrams = get_trigrams(sentences)
word_trigrams[:10]
tag_trigrams[:10]
def build_model(trigrams):

    model = {}

    

    # Contamos la frecuencia de co-ocurrencia

    for i in range(len(trigrams)):

        w1, w2, w3 = trigrams[i]

    

        # El control de excepciones se encarga de manejar los distintos casos 

        # en que un trigrama aún no ha sido registrado.

        try:

            model[w1, w2][w3] += 1

        except: # Aqui se asume que w3 lanza la excepcion

            try:

                model[w1, w2][w3] = 1

            except: # Aqui se asume que el par (w1, w2) lanza la excepcion

                model[w1, w2] = {w3:1}

            

    # Ahora transformamos el conteo en probabilidades

    for w1_w2 in model:

        total_count = float(sum(model[w1_w2].values()))

    

        for w3 in model[w1_w2]:

            model[w1_w2][w3] /= total_count

            

    return model
word_model = build_model(word_trigrams)
tag_model = build_model(tag_trigrams)
len(word_model)
len(tag_model)
word_min = 1



for i in word_model.values():

    for ii in i.values():

        if ii < word_min:

            word_min = ii

            

word_min
tag_min = 1



for i in tag_model.values():

    for ii in i.values():

        if ii < tag_min:

            tag_min = ii

            

tag_min
def edits1(word):

    "Distancia de edición 1"

    letters = 'abcdefghijklmnñopqrstuvwxyzáéíóú'

    

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    

    deletes = [L + R[1:] for L, R in splits if R]

    

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    

    replaces = [L +  C + R[1:] for L, R in splits if R for C in letters]

    

    inserts = [L + C + R for L, R in splits for C in letters]

    

    return set(deletes + transposes + replaces + inserts)
def edits2(word):

    "Distancia de edicion 2"

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
def get_dictionary():

    all_words = []

    

    for sent in sentences:

        for word in sent:

            all_words.append(word['form'].lower())

            all_words.append(word['lemma'])

    

    return set(all_words)
%%time

dictionary = get_dictionary()
def known(words):

    "Subconjunto de palabras que aparecen en el diccionario"

    

    return set(w for w in words if w in dictionary)
def suggestions(word):

    "Generate possible spelling corrections for word"

    

    word = word.lower()

    

    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
def P_Word(n_2, n_1, n):

    try:

        return word_model[(n_2, n_1)][n]

    except:

        try:

            return min(word_model[(n_2, n_1)].values())

        except:

            return word_min
def P_Tag(n_2, n_1, n):

    try:

        return tag_model[(n_2, n_1)][n]

    except:

        try:

            return min(tag_model[(n_2, n_1)].values())

        except:

            return tag_min
def P(n_2, n_1, n):

    sent = parse(nlp('{} {} {}'.format(n_2, n_1, n)).conll_file.conll_as_string())

    

    w1, w2, w3 = sent[0][0:3]

    

    return P_Word(w1['form'], w2['form'], w3['form']) * P_Tag(get_tagkey(w1), get_tagkey(w2), get_tagkey(w3))
def get_correction(badtext):

    badtext = '_ _ ' + badtext.replace('.', ' _ _ ')

    badtext = badtext.replace(',', ' _ _ ')

    

    doctest = nlp(badtext)



    listsuggestions = []



    for sent in doctest.sentences:

        for word in sent.words:

            listsuggestions.append(suggestions(word.text))



    final_text = []



    for suggestion in listsuggestions:

        if len(suggestion) == 1:

            final_text.append(list(suggestion)[0])

        else:

            maxprob = 0

            maxword = ''

        

            for word in suggestion:

                if len(final_text) >= 2:

                    w1 = final_text[-2]

                    w2 = final_text[-1]

                

                    tp = P(w1, w2, word)

                

                    if tp > maxprob:

                        maxprob = tp

                        maxword = word

                    

            final_text.append(maxword)

            

    return final_text, listsuggestions
%%time

test_text = """

Hay canbio de planes.

La probinsia orienttal.

El intrnet de las cosas.

La onra del artizta.

El camino del merkado.

La defenza de nuestros ideales.

Todo está en saver conbibir.

"""



text, listsuggestion = get_correction(test_text)
final_text = ''



for w in text:

    final_text += w + ' '

    

final_text
listsuggestion