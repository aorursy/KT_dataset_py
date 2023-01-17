from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

morph = MorphAnalyzer()
to_ud = converters.converter('opencorpora-int', 'ud14')

def convert_from_opencorpora_tag(tag, text):
    ud_tag = to_ud(str(tag), text)
    pos = ud_tag.split()[0]
    gram = ud_tag.split()[1]
    return pos, gram

def write_sentence_prediction(file, indices, sentence):
    for i, word in enumerate(sentence):
        form = morph.parse(word)[0]
        pos, gram = convert_from_opencorpora_tag(form.tag, word)
        file.write(str(indices[i]) + "," + pos + "#" + gram + "\n")

def predict(test_filename, submition_filename):
    with open(test_filename, "r") as r:
        next(r)
        with open(submition_filename, "w") as w:
            w.write("Id,Prediction\n")
            sentence = []
            indices = []
            for line in r:
                if len(line.strip()) == 0:
                    if len(sentence) == 0:
                        continue
                    write_sentence_prediction(w, indices, sentence)
                    sentence = []
                    indices = []
                    continue
                index = int(line.strip().split("\t")[0])
                indices.append(index)
                word = line.strip().split("\t")[2]
                sentence.append(word)
            if len(sentence) != 0:
                write_sentence_prediction(w, indices, sentence)
predict("../input/test.csv", "submition.csv")