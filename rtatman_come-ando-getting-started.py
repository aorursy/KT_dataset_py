# read in one of our files

with open("../input/Brazilian_Portugese_Corpus/Varias Historias.txt", "r") as f:

    text = f.readlines()

    

text[:5]
# read in one of our files but specificying the correct encoding!

with open("../input/Brazilian_Portugese_Corpus/Varias Historias.txt", "r", encoding='ISO8859_1') as f:

    text = f.readlines()

    

text[:5]