def occurences(str):

    freq = {}

    for i in str:

        if i in freq:

            freq[i] += 1

        else:

            freq[i] = 1

    return freq
occurences("si ton tonton tond mon tonton")
import numpy as np



def nb_lettres(str):

    nb_lettres = 0

    for i in str:

        nb_lettres += 1

    return nb_lettres



def entropie(str):

    total = nb_lettres(str)

    occ = occurences(str)

    p = 0

    for i in occ.keys():

        p += (occ.get(i)/total)*np.log2(occ.get(i)/total)

    return -p
entropie("si ton tonton tond mon tonton")
entropie ("un chasseur sachant chasser")
def min_letter(dico):

    m = [None, None]

    for cle,value in dico.items():

        if m == [None, None]:

            m[0] = cle

            m[1] = value

        elif m[1] >= value:

            m[0] = cle

            m[1] = value

    return m
min_letter(occurences("si ton tonton tond mon tonton"))
def min_letter_del(dico):

    m = min_letter(dico)

    del dico[m[0]]

    return m
test_minletterdel = occurences("si ton tonton tond mon tonton")
min_letter_del(test_minletterdel)
min_letter_del(test_minletterdel)
def max_letter(dico):

    m = [None, None]

    for cle,value in dico.items():

        if m == [None, None]:

            m[0] = cle

            m[1] = value

        elif m[1] <= value:

            m[0] = cle

            m[1] = value

    return m
def max_letter_del(dico):

    m = max_letter(dico)

    del dico[m[0]]

    return m
# classe noeud

class Noeud:

    def __init__(self,fg,val,fd):

        self.fg = fg

        self.val = val

        self.fd = fd
# attache 2 noeuds

def attach(n1,n2):

    return Noeud(n1,[None,n1.val[1]+n2.val[1]],n2)



# insert e au bon endroit dans l

def insert(l,e):

    i = 0

    while i < len(l):

        if l[i].val[1] > e.val[1]:

            l.insert(i,e)

            return l

        i += 1

    l.insert(len(l),e)



# construit arbre optimal

def optimal_tree(txt):

    dico = occurences(txt)

    occ = []



    # remplissage du tableau

    while len(dico) > 0:

        element = min_letter_del(dico)

        occ.append(Noeud(None,[element[0],element[1]],None))

    

    # construction de l'arbre

    while len(occ) > 1:

        n1 = occ[0]

        occ.pop(0)

        n2 = occ[0]

        occ.pop(0)

        nn = attach(n2,n1)

        insert(occ,nn)

    

    # retourne le seul element restant

    return occ[0]
tree1 = optimal_tree("si ton tonton tond mon tonton")

print(tree1.val)
tree2 = optimal_tree("un chasseur sachant chasser")

print(tree2.val)
def nb_letters(t):

    if t.fg == None and t.fd == None:

        return 1

    else:

        return nb_letters(t.fd)+nb_letters(t.fg)



def av_bits(t):

    return t.val[1]/nb_letters(t)
av_bits(tree1)
av_bits(tree2)
def rec_dico(tree, code, dico):

    if tree.fg == None and tree.fd == None:

        dico[tree.val[0]] = code

        code = ''

    else:

        rec_dico(tree.fg, code+'0', dico)

        rec_dico(tree.fd, code+'1', dico)



def extract_dico(txt):

    dico = {}

    rec_dico(optimal_tree(txt),'',dico)

    return dico
extract_dico("si ton tonton tond mon tonton")
def encode(txt):

    dico = extract_dico(txt)

    cmp_txt = ''

    for i in txt:

        cmp_txt += dico.get(i)

    return cmp_txt
encode("si ton tonton tond mon tonton")
def decode(txt, tree):

    r = ''

    t = tree

    for i in txt:

        if t.fg == None and t.fd == None:

            r += t.val[0]

            t = tree

        if i == '0':

            t = t.fg

        else:

            t = t.fd

    if t.fg == None and t.fd == None:

            r += t.val[0]

    

    return r
text_test = "si ton tonton tond mon tonton"

tree_test = optimal_tree(text_test)

encode_test = encode(text_test)

decode(encode_test, tree_test)