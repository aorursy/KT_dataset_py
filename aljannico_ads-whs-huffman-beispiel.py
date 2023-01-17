## Mini-Huffman

text = "ICH BIN KEIN BERLINER"



count = {} # Leeres Dictionary zum Zählen der Buchstaben



# Zählen der Vorkommen

# ... am None können wir erkennen, dass wir ein Blatt vor uns haben

for c in text:

    if c in count:

        count[c][0] += 1

    else:

        count[c] = [1,None]



print("Die Tabelle mit den Häufigkeiten:")                

print(count)

save_count = count.copy() # save to use for calculating code length later



# Zwei PQUEUE-Operationen

# In "schön" macht man das Objektorientiert und mit Heap!

def delete_min(pq):

    # Annahme: Die PriorityQueue pq ist nicht leer

    min_elem = None

    min_value = None

    min_children = None

    for key in pq:

        value,children = pq[key]

        if min_elem == None or min_value > value:

            min_elem = key

            min_value = value

            min_children = children

    del pq[min_elem]

    return min_elem,min_value,min_children



def insert(pg,element):

    key,value,children = element

    pg[key] = [value,children]



debug = False



# Wir benutzen jetzt direkt count als PriorityQueue (nicht nachmachen in einer echten Implementierung)

while len(count) >= 2:

    kleinstes1 = delete_min(count)

    kleinstes2 = delete_min(count)

    neues = (

        kleinstes1[0]+kleinstes2[0], # Key

        kleinstes1[1]+kleinstes2[1], # Wert

        (kleinstes1,kleinstes2)) # Kinder "links", "rechts"

    insert(count,neues)

    if debug: print(kleinstes1,"und",kleinstes2,"werden zu",neues)



# Am Ende verbleibt noch ein Element in count, nämlich die Wurzel des Baumes,

# die zudem den Baum enthält...cool, oder?



# Die Wurzel holen wir raus (count ist jetzt leer)

root = delete_min(count)

assert(len(count)==0) # Wir sichern zu, dass nichts mehr in der PQUEUE ist

# Wäre das verletzt, gäbe es eine Exception



print("\nDer Baum:")

import pprint

pprint.pprint(root)

    

# Jetzt konstruieren wir mal das Codebuch aus dem Baum



codes = {}



def codify(node,weg):

    key,value,children = node

    if children == None:

        # wir haben ein Blatt

        codes[key] = weg

    else:

        left,right = children

        codify(left,weg+'0')

        codify(right,weg+'1')



codify(root,'')

print("\nDas Codebuch:")

print(codes)



# Na, die Funktion ist cool, oder? Rekursiver Abstieg mit

# "left-first" Tiefensuche! Magie... ;)



# Jetzt können wir das Wort kodieren...

code = ''

for c in text:

    code = code+codes[c]



print("\nDer Code:",code)



# und die Länge durch Abzählen bestimmen

print("\nLänge des Codes in Bits:",len(code))



# ... oder auch berechnen:

bit_count = 0

for c in save_count:

    # Anzahl Vorkommen von Buchstabe c

    anzahl,_ = save_count[c]

    # * Codelänge für Buchtstabe c

    bit_count += anzahl * len(codes[c])



print("Berechnete Länge des Codes in Bits:",bit_count)