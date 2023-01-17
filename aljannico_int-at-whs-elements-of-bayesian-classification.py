# Erstmal lesen wir die Daten ein, ja ja, mit Pandas ist das alles

# einfacher, aber machen wir ruhig mal alles "zu Fuß" :)



data = [

 ('Deadline?','Party?   ','faul?    ','Aktivität'),

 ('Dringend','Ja','Ja','Party'),

 ('Dringend','Nein','Ja','Lernen'),

 ('Bald','Ja','Ja','Party'),

 ('Nein','Ja','Nein','Party'),

 ('Nein','Nein','Ja','Kneipe'),

 ('Nein','Ja','Nein','Party'),

 ('Bald','Nein','Nein','Lernen'),

 ('Bald','Nein','Ja','TV'),

 ('Bald','Ja','Ja','Party'),

 ('Dringend','Nein','Nein','Lernen')

]



## Read the data

def attribute_values(data,pos):

    '''

        Collect all different values that occur in

        a given column and return them as a list.

        Note that duplicates are left out.

    '''

    vl = [] # we could use sets instead, but we want to index it later

    for d in data:

        if not (d[pos] in vl):

            vl.append(d[pos])

    return vl



# Strip the header of and keep it (as a copy)

attribute_names = data[0][:]

data = data[1:]



# Assume that the last attribute is the class to predict

classpos   = len(data[0])-1

classes    = attribute_values(data,classpos)

attvalues  = []

attributes = list(range(0,len(data[0])-1)) # assume non-empty data 

for att in attributes: 

    attvalues.append(attribute_values(data,att))



print("Domains of the attributes: ")

for i,att in enumerate(attvalues):

    print("   ",attribute_names[i],": ",att)



print("\nClasses: \n   ",attribute_names[classpos],": ",classes,"\n")
def build_named_P_from_data(data,classpos,classes,correct=False):

    '''

    Computes the distribution of probabilities for the given classes.

    DON'T CALL WITH EMPTY data list, as this makes no sense and will

    produce a runtime error anyway (and rightly so)

    '''

    # print(data,classpos,classes)

    p = {}  # List of probabilities, for each element in classes this will

            # contain a probability in the same position as the class in

            # the classes list

    count = {}

    n = float(len(data)) # Works also in Python 2

    for c in classes:

        count[c] = 0

    for d in data:

        count[d[classpos]] += 1

    for c in classes:

        if not correct:

            p[c] = count[c]/n

        else:

            p[c] = (count[c]+1)/(n+len(classes))

    return p





print("Prior probability distribution for the classes: \n")   

P_named_classes = build_named_P_from_data(data,classpos,classes)

print("\n",P_named_classes)
def show_data(data):

    for d in data:

        print("  ",d)



# Select tuples from some given data

def select_data(data,attribute,value):

    return [d for d in data if d[attribute] == value]



S_party_ja = select_data(data,classpos,'Party')

show_data(S_party_ja)

print("\nLikelihoods of  over shown dataset for Attribute",attribute_names[0],":",

      build_named_P_from_data(S_party_ja,0,attvalues[0]))



print("\nNow we want to correct it as done in the text above:")

print("Corrected Likelihoods of for Attribute",attribute_names[0],":",

      build_named_P_from_data(S_party_ja,0,attvalues[0],correct=True))



P_attribute_given_class = {}



print("\nAll likelihoods: \n")

# Compute it for all classes

for a in attributes:

    P_attribute_given_class[attribute_names[a]] = {}

    print("\n",attribute_names[a])

    for c in classes:

        p = build_named_P_from_data(select_data(data,classpos,c),a,attvalues[a],correct=True)           

        P_attribute_given_class[attribute_names[a]][c] = p

        print("  ",c,":",p)    
## Now, let's predict!



def P_class_given_sample(s):    

    global P_classes # has been determined already, see above

    p = {}

    p_result = []

    # P(Attribute=value|class) * ...

    for i,v in enumerate(s):

        for c in classes:

            p1 = P_attribute_given_class[attribute_names[i]][c][v]

            if not c in p: 

                p[c] = p1

            else:

                p[c] *= p1

                

    # ... * P(class)

    for c in classes:

        p[c] *= P_named_classes[c]

        p_result.append(p[c])

    return p_result # or return p for named results (as a dictionary)



s1 = ('Dringend','Nein','Nein')

s2 = ('Dringend','Ja','Nein')



print(classes)

print(s1,"-->",P_class_given_sample(s1))

print(s2,"-->",P_class_given_sample(s2))



# Argmax delivers in our case the index of the maximum entry in a list

# one could also use argmax from numpy ... or do it manually or

# with a shorthand requiring two passes: lst.index(max(lst))

def argmax(iterable):

    return max(enumerate(iterable), key=lambda x: x[1])[0]



# predict gives us ONE OF (it should be the first in the list)

# the classes (as an index value pointing into "classes")

# with the maximum probability.

# NOTE: It may be useful sometime to deliver a normalized

# probability distribution instead of a single class

def predict(cond_prob,debug=True):

    # Look for a maximum entry

    if debug: print("   ",cond_prob)

    return classes[argmax(cond_prob)]



print("\nNow let's make prediction for all samples we know already:")

for d in data:

    print(d,"->",predict(P_class_given_sample(d[0:classpos])))
## Wenden wir das auf die noch nie gesehenen unklassifizierten Samples an:

unknown_samples = [

    ('Dringend','Ja','Nein',None),

    ('Bald','Ja','Nein',None),

    ('Nein','Ja','Ja',None),

    ('Nein','Nein','Nein',None)

]



print(classes)

for d in unknown_samples:

    print(d,"->",predict(P_class_given_sample(d[0:classpos])))
## ignore

from IPython.display import display, Markdown, Latex

display(Markdown('*some markdown* $\phi$'))