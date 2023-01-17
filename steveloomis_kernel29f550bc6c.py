def statestring(start_letter,abbrevs_left):

    eligibles=[a for a in abbrevs_left if a[0]==start_letter]

    #print(eligibles)

    branchlist=[]

    for e in eligibles:

        #print(e)

        #print([a for a in abbrevs_left if a != e])

        newbranch=statestring(e[1],[a for a in abbrevs_left if a != e])

        for n in newbranch:

            #print(f"new branch {n}")

            branchlist.append(e[1]+n)

        #print(branchlist)

    if branchlist==[]:branchlist=['']

    #print(f"returning {branchlist}")

    return branchlist
abbrevs=['AL','AK','AS','AZ','AR','CA','CO','CT','DE','DC','FM','FL','GA','GU','HI','ID','IL','IN','IA','KS','KY','LA','ME','MH','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','MP','OH','OK','OR','PW','PA','PR','RI','SC','SD','TN','TX','UT','VT','VI','VA','WA','WV','WI','WY']

stringlist=[]

max_length=0

max_string=''

first_letters=[a[0] for a in abbrevs]

for f in first_letters:

    f_tree=statestring(f,abbrevs)

    for branch in f_tree:

        full_string=f+branch

        if len(full_string)>max_length:

            max_length=len(full_string)

            max_string=full_string

        stringlist.append(full_string)



        

    

print(f"Longest string is {max_string} at {max_length} characters.")



stringlengths=[len(s) for s in stringlist]

len(stringlengths)
import collections

collections.Counter(stringlengths)
stringlist31=[s for s in stringlist if len(s)==31]

for x in range(10):

    print(stringlist31[x])
for x in range(19):

    print(stringlist31[1000*x])
starting_twos=[s[0:2] for s in stringlist31]

set_of_starting_twos=set(starting_twos)

set_of_starting_twos
x=6

starting_x=[s[0:x] for s in stringlist31]

set(starting_x)
x=6

ending_x=[s[-x:] for s in stringlist31]

set(ending_x)

alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

first_letters=[code[0] for code in abbrevs]

second_letters=[code[1] for code in abbrevs]

cfl=collections.Counter(first_letters)

csl=collections.Counter(second_letters)

maximum_uses={letter:min(cfl[letter],csl[letter]) for letter in alphabet}

print(maximum_uses)

sum(maximum_uses.values())
code_appears=[]

for sl in stringlist31:

    for x in range(30):

        code_appears.append(sl[x:x+2])

collections.Counter(code_appears)