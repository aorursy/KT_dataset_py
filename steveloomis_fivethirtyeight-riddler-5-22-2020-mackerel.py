main_file_path = '../input/word.list.txt' 

wordlist=open(main_file_path,'r').read().split('\n')



statelist=['alabama','alaska','arizona','arkansas','california','colorado','connecticut','delaware','florida','georgia','Hawaii','idaho','illinois','indiana','iowa','kansas','kentucky',

           'louisiana','maine','maryland','massachusetts','michigan','minnesota','mississippi','missouri','montana','nebraska','nevada','newhampshire','newjersey','newmexico','newyork',

           'northcarolina','northdakota','ohio','oklahoma','oregon','pennsylvania','rhodeisland','southcarolina','southdakota','tennessee','texas','utah','vermont','virginia',

           'washington','westvirginia','wisconsin','wyoming']



def mackerel(word,state):

    wordset={x for x in word}

    stateset={x for x in state}

    is_mackerel=wordset.intersection(stateset)==set()

    return(is_mackerel)



print(mackerel('mackerel','ohio'))

print(mackerel('mackerel','kentucky'))

print(mackerel('goldfish','kentucky'))

print(mackerel('monkfish','delaware'))

print(mackerel('jellyfish','montana'))
def unique_mackerel(word,statelist):

    macklist=[state for state in statelist if mackerel(word,state)]

    if len(macklist)==1:

        return(macklist[0])



print(unique_mackerel('goldfish',statelist))

print(unique_mackerel('mackerel',statelist))

print(unique_mackerel('south',statelist))
counts={state:0 for state in statelist}

mackerels=[]

for word in wordlist:

    if unique_mackerel(word,statelist):

        ## print(f'{word} {unique_mackerel(word,statelist)}')

        mackerels.append((word,unique_mackerel(word,statelist)))

        counts[unique_mackerel(word,statelist)]+=1

mackerels       

lenmacks=[len(x[0]) for x in mackerels]

maxlen=max(lenmacks)

maxlenmacks=[x for x in mackerels if len(x[0])==maxlen]

print(maxlen)

print(maxlenmacks)





maxcount=max(counts.values())



{x:counts[x] for x in counts if counts[x]==maxcount}



[(m,s) for m,s in mackerels if s=='ohio']



max_by_state=[]

for state in statelist:

    lms=[len(x[0]) for x in mackerels if x[1]==state]

    if len(lms)>0:

        maxlms=max(lms)

        statemaxlenmacks=[x for x in mackerels if (len(x[0])==maxlms and x[1]==state)]

        max_by_state.append(statemaxlenmacks)                                           



import pprint

pprint.pprint(max_by_state)
minlen=min(lenmacks)

minlenmacks=[x for x in mackerels if len(x[0])==minlen]

print(minlen)

print(minlenmacks)

[(m,s) for m,s in mackerels if s=='kentucky']